import abc
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from run_modes import base_runner

torch.set_num_threads(torch.get_num_threads())

from teacher_student_dynamics import constants, experiments
from teacher_student_dynamics.curricula import (
    base_curriculum,
    hard_steps_curriculum,
    periodic_curriculum,
    threshold_curriculum,
)
from teacher_student_dynamics.networks import multi_head_network
from teacher_student_dynamics.networks.network_ensembles import (
    identical_ensemble,
    node_sharing_ensemble,
    rotation_ensemble,
)
from teacher_student_dynamics.utils import decorators


class BaseNetworkRunner(base_runner.BaseRunner, abc.ABC):
    """Runner for network simulations.

    Class for orchestrating student teacher framework including training
    and test loops.
    """

    def __init__(self, config: experiments.config.Config, unique_id: str = "") -> None:
        """
        Class constructor.

        Args:
            config: configuration object containing parameters to specify run.
        """
        # extract class-relevant attributes from config
        self._seed = config.seed
        self._device = config.experiment_device
        self._input_dimension = config.input_dimension
        self._checkpoint_frequency = config.checkpoint_frequency
        self._stdout_frequency = config.stdout_frequency
        self._checkpoint_path = config.checkpoint_path
        self._total_training_steps = config.total_training_steps
        self._test_frequency = config.test_frequency
        self._total_step_count = 0
        self._overlap_frequency = config.overlap_frequency or np.inf
        self._multi_head = config.multi_head
        self._replay_schedule = config.schedule
        self._replay_strategy = config.strategy

        # initialise student, teachers, logger_module,
        # data_module, loss_module, torch optimiser, and curriculum object
        self._teachers = self._setup_teachers(config=config)
        self._num_teachers = len(self._teachers.networks)
        self._student = self._setup_student(config=config)
        # self._logger = self._setup_logger(config=config)

        (
            self._data_module,
            self._test_data_inputs,
            self._test_teacher_outputs,
            self._label_noise_modules,
            self._input_noise_modules,
        ) = self._setup_data(config=config)

        self._loss_function = self._setup_loss(config=config)
        self._optimiser = self._setup_optimiser(config=config)
        self._curriculum = self._setup_curriculum(config=config)

        self._freeze_units = config.freeze_units
        self._unit_masks = self._setup_unit_masks(config=config)

        self._manage_network_devices()

        self._data_columns = self._setup_data_columns()
        self._log_columns = self._get_data_columns()

        abc.ABC.__init__(self)
        base_runner.BaseRunner.__init__(self, config=config, unique_id=unique_id)

    @abc.abstractmethod
    def get_network_configuration(self):
        """Get configuration of networks e.g. in terms of macroscopic order parameters.

        Can be used for both logging purposes and e.g. as input to ODE runner.
        """
        pass

    @decorators.timer
    @abc.abstractmethod
    def _setup_data(self, config: experiments.config.Config):
        """Prepare aspects related to the data e.g. train/test data."""
        pass

    @abc.abstractmethod
    def _training_step(self, teacher_index: int):
        """Perform single training step."""
        pass

    @abc.abstractmethod
    def _compute_generalisation_errors(self) -> List[float]:
        """Compute test errors for student with respect to all teachers."""
        pass

    def _get_data_columns(self):
        columns = [constants.TEACHER_INDEX, constants.LOSS]
        for i in range(self._num_teachers):
            columns.append(f"{constants.GENERALISATION_ERROR}_{i}")
            columns.append(f"{constants.LOG_GENERALISATION_ERROR}_{i}")
        if self._overlap_frequency < np.inf:
            sample_network_config = self.get_network_configuration()
            columns.extend(list(sample_network_config.sub_dictionary.keys()))
        return columns

    def _setup_data_columns(self):
        data_columns = {}
        for key in self._get_data_columns():
            arr = np.empty(self._checkpoint_frequency)
            arr[:] = np.NaN
            data_columns[key] = arr

        return data_columns

    def _checkpoint_data(self):
        log_dict = {k: self._data_columns[k] for k in self._log_columns}
        self._data_logger.logger_data = pd.DataFrame.from_dict(log_dict)
        self._data_logger.checkpoint()
        self._data_columns = self._setup_data_columns()
        self._data_index = 0

    @decorators.timer
    def _setup_teachers(self, config: experiments.config.Config):
        """Initialise teacher object containing teacher networks."""
        base_arguments = {
            constants.INPUT_DIMENSION: self._teacher_input_dimension,
            constants.HIDDEN_DIMENSION: config.teacher_hidden,
            constants.OUTPUT_DIMENSION: config.output_dimension,
            constants.ENSEMBLE_SIZE: config.num_teachers,
            constants.BIAS: config.teacher_bias,
            constants.NONLINEARITY: config.nonlinearity,
            constants.INITIALISATION_STD: config.teacher_initialisation_std,
            constants.NORMALISE_WEIGHTS: config.normalise_teachers,
            constants.UNIT_NORM_HEAD: config.unit_norm_teacher_head,
        }
        if config.teacher_configuration == constants.ROTATION:
            teachers_class = rotation_ensemble.RotationEnsemble
            additional_arguments = {
                constants.FEATURE_ROTATION_ALPHA: config.feature_rotation_alpha,
                constants.READOUT_ROTATION_ALPHA: config.readout_rotation_alpha,
            }
        elif config.teacher_configuration == constants.NODE_SHARING:
            teachers_class = node_sharing_ensemble.NodeSharingEnsemble
            additional_arguments = {
                constants.NUM_SHARED_NODES: config.num_shared_nodes,
                constants.FEATURE_ROTATION_ALPHA: config.feature_rotation_alpha,
            }
        elif config.teacher_configuration == constants.IDENTICAL:
            teachers_class = identical_ensemble.IdenticalEnsemble
            additional_arguments = {}
        else:
            raise ValueError(
                f"Teacher configuration '{config.teacher_configuration}' not recognised."
            )

        teachers = teachers_class(**base_arguments, **additional_arguments)

        save_path = os.path.join(
            config.checkpoint_path, constants.TEACHER_WEIGHT_SAVE_PATH
        )
        teachers.save_all_network_weights(save_path=save_path)

        return teachers

    @decorators.timer
    def _setup_student(self, config: experiments.config.Config):
        """Initialise object containing student network."""
        if self._multi_head:
            num_heads = self._num_teachers
        else:
            num_heads = 1
        return multi_head_network.MultiHeadNetwork(
            input_dimension=config.input_dimension,
            hidden_dimension=config.student_hidden,
            output_dimension=config.output_dimension,
            bias=config.student_bias,
            num_heads=num_heads,
            nonlinearity=config.nonlinearity,
            initialisation_std=config.student_initialisation_std,
            train_hidden_layer=config.train_hidden_layer,
            train_head_layer=config.train_head_layer,
        )

    @decorators.timer
    def _setup_optimiser(self, config: experiments.config.Config) -> torch.optim:
        """Initialise optimiser with trainable parameters of student."""
        (
            trainable_hidden_parameters,
            trainable_head_parameters,
        ) = self._student.get_trainable_parameters()

        # scale head parameter learning rates
        for param_set in trainable_head_parameters:
            param_set["lr"] = config.learning_rate / config.input_dimension

        trainable_parameters = trainable_hidden_parameters + trainable_head_parameters

        if config.optimiser == constants.SGD:
            optimiser = torch.optim.SGD(trainable_parameters, lr=config.learning_rate)
        else:
            raise ValueError(f"Optimiser type {config.optimiser} not recognised.")
        return optimiser

    @decorators.timer
    def _setup_unit_masks(self, config: experiments.config.Config) -> None:
        """setup masks for specific hidden units."""
        unit_masks = []
        for num_units in config.freeze_units:
            mask = torch.zeros(num_units, config.input_dimension)
            unit_masks.append(mask)
        return unit_masks

    @decorators.timer
    def _setup_loss(self, config: experiments.config.Config) -> Callable:
        """Instantiate torch loss function"""
        if config.loss_function == "mse":
            loss_function = nn.MSELoss()
        elif config.loss_function == "bce":
            loss_function = nn.BCELoss()
        else:
            raise NotImplementedError(
                f"Loss function {config.loss_function} not recognised."
            )
        return loss_function

    @decorators.timer
    def _setup_curriculum(
        self, config: experiments.config.Config
    ) -> base_curriculum.BaseCurriculum:
        """Initialise curriculum object (when to switch teacher,
        how to decide subsequent teacher etc.)

        Raises:
            ValueError: if stopping condition is not recognised.
        """
        if config.stopping_condition == constants.FIXED_PERIOD:
            curriculum = periodic_curriculum.PeriodicCurriculum(config=config)
        elif config.stopping_condition == constants.LOSS_THRESHOLDS:
            curriculum = threshold_curriculum.ThresholdCurriculum(config=config)
        elif config.stopping_condition == constants.SWITCH_STEPS:
            curriculum = hard_steps_curriculum.HardStepsCurriculum(config=config)
        else:
            raise ValueError(
                f"Stopping condition {config.stopping_condition} not recognised."
            )
        return curriculum

    @decorators.timer
    def _manage_network_devices(self) -> None:
        """Move relevant networks etc. to device specified in config (CPU or GPU)."""
        self._student.to(device=self._device)
        for teacher in self._teachers.networks:
            teacher.to(device=self._device)

    def _compute_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate loss of prediction of student vs. target from teacher

        Args:
            prediction: prediction made by student network on given input
            target: teacher output on same input

        Returns:
            loss: loss between target (from teacher) and prediction (from student)
        """
        # TODO: to make this more general, should the 0.5 be absorbed in to the loss fucntion?
        loss = 0.5 * self._loss_function(prediction, target)
        return loss

    def _log_step_data(self, step: int, logging_dict: Dict[str, Any]):
        for tag, scalar in logging_dict.items():
            self._data_logger.write_scalar(tag=tag, step=step, scalar=scalar)

    def train(self):

        self._data_index = 0

        while self._total_step_count <= self._total_training_steps:
            teacher_index, replaying = next(self._curriculum)

            print(self._total_step_count, teacher_index, replaying)

            self._train_on_teacher(teacher_index=teacher_index, replaying=replaying)

    def _train_on_teacher(self, teacher_index: int, replaying: Optional[bool] = None):
        """One phase of training (wrt one teacher)."""
        if self._multi_head:
            self._student.signal_boundary(new_head=teacher_index)
        else:
            self._student.signal_boundary(new_head=0)

        task_step_count = 0
        latest_generalisation_errors = [np.inf for _ in range(self._num_teachers)]

        timer = time.time()

        while self._total_step_count <= self._total_training_steps:

            generalisation_errors = self._train_test_step(
                teacher_index=teacher_index, replaying=replaying
            )

            latest_generalisation_errors = [
                generalisation_errors.get(
                    f"{constants.GENERALISATION_ERROR}_{i}",
                    latest_generalisation_errors[i],
                )
                for i in range(self._num_teachers)
            ]

            if self._total_step_count % self._stdout_frequency == 0:
                if self._total_step_count != 0:
                    self._logger.info(
                        f"Time for last {self._stdout_frequency} steps: {time.time() - timer}"
                    )
                    timer = time.time()
                self._logger.info(
                    f"Generalisation errors @ (~) step {self._total_step_count} "
                    f"({task_step_count}'th step training on teacher {teacher_index}): "
                )
                for i in range(self._num_teachers):
                    self._logger.info(
                        f"    Teacher {i}: {latest_generalisation_errors[i]}\n"
                    )

            self._total_step_count += 1
            task_step_count += 1
            self._data_index += 1

            if self._total_step_count % self._checkpoint_frequency == 0:
                self._checkpoint_data()

            if self._curriculum.to_switch(
                task_step=task_step_count,
                error=latest_generalisation_errors[teacher_index],
            ):
                break

    def _train_test_step(
        self, teacher_index: int, replaying: Optional[bool] = None
    ) -> Dict[str, Any]:
        self._training_step(teacher_index=teacher_index, replaying=replaying)

        if self._total_step_count % self._test_frequency == 0:
            generalisation_errors = self._compute_generalisation_errors()
        else:
            generalisation_errors = {}

        if self._total_step_count % self._overlap_frequency == 0:
            self.get_network_configuration()

        self._data_columns[constants.TEACHER_INDEX][self._data_index] = teacher_index

        return generalisation_errors
