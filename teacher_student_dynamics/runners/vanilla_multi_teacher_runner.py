import os
import time
from typing import Any, Dict, List

import numpy as np
import torch

from teacher_student_dynamics import constants, experiments
from teacher_student_dynamics.data_modules import base_data_module, iid_gaussian
from teacher_student_dynamics.networks import multi_head_network
from teacher_student_dynamics.networks.network_ensembles import (
    identical_ensemble,
    node_sharing_ensemble,
    rotation_ensemble,
)
from teacher_student_dynamics.runners import base_network_runner
from teacher_student_dynamics.utils import network_configuration


class VanillaMultiTeacherRunner(base_network_runner.BaseNetworkRunner):
    """Implementation of teacher-student model with multiple teachers.

    Used in Lee et. al 21, 22

    Extension of Saad & Solla (https://journals.aps.org/pre/abstract/10.1103/PhysRevE.52.4225).
    """

    def __init__(self, config: experiments.config.Config, unique_id: str = "") -> None:
        super().__init__(config, unique_id)

    def get_network_configuration(self):
        with torch.no_grad():
            student_head_weights = [
                head.weight.data.cpu().numpy().flatten() for head in self._student.heads
            ]
            teacher_head_weights = [
                teacher.heads[0].weight.data.cpu().numpy().flatten()
                for teacher in self._teachers.networks
            ]
            student_self_overlap = self._student.self_overlap.cpu().numpy()
            teacher_self_overlaps = [
                teacher.self_overlap.cpu().numpy()
                for teacher in self._teachers.networks
            ]
            teacher_cross_overlaps = [
                o.cpu().numpy() for o in self._teachers.cross_overlaps
            ]
            student_layer = self._student.layers[0].weight.data
            student_teacher_overlaps = [
                student_layer.mm(teacher.layers[0].weight.data.t()).cpu().numpy()
                / self._input_dimension
                for teacher in self._teachers.networks
            ]

        return network_configuration.NetworkConfiguration(
            student_head_weights=student_head_weights,
            teacher_head_weights=teacher_head_weights,
            student_self_overlap=student_self_overlap,
            teacher_self_overlaps=teacher_self_overlaps,
            teacher_cross_overlaps=teacher_cross_overlaps,
            student_teacher_overlaps=student_teacher_overlaps,
        )

    def _get_data_columns(self):
        columns = [constants.TEACHER_INDEX, constants.LOSS]
        for i in range(self._num_teachers):
            columns.append(f"{constants.GENERALISATION_ERROR}_{i}")
            columns.append(f"{constants.LOG_GENERALISATION_ERROR}_{i}")
        return columns

    def _setup_teachers(self, config: experiments.config.Config):
        """Initialise teacher object containing teacher networks."""
        base_arguments = {
            constants.INPUT_DIMENSION: config.input_dimension,
            constants.HIDDEN_DIMENSION: config.teacher_hidden,
            constants.OUTPUT_DIMENSION: config.output_dimension,
            constants.ENSEMBLE_SIZE: config.num_teachers,
            constants.BIAS: config.teacher_bias,
            constants.NONLINEARITY: config.nonlinearity,
            constants.INITIALISATION_STD: config.teacher_initialisation_std,
            constants.NORMALISE_WEIGHTS: config.normalise_teachers,
            constants.UNIT_NORM_HEAD: config.unit_norm_teacher_head,
            constants.NOISE_STDS: config.teacher_output_noises,
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
                constants.FEATURE_ROTATION_MAGNITUDE: config.feature_rotation_magnitude,
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

    def _setup_student(self, config: experiments.config.Config):
        """Initialise object containing student network."""
        return multi_head_network.MultiHeadNetwork(
            input_dimension=config.input_dimension,
            hidden_dimension=config.student_hidden,
            output_dimension=config.output_dimension,
            bias=config.student_bias,
            num_heads=self._num_teachers,
            nonlinearity=config.nonlinearity,
            initialisation_std=config.student_initialisation_std,
            train_hidden_layer=config.train_hidden_layer,
            train_head_layer=config.train_head_layer,
        )

    def _setup_optimiser(self, config: experiments.config.Config) -> torch.optim:
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

    def _setup_data(
        self, config: experiments.config.Config
    ) -> base_data_module.BaseData:
        """This method prepares several aspects of the data.

            - Initialise train data module.
            - Construct a test dataset.
            - Construct noise module for student inputs.
            - Construct noise module for teacher outputs.

        This method must be called before training loop is called."""

        # core data module
        if config.input_source == constants.IID_GAUSSIAN:
            data_module = iid_gaussian.IIDGaussian(
                train_batch_size=config.train_batch_size,
                test_batch_size=config.test_batch_size,
                input_dimension=config.input_dimension,
                mean=config.mean,
                variance=config.variance,
                dataset_size=config.dataset_size,
            )
        else:
            raise ValueError(
                f"Data module (specified by input source) {config.input_source} not recognised"
            )

        # test data: get fixed sample from data module and generate labels from teachers.
        test_data_inputs = data_module.get_test_data()[constants.X].to(self._device)
        test_teacher_outputs = self._teachers.forward_all(test_data_inputs)

        # noise for outputs on teachers, noise for inputs to students.
        label_noise_modules = [None, None]
        input_noise_modules = [None, None]

        return (
            data_module,
            test_data_inputs,
            test_teacher_outputs,
            label_noise_modules,
            input_noise_modules,
        )

    def train(self):
        while self._total_step_count <= self._total_training_steps:
            teacher_index = next(self._curriculum)

            self._train_on_teacher(teacher_index=teacher_index)

    def _train_on_teacher(self, teacher_index: int):
        """One phase of training (wrt one teacher)."""
        self._student.signal_boundary(new_head=teacher_index)

        task_step_count = 0
        latest_generalisation_errors = [np.inf for _ in range(self._num_teachers)]

        timer = time.time()

        while self._total_step_count <= self._total_training_steps:

            if (
                self._total_step_count % self._checkpoint_frequency == 0
                and self._total_step_count != 0
            ):
                self._data_logger.checkpoint()

            self._total_step_count += 1
            task_step_count += 1

            step_logging_dict = self._train_test_step(teacher_index=teacher_index)

            latest_generalisation_errors = [
                step_logging_dict.get(
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

            if self._curriculum.to_switch(
                task_step=task_step_count,
                error=latest_generalisation_errors[teacher_index],
            ):
                break

            self._log_step_data(
                step=self._total_step_count, logging_dict=step_logging_dict
            )

    def _train_test_step(self, teacher_index: int) -> Dict[str, Any]:
        step_logging_dict = self._training_step(teacher_index=teacher_index)

        if self._total_step_count % self._test_frequency == 0:
            generalisation_errors = self._compute_generalisation_errors()
            step_logging_dict = {**step_logging_dict, **generalisation_errors}

        step_logging_dict[constants.TEACHER_INDEX] = teacher_index

        return step_logging_dict

    def _training_step(self, teacher_index: int):
        """Perform single training step."""

        training_step_dict = {}

        batch = self._data_module.get_batch()
        batch_input = batch[constants.X].to(self._device)

        input_noise_module = self._input_noise_modules[teacher_index]
        label_noise_module = self._label_noise_modules[teacher_index]

        if input_noise_module is None:
            student_batch_input = batch_input
        else:
            noise = input_noise_module.get_batch()
            noise_input = noise[constants.X].to(self._device)
            student_batch_input = batch_input + noise_input

        # forward through student network
        student_output = self._student.forward(student_batch_input)

        # forward through teacher network(s)
        teacher_output = self._teachers.forward(teacher_index, batch_input)

        if label_noise_module is None:
            teacher_output = teacher_output
        else:
            noise = label_noise_module.get_batch()
            label_noise = noise[constants.X].to(self._device)
            teacher_output += label_noise

        # training iteration
        self._optimiser.zero_grad()
        loss = self._compute_loss(student_output, teacher_output)

        training_step_dict[constants.LOSS] = loss.item()

        loss.backward()

        self._optimiser.step()

        return training_step_dict

    def _compute_generalisation_errors(self) -> List[float]:
        """Compute test errors for student with respect to all teachers."""
        self._student.eval()

        generalisation_errors = {}

        with torch.no_grad():
            student_outputs = self._student.forward_all(self._test_data_inputs)

            # meta student will only have one set of outputs from forward_all call
            if len(student_outputs) == 1:
                student_outputs = [
                    student_outputs[0] for _ in range(len(self._test_teacher_outputs))
                ]

            for i, (student_output, teacher_output) in enumerate(
                zip(student_outputs, self._test_teacher_outputs)
            ):
                loss = self._compute_loss(student_output, teacher_output)
                generalisation_errors[
                    f"{constants.GENERALISATION_ERROR}_{i}"
                ] = loss.item()
                generalisation_errors[
                    f"{constants.LOG_GENERALISATION_ERROR}_{i}"
                ] = np.log10(loss.item())

        self._student.train()

        return generalisation_errors
