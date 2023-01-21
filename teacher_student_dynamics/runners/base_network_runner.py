import abc
from typing import Any, Callable, Dict

import torch
import torch.nn as nn
from run_modes import base_runner

from teacher_student_dynamics import constants, experiments
from teacher_student_dynamics.curricula import (
    base_curriculum,
    hard_steps_curriculum,
    periodic_curriculum,
    threshold_curriculum,
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

        self._manage_network_devices()

        abc.ABC.__init__(self)
        base_runner.BaseRunner.__init__(self, config=config, unique_id=unique_id)

    @abc.abstractmethod
    def get_network_configuration(self):
        """Get configuration of networks e.g. in terms of macroscopic order parameters.

        Can be used for both logging purposes and e.g. as input to ODE runner.
        """
        pass

    @abc.abstractmethod
    def _get_data_columns(self):
        pass

    @decorators.timer
    @abc.abstractmethod
    def _setup_teachers(self, config: experiments.config.Config):
        """Initialise teacher object containing teacher networks."""
        pass

    @decorators.timer
    @abc.abstractmethod
    def _setup_student(self, config: experiments.config.Config):
        """Initialise object containing student network."""
        pass

    @decorators.timer
    @abc.abstractmethod
    def _setup_optimiser(self, config: experiments.config.Config) -> torch.optim:
        """Initialise optimiser with trainable parameters of student."""
        pass

    @decorators.timer
    @abc.abstractmethod
    def _setup_data(self, config: experiments.config.Config):
        """Prepare aspects related to the data e.g. train/test data."""
        pass

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
