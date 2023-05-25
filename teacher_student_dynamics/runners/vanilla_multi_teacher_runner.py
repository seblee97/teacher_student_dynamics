from typing import Dict, List, Optional

import numpy as np
import torch

from teacher_student_dynamics import constants, experiments
from teacher_student_dynamics.data_modules import base_data_module, iid_gaussian
from teacher_student_dynamics.runners import base_network_runner
from teacher_student_dynamics.utils import network_configuration


class VanillaMultiTeacherRunner(base_network_runner.BaseNetworkRunner):
    """Implementation of teacher-student model with multiple teachers.

    Used in Lee et. al 21, 22

    Extension of Saad & Solla (https://journals.aps.org/pre/abstract/10.1103/PhysRevE.52.4225).
    """

    def __init__(self, config: experiments.config.Config, unique_id: str = "") -> None:

        self._teacher_input_dimension = config.input_dimension

        super().__init__(config, unique_id)
        self._logger.info("Setting up vanilla teacher-student network runner...")

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
        label_noise_modules = []
        input_noise_modules = []

        for noise_spec in config.noise_to_student_input:
            if not noise_spec:
                input_noise_modules.append(None)
            else:
                mean, variance = noise_spec
                noise_module = iid_gaussian.IIDGaussian(
                    train_batch_size=config.train_batch_size,
                    test_batch_size=config.test_batch_size,
                    input_dimension=config.input_dimension,
                    mean=mean,
                    variance=variance,
                    dataset_size=config.dataset_size,
                )
                input_noise_modules.append(noise_module)

        for noise_spec in config.noise_to_teacher_output:
            if not noise_spec:
                label_noise_modules.append(None)
            else:
                mean, variance = noise_spec
                noise_module = iid_gaussian.IIDGaussian(
                    train_batch_size=config.train_batch_size,
                    test_batch_size=config.test_batch_size,
                    input_dimension=config.output_dimension,
                    mean=mean,
                    variance=variance,
                    dataset_size=config.dataset_size,
                )
                label_noise_modules.append(noise_module)

        return (
            data_module,
            test_data_inputs,
            test_teacher_outputs,
            label_noise_modules,
            input_noise_modules,
        )

    def _training_step(self, teacher_index: int, replaying: Optional[bool] = None):
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

        # mask units in first layer
        if self._freeze_units[teacher_index] > 0:
            for params in self._student.layers[0].parameters():
                for unit_index, unit in enumerate(
                    range(self._freeze_units[teacher_index])
                ):
                    params.grad[unit, :] = self._unit_masks[teacher_index][unit_index]

        self._optimiser.step()

        return training_step_dict

    def _compute_generalisation_errors(self) -> List[float]:
        """Compute test errors for student with respect to all teachers."""
        self._student.eval()

        generalisation_errors = {}

        with torch.no_grad():
            student_outputs = self._student.forward_all(self._test_data_inputs)

            if not self._multi_head:
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
