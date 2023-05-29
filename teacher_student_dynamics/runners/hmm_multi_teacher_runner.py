from typing import Dict, List, Optional

import numpy as np
import torch

from teacher_student_dynamics import constants, experiments
from teacher_student_dynamics.data_modules import (
    base_data_module,
    hidden_manifold,
    iid_gaussian,
)
from teacher_student_dynamics.runners import base_network_runner
from teacher_student_dynamics.utils import network_configuration


class HMMMultiTeacherRunner(base_network_runner.BaseNetworkRunner):
    """Implementation of hidden manifold model with multiple teachers.

    Extension of Goldt et. al (https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.041044).
    """

    def __init__(self, config: experiments.config.Config, unique_id: str = "") -> None:

        self._teacher_input_dimension = config.latent_dimension
        self._latent_dimension = config.latent_dimension

        if config.strategy == constants.GAMMA:
            self._replay_gamma = config.gamma

        super().__init__(config, unique_id)
        self._logger.info("Setting up hidden manifold network runner...")

    def get_network_configuration(self):

        with torch.no_grad():

            student_head_weights = [
                head.weight.data.cpu().numpy().flatten() for head in self._student.heads
            ]
            teacher_head_weights = [
                teacher.heads[0].weight.data.cpu().numpy().flatten()
                for teacher in self._teachers.networks
            ]
            # W^{kl}
            student_self_overlap = self._student.self_overlap.cpu().numpy()
            # S^k_r
            student_weighted_feature_matrices = [
                self._student.layers[0].weight.data.mm(data_module.feature_matrix)
                / np.sqrt(self._input_dimension)
                for data_module in self._data_module
            ]
            # Sigma^{kl}
            student_weighted_feature_matrix_self_overlaps = [
                (
                    weighted_feature_matrix.mm(weighted_feature_matrix.t())
                    / self._latent_dimension
                )
                for weighted_feature_matrix in student_weighted_feature_matrices
            ]
            # Q^{kl}
            student_local_field_covariances = [
                (
                    (
                        data_module.folding_function_coefficients[2]
                        - data_module.folding_function_coefficients[0] ** 2
                        - data_module.folding_function_coefficients[1] ** 2
                    )
                    * student_self_overlap
                    + data_module.folding_function_coefficients[1] ** 2
                    * weighted_feature_matrix_self_overlap.cpu().numpy()
                )
                for data_module, weighted_feature_matrix_self_overlap in zip(
                    self._data_module, student_weighted_feature_matrix_self_overlaps
                )
            ]
            # T^{mn} and equivalent for second teacher
            teacher_self_overlaps = [
                teacher.self_overlap.cpu().numpy()
                for teacher in self._teachers.networks
            ]
            teacher_cross_overlaps = [
                o.cpu().numpy() for o in self._teachers.cross_overlaps
            ]
            student_teacher_overlaps = [
                data_module.folding_function_coefficients[1]
                * weighted_feature_matrix.mm(teacher.layers[0].weight.data.t())
                .cpu()
                .numpy()
                / self._latent_dimension
                for data_module, weighted_feature_matrix, teacher in zip(
                    self._data_module,
                    student_weighted_feature_matrices,
                    self._teachers.networks,
                )
            ]

            feature_matrix_overlaps = [
                data_module.feature_matrix.mm(data_module.feature_matrix.t())
                for data_module in self._data_module
            ]

        return network_configuration.HiddenManifoldNetworkConfiguration(
            student_head_weights=student_head_weights,
            teacher_head_weights=teacher_head_weights,
            student_self_overlap=student_self_overlap,
            teacher_self_overlaps=teacher_self_overlaps,
            teacher_cross_overlaps=teacher_cross_overlaps,
            student_teacher_overlaps=student_teacher_overlaps,
            student_weighted_feature_matrices=student_weighted_feature_matrices,
            student_local_field_covariances=student_local_field_covariances,
            student_weighted_feature_matrix_self_overlaps=student_weighted_feature_matrix_self_overlaps,
            feature_matrix_overlaps=feature_matrix_overlaps,
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
        if config.input_source == constants.HIDDEN_MANIFOLD:
            base_feature_matrix = torch.normal(
                mean=0.0,
                std=1.0,
                size=(self._input_dimension, self._latent_dimension),
                device=self._device,
            )
            data_modules = [
                hidden_manifold.HiddenManifold(
                    device=config.experiment_device,
                    train_batch_size=config.train_batch_size,
                    test_batch_size=config.test_batch_size,
                    input_dimension=config.input_dimension,
                    latent_dimension=config.latent_dimension,
                    mean=config.mean,
                    variance=config.variance,
                    activation=config.activation,
                    feature_matrix=base_feature_matrix,
                    precompute_data=config.precompute_data,
                )
            ]
            for feature_correlation in config.feature_matrix_correlations:
                random_feature_matrix = torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=(self._input_dimension, self._latent_dimension),
                    device=self._device,
                )
                next_feature_matrix = (
                    feature_correlation * base_feature_matrix
                    + np.sqrt(1 - feature_correlation**2) * random_feature_matrix
                )
                data_modules.append(
                    hidden_manifold.HiddenManifold(
                        device=config.experiment_device,
                        train_batch_size=config.train_batch_size,
                        test_batch_size=config.test_batch_size,
                        input_dimension=config.input_dimension,
                        latent_dimension=config.latent_dimension,
                        mean=config.mean,
                        variance=config.variance,
                        activation=config.activation,
                        feature_matrix=next_feature_matrix,
                        precompute_data=config.precompute_data,
                    )
                )
            if (
                self._replay_schedule is not None
                and self._replay_strategy == constants.GAMMA
            ):
                data_modules[0].surrogate_feature_matrices = [next_feature_matrix]
        else:
            raise ValueError(
                f"Data module (specified by input source) {config.input_source} not recognised"
            )

        # test data: get fixed sample from data module and generate labels from teachers.
        test_data = [data_module.get_test_data() for data_module in data_modules]
        test_data_inputs = [t[constants.X] for t in test_data]
        test_data_latents = [t[constants.LATENT] for t in test_data]

        test_teacher_outputs = self._teachers.forward_all_batches(test_data_latents)

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
            data_modules,
            test_data_inputs,
            test_teacher_outputs,
            label_noise_modules,
            input_noise_modules,
        )

    def _training_step(self, teacher_index: int, replaying: Optional[bool] = None):
        """Perform single training step."""

        precompute_labels_on = self._data_module[teacher_index].precompute_labels_on
        if precompute_labels_on is not None:
            self._data_module[teacher_index].add_precomputed_labels(
                self._teachers.forward(teacher_index, precompute_labels_on)
            )

        if replaying:
            if self._replay_strategy == constants.GAMMA:
                batch = self._data_module[teacher_index].get_mixed_batch(
                    gamma=self._replay_gamma, surrogate_index=0
                )
            else:
                batch = self._data_module[teacher_index].get_batch()
        else:
            batch = self._data_module[teacher_index].get_batch()

        batch_input = batch[constants.X]
        batch_latent = batch[constants.LATENT]

        input_noise_module = self._input_noise_modules[teacher_index]
        label_noise_module = self._label_noise_modules[teacher_index]

        if input_noise_module is None:
            student_batch_input = batch_input
        else:
            noise = input_noise_module.get_batch()
            noise_input = noise[constants.X]
            student_batch_input = batch_input + noise_input

        # forward through student network
        student_output = self._student.forward(student_batch_input)

        # forward through teacher network(s)
        teacher_output = batch.get(
            constants.Y, self._teachers.forward(teacher_index, batch_latent)
        )

        if label_noise_module is None:
            teacher_output = teacher_output
        else:
            noise = label_noise_module.get_batch()
            label_noise = noise[constants.X]
            teacher_output += label_noise

        # training iteration
        self._optimiser.zero_grad()
        loss = self._compute_loss(student_output, teacher_output)

        self._data_columns[constants.LOSS][self._data_index] = loss.item()

        loss.backward()

        self._optimiser.step()

    def _compute_generalisation_errors(self) -> List[float]:
        """Compute test errors for student with respect to all teachers."""
        self._student.eval()

        generalisation_errors = {}

        with torch.no_grad():
            student_outputs = self._student.forward_all_batches(self._test_data_inputs)

            for i, (student_output, teacher_output) in enumerate(
                zip(student_outputs, self._test_teacher_outputs)
            ):
                loss = self._compute_loss(student_output, teacher_output)
                self._data_columns[f"{constants.GENERALISATION_ERROR}_{i}"][
                    self._data_index
                ] = loss.item()
                self._data_columns[f"{constants.LOG_GENERALISATION_ERROR}_{i}"][
                    self._data_index
                ] = np.log10(loss.item())
                generalisation_errors[
                    f"{constants.GENERALISATION_ERROR}_{i}"
                ] = loss.item()

        self._student.train()

        return generalisation_errors
