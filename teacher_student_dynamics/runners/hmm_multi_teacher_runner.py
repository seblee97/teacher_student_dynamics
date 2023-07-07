import os
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
        # self._num_bins = config.num_bins
        self._delta = config.latent_dimension / config.input_dimension

        # RHO_MIN = (1 - np.sqrt(self._delta)) ** 2
        # RHO_MAX = (1 + np.sqrt(self._delta)) ** 2
        # self._rho_bins = np.linspace(RHO_MIN, RHO_MAX, self._num_bins + 1)

        if config.strategy == constants.GAMMA:
            self._replay_gamma = config.gamma

        super().__init__(config, unique_id)
        self._logger.info("Setting up hidden manifold network runner...")

    def get_network_configuration(self, update=True):

        with torch.no_grad():

            student_head_weights = [
                head.weight.data.cpu().numpy().flatten() for head in self._student.heads
            ]

            # W^{kl}
            student_self_overlap = self._student.self_overlap.cpu().numpy()
            # S^k_r (student weights projected onto manifold)
            student_weighted_feature_matrices = [
                self._student.layers[0].weight.data.mm(data_module.feature_matrix)
                / np.sqrt(self._input_dimension)
                for data_module in self._data_module
            ]
            # Sigma^{kl} (self overlap in the projected space)
            student_weighted_feature_matrix_self_overlaps = [
                (
                    weighted_feature_matrix.mm(weighted_feature_matrix.t())
                    / self._latent_dimension
                )
                for weighted_feature_matrix in student_weighted_feature_matrices
            ]
            # Q^{kl} (combination of ambient and latent self overlaps) [sigma not rotated]
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

            # R_{km} [not rotated]
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

            if not update:
                # T^{mn} and equivalent for second teacher
                teacher_self_overlaps = [
                    teacher.self_overlap.cpu().numpy()
                    for teacher in self._teachers.networks
                ]
                teacher_cross_overlaps = [
                    o.cpu().numpy() for o in self._teachers.cross_overlaps
                ]
                feature_matrix_overlaps = [
                    data_module.feature_matrix.t().mm(data_module.feature_matrix)
                    / self._input_dimension
                    for data_module in self._data_module
                ]
                teacher_head_weights = [
                    teacher.heads[0].weight.data.cpu().numpy().flatten()
                    for teacher in self._teachers.networks
                ]

                feature_matrix_eigenvectors = []
                feature_matrix_eigenvalues = []
                for overlap in feature_matrix_overlaps:
                    eigenvalues, eigenvectors = torch.linalg.eigh(overlap)
                    feature_matrix_eigenvalues.append(eigenvalues.to(torch.float))

                    # need to be careful here about normalisation/orientation condition
                    # of B17, below too.
                    feature_matrix_eigenvectors.append(
                        np.sqrt(self._latent_dimension) * eigenvectors.to(torch.float)
                    )

                # (Eq. B19) student weights projected onto eigenbasis of Omega
                gamma_tau_k = [
                    overlap.mm(eigenvectors) / np.sqrt(self._latent_dimension)
                    for overlap, eigenvectors in zip(
                        student_weighted_feature_matrices,
                        feature_matrix_eigenvectors,
                    )
                ]

                # (Eq. B20) teacher weights projected onto eigenbasis of Omega
                w_tilde_tau = [
                    teacher.state_dict()["_layers.0.weight"].mm(eigenvectors)
                    / np.sqrt(self._latent_dimension)
                    for teacher, eigenvectors in zip(
                        self._teachers.networks, feature_matrix_eigenvectors
                    )
                ]

                # (Eq. B29) teacher self-overlap in projected eigenbasis
                projected_teacher_self_overlaps = [
                    w_tilde.mm(torch.diag(eigenvalues).mm(w_tilde.T)).numpy()
                    / self._latent_dimension
                    for w_tilde, eigenvalues in zip(
                        w_tilde_tau, feature_matrix_eigenvalues
                    )
                ]
                # projected_teacher_self_overlaps = [
                #     (eigen_values[0].to(torch.float) * w_tilde).mm(w_tilde.T).numpy()
                #     / self._latent_dimension
                #     for w_tilde, eigen_values in zip(
                #         w_tilde_tau, feature_matrix_eigenspectra
                #     )
                # ]

                # (Eq. B31) density r_km
                student_teacher_overlap_densities = []
                student_hidden_dim = student_head_weights[0].shape[0]
                teacher_hidden_dim = teacher_head_weights[0].shape[0]

                for gamma, w_tilde in zip(gamma_tau_k, w_tilde_tau):
                    r_km = np.zeros(
                        shape=(
                            student_hidden_dim * teacher_hidden_dim,
                            self._latent_dimension,
                        )
                    )
                    for k in range(student_hidden_dim):
                        for m in range(teacher_hidden_dim):
                            r_km[k * teacher_hidden_dim + m, :] = (
                                gamma[k] * w_tilde[m]
                            ).numpy()
                    student_teacher_overlap_densities.append(r_km)
                # for task, eigenspectrum in enumerate(feature_matrix_eigenspectra):
                #     r_km = np.zeros(
                #         shape=(
                #             len(self._rho_bins) - 1,
                #             student_hidden_dim,
                #             teacher_hidden_dim,
                #         )
                #     )
                #     for i in range(len(r_km)):
                #         for tau, eigenvalue in enumerate(
                #             eigenspectrum[0].to(torch.float)
                #         ):
                #             if (
                #                 eigenvalue.item() < self._rho_bins[i + 1]
                #                 and eigenvalue.item() > self._rho_bins[i]
                #             ):
                #                 r_km[i] = (
                #                     gamma_tau_k[task][:, [tau]]
                #                     .mm(w_tilde_tau[task][:, [tau]].t())
                #                     .numpy()
                #                 )
                #     student_teacher_overlap_densities.append(
                #         r_km.reshape(-1, student_hidden_dim * teacher_hidden_dim).T
                #     )

                # (Eq. B47) density sigma_kl
                student_latent_self_overlap_densities = []
                for gamma in gamma_tau_k:
                    sigma_kl = np.zeros(
                        shape=(
                            student_hidden_dim**2,
                            self._latent_dimension,
                        )
                    )
                    for k in range(student_hidden_dim):
                        for l in range(student_hidden_dim):
                            sigma_kl[k * student_hidden_dim + l] = (
                                gamma[k] * gamma[l]
                            ).numpy()

                    student_latent_self_overlap_densities.append(sigma_kl)

                # for task, eigenspectrum in enumerate(feature_matrix_eigenspectra):
                #     sigma_kl = np.zeros(
                #         shape=(
                #             len(self._rho_bins) - 1,
                #             student_hidden_dim,
                #             student_hidden_dim,
                #         )
                #     )
                #     for i in range(len(sigma_kl)):
                #         for tau, eigenvalue in enumerate(
                #             eigenspectrum[0].to(torch.float)
                #         ):
                #             if (
                #                 eigenvalue.item() < self._rho_bins[i + 1]
                #                 and eigenvalue.item() > self._rho_bins[i]
                #             ):
                #                 sigma_kl[i] = (
                #                     gamma_tau_k[task][:, [tau]]
                #                     .mm(gamma_tau_k[task][:, [tau]].t())
                #                     .numpy()
                #                 )
                #     student_latent_self_overlap_densities.append(
                #         sigma_kl.reshape(-1, student_hidden_dim * student_hidden_dim).T
                #     )

                feature_matrix_eigenvalues = [
                    e.numpy()[np.newaxis, :] for e in feature_matrix_eigenvalues
                ]

            if not update:
                # rotated_student_teacher_overlaps = [
                #     data_module.folding_function_coefficients[1]
                #     * rotated_weighted_feature_matrix.mm(teacher_weights.t())
                #     .cpu()
                #     .numpy()
                #     / (self._latent_dimension)
                #     for data_module, rotated_weighted_feature_matrix, teacher_weights in zip(
                #         self._data_module, gamma_tau_k, w_tilde_tau
                #     )
                # ]

                rotated_student_teacher_overlaps = [
                    data_module.folding_function_coefficients[1]
                    * density.mean(1).reshape((student_hidden_dim, teacher_hidden_dim))
                    for data_module, density in zip(
                        self._data_module, student_teacher_overlap_densities
                    )
                ]
                # Sigma^{kl} rotated into eigenbasis (B43)
                # rotated_student_weighted_feature_matrix_self_overlaps = [
                #     (
                #         weighted_feature_matrix.mm(weighted_feature_matrix.t())
                #         / self._latent_dimension
                #     )
                #     for weighted_feature_matrix in gamma_tau_k
                # ]
                rotated_student_weighted_feature_matrix_self_overlaps = [
                    (
                        rotated_weighted_feature_matrix.mm(
                            rotated_weighted_feature_matrix.t()
                        )
                        / self._latent_dimension
                    ).numpy()
                    for rotated_weighted_feature_matrix in gamma_tau_k
                ]
            else:
                rotated_student_teacher_overlaps = [
                    data_module.folding_function_coefficients[1]
                    * overlap.mm(eigenvectors.T)
                    .mm(teacher.state_dict()["_layers.0.weight"].mm(eigenvectors.T).t())
                    .cpu()
                    .numpy()
                    / (self._latent_dimension**2)
                    for data_module, overlap, eigenvectors, teacher in zip(
                        self._data_module,
                        student_weighted_feature_matrices,
                        # self._network_configuration.student_weighted_feature_matrices,
                        self._network_configuration.feature_matrix_overlap_eigenvectors,
                        self._teachers.networks,
                    )
                ]
                # Sigma^{kl} rotated into eigenbasis (B43)
                rotated_student_weighted_feature_matrix_self_overlaps = [
                    (
                        overlap.mm(eigenvectors.T)
                        .mm(overlap.mm(eigenvectors.T).T)
                        .cpu()
                        .numpy()
                        / (self._latent_dimension**2)
                    )
                    for overlap, eigenvectors in zip(
                        self._network_configuration.student_weighted_feature_matrices,
                        self._network_configuration.feature_matrix_overlap_eigenvectors,
                    )
                ]
            # Q^{kl} (combination of ambient and latent self overlaps) [sigma rotated]
            rotated_student_local_field_covariances = [
                (
                    (
                        data_module.folding_function_coefficients[2]
                        - data_module.folding_function_coefficients[0] ** 2
                        - data_module.folding_function_coefficients[1] ** 2
                    )
                    * student_self_overlap
                    + data_module.folding_function_coefficients[1] ** 2
                    * weighted_feature_matrix_self_overlap
                )
                for data_module, weighted_feature_matrix_self_overlap in zip(
                    self._data_module,
                    rotated_student_weighted_feature_matrix_self_overlaps,
                )
            ]

        if not update:
            return network_configuration.HiddenManifoldNetworkConfiguration(
                student_head_weights=student_head_weights,
                teacher_head_weights=teacher_head_weights,
                student_self_overlap=student_self_overlap,
                teacher_self_overlaps=teacher_self_overlaps,
                teacher_cross_overlaps=teacher_cross_overlaps,
                student_teacher_overlaps=student_teacher_overlaps,
                rotated_student_teacher_overlaps=rotated_student_teacher_overlaps,
                student_weighted_feature_matrices=student_weighted_feature_matrices,
                student_local_field_covariances=student_local_field_covariances,
                rotated_student_local_field_covariances=rotated_student_local_field_covariances,
                student_weighted_feature_matrix_self_overlaps=student_weighted_feature_matrix_self_overlaps,
                rotated_student_weighted_feature_matrix_self_overlaps=rotated_student_weighted_feature_matrix_self_overlaps,
                feature_matrix_overlaps=feature_matrix_overlaps,
                feature_matrix_overlap_eigenvalues=feature_matrix_eigenvalues,
                feature_matrix_overlap_eigenvectors=feature_matrix_eigenvectors,
                student_teacher_overlap_densities=student_teacher_overlap_densities,
                student_latent_self_overlap_densities=student_latent_self_overlap_densities,
                projected_teacher_self_overlaps=projected_teacher_self_overlaps,
            )
        else:
            self._network_configuration.student_head_weights = student_head_weights
            self._network_configuration.student_self_overlap = student_self_overlap
            self._network_configuration.student_teacher_overlaps = (
                student_teacher_overlaps
            )
            self._network_configuration.rotated_student_teacher_overlaps = (
                rotated_student_teacher_overlaps
            )
            self._network_configuration.student_local_field_covariances = (
                student_local_field_covariances
            )
            self._network_configuration.rotated_student_local_field_covariances = (
                rotated_student_local_field_covariances
            )
            self._network_configuration.rotated_student_weighted_feature_matrix_self_overlaps = (
                rotated_student_weighted_feature_matrix_self_overlaps
            )
            self._network_configuration.student_weighted_feature_matrix_self_overlaps = (
                student_weighted_feature_matrix_self_overlaps
            )
            self._network_configuration.student_weighted_feature_matrices = (
                student_weighted_feature_matrices
            )

    def save_network_configuration(self, network_configuration, step: int):

        if step is not None:
            step = f"_{step}"
        else:
            step = ""

        order_params = {
            f"W{step}.csv": network_configuration.student_self_overlap,
            f"Sigma1{step}.csv": network_configuration.rotated_student_weighted_feature_matrix_self_overlaps[
                0
            ],
            f"Sigma2{step}.csv": network_configuration.rotated_student_weighted_feature_matrix_self_overlaps[
                1
            ],
            f"r_density{step}.csv": network_configuration.student_teacher_overlap_densities[
                0
            ],
            f"u_density{step}.csv": network_configuration.student_teacher_overlap_densities[
                1
            ],
            f"sigma_1_density{step}.csv": network_configuration.student_latent_self_overlap_densities[
                0
            ],
            f"sigma_2_density{step}.csv": network_configuration.student_latent_self_overlap_densities[
                1
            ],
            f"Q{step}.csv": network_configuration.rotated_student_local_field_covariances[
                0
            ],
            f"R{step}.csv": network_configuration.rotated_student_teacher_overlaps[0],
            f"U{step}.csv": network_configuration.rotated_student_teacher_overlaps[1],
            f"h1{step}.csv": network_configuration.student_head_weights[0],
        }
        if len(network_configuration.student_head_weights) > 1:
            # multi-head
            order_params["h2.csv"] = network_configuration.student_head_weights[1]

        if step == "":
            order_params = {
                **order_params,
                **{
                    f"th1{step}.csv": network_configuration.teacher_head_weights[0],
                    f"th2{step}.csv": network_configuration.teacher_head_weights[1],
                    f"T{step}.csv": network_configuration.teacher_self_overlaps[0],
                    f"H{step}.csv": network_configuration.teacher_self_overlaps[1],
                    f"T_tilde{step}.csv": network_configuration.projected_teacher_self_overlaps[
                        0
                    ],
                    f"H_tilde{step}.csv": network_configuration.projected_teacher_self_overlaps[
                        1
                    ],
                    f"V{step}.csv": network_configuration.teacher_cross_overlaps[0],
                    f"rho_1{step}.csv": network_configuration.feature_matrix_overlap_eigenvalues[
                        0
                    ],
                    f"rho_2{step}.csv": network_configuration.feature_matrix_overlap_eigenvalues[
                        1
                    ],
                },
            }

        order_param_path = os.path.join(
            self._ode_file_path, f"order_parameter{step}.txt"
        )

        with open(order_param_path, "+w") as txt_file:
            for k, v in order_params.items():
                op_csv_path = os.path.join(self._ode_file_path, k)
                np.savetxt(op_csv_path, v, delimiter=",")
                if step == "":
                    param_name = k.split(".")[0]
                    txt_file.write(f"{param_name},{op_csv_path}\n")
                else:
                    param_name = k.split(".")[0][: -len(step)]
                    if param_name in self._debug_copy:
                        txt_file.write(f"{param_name},{op_csv_path}\n")

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
