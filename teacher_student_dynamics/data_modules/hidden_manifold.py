from typing import Callable, Dict, List, Union

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from teacher_student_dynamics import constants
from teacher_student_dynamics.data_modules import base_data_module
from teacher_student_dynamics.utils import custom_activations


class HiddenManifold(base_data_module.BaseData):
    """Class for generating data drawn i.i.d from unit normal Gaussian."""

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        input_dimension: int,
        latent_dimension: int,
        mean: Union[int, float],
        variance: Union[int, float],
        activation: str,
        feature_matrix: np.ndarray,
        precompute_data: Union[None, int],
    ):

        self._latent_dimension = latent_dimension

        if variance == 0:
            self._latent_distribution = tdist.Categorical(torch.Tensor([1.0]))
        else:
            self._latent_distribution = tdist.Normal(mean, np.sqrt(variance))

        self._activation_name = activation
        self._activation = self._get_activation_function()
        self._feature_matrix = feature_matrix

        self._surrogate_feature_matrices = []

        super().__init__(
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            input_dimension=input_dimension,
            precompute_data=precompute_data,
        )

    @property
    def folding_function_coefficients(self):
        # a, b, b
        if self._activation_name == constants.SCALED_ERF:
            # from wolfram alpha (?)
            a = 0
            b = np.sqrt(1 / np.pi)
            c = 1 / 3
        else:
            raise ValueError(
                f"Folding function coefficients for {self._activation_name} unknown."
            )
        return a, b, c

    @property
    def feature_matrix(self):
        return self._feature_matrix

    @property
    def surrogate_feature_matrices(self):
        return self._surrogate_feature_matrices

    @surrogate_feature_matrices.setter
    def surrogate_feature_matrices(self, surrogate_feature_matrices: List):
        self._surrogate_feature_matrices = surrogate_feature_matrices

    @property
    def precompute_labels_on(self) -> Union[torch.Tensor, None]:
        if self._precompute_data is not None and self._precompute_labels:
            return self._precomputed_latents

    def _get_precomputed_data(self):
        grouped_training_data = self._get_fixed_data(size=self._precompute_data)
        self._precomputed_latents = grouped_training_data[constants.LATENT]
        self._training_data = [
            {constants.X: i, constants.LATENT: l}
            for i, l in zip(
                grouped_training_data[constants.X], self._precomputed_latents
            )
        ]
        self._precompute_labels = True

    def _get_activation_function(self) -> Callable:
        """Makes the activation function specified by the config.

        Returns:
            activation_function: Callable object

        Raises:
            ValueError: if activation provided in config is not recognised
        """
        if self._activation_name == constants.RELU:
            activation_function = F.relu
        elif self._activation_name == constants.SCALED_ERF:
            activation_function = custom_activations.scaled_erf_activation
        elif self._activation_name == constants.LINEAR:
            activation_function = custom_activations.linear_activation
        else:
            raise ValueError(f"Unknown activation: {self._activation_name}")
        return activation_function

    def _get_fixed_data(self, size: int) -> Dict[str, torch.Tensor]:
        data_latent = self._latent_distribution.sample((size, self._latent_dimension))

        data_inputs = self._activation(
            np.matmul(data_latent, self._feature_matrix.T)
            / np.sqrt(self._input_dimension)
        )
        return {
            constants.X: data_inputs,
            constants.LATENT: data_latent,
        }

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Returns batch of training data (input only)"""
        if self._precompute_data is not None:
            if not self._training_data:
                self._get_precomputed_data()
            return self._training_data.pop()

        latent = self._latent_distribution.sample(
            (self._train_batch_size, self._latent_dimension)
        )
        batch = self._activation(
            np.matmul(latent, self._feature_matrix.T) / np.sqrt(self._input_dimension)
        )
        return {constants.X: batch, constants.LATENT: latent}

    def get_mixed_batch(self, gamma: float, surrogate_index: int):
        latent = self._latent_distribution.sample(
            (self._train_batch_size, self._latent_dimension)
        )
        mixed_feature_matrix = (
            gamma**2 * self._feature_matrix
            + np.sqrt(1 - gamma**2)
            * self._surrogate_feature_matrices[surrogate_index]
        )
        batch = self._activation(
            np.matmul(latent, mixed_feature_matrix.T) / np.sqrt(self._input_dimension)
        )
        return {constants.X: batch, constants.LATENT: latent}
