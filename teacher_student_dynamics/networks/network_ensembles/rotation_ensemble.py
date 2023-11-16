import warnings
from typing import List, Union

import numpy as np
import torch

from teacher_student_dynamics.networks.network_ensembles import base_ensemble
from teacher_student_dynamics.utils import custom_functions


class RotationEnsemble(base_ensemble.BaseEnsemble):
    """Ensemble in which both feature and readout similarities are tuned by rotation."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        ensemble_size: int,
        bias: bool,
        nonlinearity: str,
        initialisation_std: float,
        normalise_weights: bool,
        heads_one: bool,
        unit_norm_head: bool,
        feature_rotation_alpha: float,
        readout_rotation_alpha: float,
    ):
        self._feature_rotation_alpha = feature_rotation_alpha
        self._readout_rotation_alpha = readout_rotation_alpha
        super().__init__(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=output_dimension,
            ensemble_size=ensemble_size,
            bias=bias,
            nonlinearity=nonlinearity,
            initialisation_std=initialisation_std,
            normalise_weights=normalise_weights,
            heads_one=heads_one,
            unit_norm_head=unit_norm_head,
        )

    def _setup_networks(self) -> None:
        """Setup networks with rotations across input to hidden and
        hidden to output weights.

        Raises:
            AssertionError: If more than 2 networks are requested.
            Warning: If the hidden dimension is not greater than 1,
            this is for the notion of rotation to have meaning in the readout.
        """
        assert (
            self._ensemble_size == 2
        ), "Both rotation ensemble currently implemented for 2 networks only."

        if self._hidden_dimension == 1:
            warnings.warn(
                "Both rotation ensemble only valid for hidden dimensions > 1."
            )

        networks = [self._init_network() for _ in range(self._ensemble_size)]

        with torch.no_grad():

            (
                network_0_feature_weights,
                network_1_feature_weights,
            ) = self._get_rotated_weights(
                unrotated_weights=networks[0].layers[0].weight.data.T,
                alpha=self._feature_rotation_alpha,
                normalisation=self._hidden_dimension,
            )

            networks[0].layers[0].weight.data = network_0_feature_weights.T
            networks[1].layers[0].weight.data = network_1_feature_weights.T

            if self._hidden_dimension > 1:
                (
                    network_0_readout_weights,
                    network_1_readout_weights,
                ) = self._get_rotated_readout_weights(networks=networks)

                networks[0].heads[0].weight.data = network_0_readout_weights
                networks[1].heads[0].weight.data = network_1_readout_weights

        return networks

    def _feature_overlap(self, feature_1: torch.Tensor, feature_2: torch.Tensor):
        alpha_matrix = torch.mm(feature_1, feature_2.T) / self._hidden_dimension
        alpha = torch.mean(alpha_matrix.diagonal())

        return alpha

    def _readout_overlap(self, feature_1: torch.Tensor, feature_2: torch.Tensor):
        alpha = torch.mm(feature_1, feature_2.T) / (
            torch.norm(feature_1) * torch.norm(feature_2)
        )
        return alpha

    def _get_rotated_weights(
        self,
        unrotated_weights: torch.Tensor,
        alpha: float,
        normalisation: Union[None, int],
    ):
        return custom_functions.generate_rotated_matrices(
            unrotated_weights=unrotated_weights, alpha=alpha, orthogonalise=False
        )
        # if normalisation is not None:
        #     # orthonormalise input to hidden weights of first network
        #     self_overlap = (
        #         torch.mm(unrotated_weights, unrotated_weights.T) / normalisation
        #     )
        #     L = torch.cholesky(self_overlap)
        #     orthonormal_weights = torch.mm(torch.inverse(L), unrotated_weights)
        # else:
        #     orthonormal_weights = unrotated_weights

        # # construct input to hidden weights of second network
        # second_network_rotated_weights = alpha * orthonormal_weights + np.sqrt(
        #     1 - alpha**2
        # ) * torch.randn(orthonormal_weights.shape)

        # return orthonormal_weights, second_network_rotated_weights

    def _get_rotated_readout_weights(self, networks: List):

        theta = np.arccos(self._readout_rotation_alpha)

        # keep current norms
        current_norm = np.mean(
            [torch.norm(network.heads[0].weight) for network in networks]
        )

        rotated_weight_vectors = custom_functions.generate_rotated_vectors(
            dimension=self._hidden_dimension,
            theta=theta,
            normalisation=current_norm,
        )

        network_0_rotated_weight_tensor = torch.Tensor(
            rotated_weight_vectors[0]
        ).reshape(networks[0].heads[0].weight.data.shape)

        network_1_rotated_weight_tensor = torch.Tensor(
            rotated_weight_vectors[1]
        ).reshape(networks[1].heads[0].weight.data.shape)

        return network_0_rotated_weight_tensor, network_1_rotated_weight_tensor
