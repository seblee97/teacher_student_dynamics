from typing import List, Union

import numpy as np
import torch

from teacher_student_dynamics.networks.network_ensembles import base_ensemble
from teacher_student_dynamics.utils import custom_functions


class NodeSharingEnsemble(base_ensemble.BaseEnsemble):
    """Network ensemble in which a subset of nodes are shared between networks
    and the remaining features (input to hidden) weights are rotated by
    a given amount."""

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
        num_shared_nodes: int,
        feature_rotation_alpha: float,
    ):
        self._num_shared_nodes = num_shared_nodes
        self._feature_rotation_alpha = feature_rotation_alpha
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
        """Instantiate networks with subset of nodes copied and rest rotated.

        Raises:
            AssertionError: If more than 2 networks are requested.
        """

        assert (
            self._ensemble_size == 2
        ), "Node sharing ensemble currently implemented for 2 networks only."

        networks = [self._init_network() for _ in range(self._ensemble_size)]

        weights_to_be_copied = (
            networks[0].layers[0].weight.data[: self._num_shared_nodes]
        )
        weights_to_be_rotated = (
            networks[0].layers[0].weight.data[self._num_shared_nodes :]
        )

        with torch.no_grad():
            rotated_weight_vectors = custom_functions.generate_rotated_matrices(
                unrotated_weights=weights_to_be_rotated,
                alpha=self._feature_rotation_alpha,
                normalisation=self._input_dimension,
                orthogonalise=False,
            )

            for i in range(self._ensemble_size):
                network_weight_tensors = torch.cat(
                    (
                        weights_to_be_copied,
                        torch.Tensor(
                            rotated_weight_vectors[i].reshape(
                                self._hidden_dimension - self._num_shared_nodes,
                                self._input_dimension,
                            )
                        ),
                    ),
                    dim=0,
                )

                networks[i].layers[0].weight.data = network_weight_tensors

        return networks
