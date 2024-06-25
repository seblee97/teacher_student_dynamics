from typing import List, Union

import numpy as np
import torch

from teacher_student_dynamics.networks.network_ensembles import base_ensemble
from teacher_student_dynamics.utils import custom_functions


class IdenticalEnsemble(base_ensemble.BaseEnsemble):
    """Ensemble in which networks are identical."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        ensemble_size: int,
        bias: bool,
        nonlinearity: str,
        initialisation_std: float,
        head_initialisation_std: float,
        heads_one: bool,
        unit_norm_head: bool,
        normalise_weights: bool,
    ):
        super().__init__(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=output_dimension,
            ensemble_size=ensemble_size,
            bias=bias,
            nonlinearity=nonlinearity,
            initialisation_std=initialisation_std,
            head_initialisation_std=head_initialisation_std,
            heads_one=heads_one,
            unit_norm_head=unit_norm_head,
            normalise_weights=normalise_weights,
        )

    def _setup_networks(self) -> None:
        """Instantiate identical networks."""

        networks = [self._init_network() for _ in range(self._ensemble_size)]

        input_hidden_weights_to_be_copied = networks[0].layers[0].weight.data
        hidden_output_weights_to_be_copied = networks[0].heads[0].weight.data

        with torch.no_grad():

            for i in range(1, self._ensemble_size):

                networks[i].layers[0].weight.data = input_hidden_weights_to_be_copied
                networks[i].heads[0].weight.data = hidden_output_weights_to_be_copied

        return networks
