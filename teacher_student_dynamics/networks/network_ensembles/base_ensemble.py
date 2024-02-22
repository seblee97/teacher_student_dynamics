import abc
from typing import List, Union

import numpy as np
import torch

from teacher_student_dynamics import constants
from teacher_student_dynamics.networks import multi_head_network


class BaseEnsemble(abc.ABC):
    """Base class for sets/ensembles of networks
    (as opposed to single network)."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: List[int],
        output_dimension: int,
        ensemble_size: int,
        bias: bool,
        nonlinearity: List[str],
        initialisation_std: float,
        heads_one: bool,
        unit_norm_head: bool,
        normalise_weights: bool,
    ) -> None:
        self._input_dimension = input_dimension
        self._hidden_dimension = hidden_dimension
        self._output_dimension = output_dimension
        self._ensemble_size = ensemble_size
        self._bias = bias
        self._nonlinearity = nonlinearity
        self._initialisation_std = initialisation_std
        self._normalise_weights = normalise_weights
        self._heads_one = heads_one
        self._unit_norm_head = unit_norm_head

        self._networks = self._setup_networks()

    @property
    def networks(self) -> List:
        """Getter method for networks networks."""
        return self._networks

    @property
    def cross_overlaps(self):
        overlaps = []
        with torch.no_grad():
            for i in range(len(self._networks)):
                for j in range(i, len(self._networks)):
                    if i != j:
                        if self._hidden_dimension is None:
                            overlap = (
                                torch.mm(
                                    self._networks[i].heads[0].weight.data,
                                    self._networks[j].heads[0].weight.data.T,
                                )
                                / self._input_dimension
                            )
                        else:
                            overlap = (
                                torch.mm(
                                    self._networks[i].layers[0].weight.data,
                                    self._networks[j].layers[0].weight.data.T,
                                )
                                / self._input_dimension
                            )
                        overlaps.append(overlap)
        return overlaps

    @abc.abstractmethod
    def _setup_networks(self) -> None:
        """instantiate network(s)"""
        pass

    def forward(self, network_index: int, batch: torch.Tensor) -> torch.Tensor:
        """Call to current network forward."""
        output = self._networks[network_index](batch)
        return output

    def _init_network(self, num_heads: int = 1):
        return multi_head_network.MultiHeadNetwork(
            input_dimension=self._input_dimension,
            hidden_dimension=self._hidden_dimension,
            output_dimension=self._output_dimension,
            num_heads=num_heads,
            bias=self._bias,
            nonlinearity=self._nonlinearity,
            initialisation_std=self._initialisation_std,
            normalise_weights=self._normalise_weights,
            heads_one=self._heads_one,
            unit_norm_head=self._unit_norm_head,
            freeze=True,
        )

    def save_all_network_weights(self, save_path: str) -> None:
        """Save weights associated with each network.

        Args:
            save_path: path to save weights, will be concatenated with
            _i where i is the index of the network.
        """
        for t, network in enumerate(self._networks):
            torch.save(network.state_dict(), f"{save_path}_{t}")

    def save_weights(self, network_index: int, save_path: str) -> None:
        """Save weights associated with given network index"""
        torch.save(self._networks[network_index].state_dict(), save_path)

    def forward_all(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """Call to forward of all networks (used primarily for evaluation)"""
        outputs = [self.forward(t, batch) for t in range(self._ensemble_size)]
        return outputs

    def forward_all_batches(self, batches: torch.Tensor) -> List[torch.Tensor]:
        """Call to forward of all networks where each network gets a different input"""
        outputs = [
            self.forward(t, batch)
            for t, batch in zip(range(self._ensemble_size), batches)
        ]
        return outputs
