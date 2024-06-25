import abc
import copy
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from teacher_student_dynamics import constants
from teacher_student_dynamics.utils import custom_activations


class MultiHeadNetwork(nn.Module, abc.ABC):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        num_heads: int,
        bias: bool,
        nonlinearity: str,
        initialisation_std: Optional[float],
        head_initialisation_std: Optional[float],
        normalise_weights: Optional[bool] = False,
        heads_one: Optional[bool] = False,
        unit_norm_head: Optional[bool] = False,
        train_hidden_layer: Optional[bool] = False,
        train_head_layer: Optional[bool] = False,
        freeze: Optional[bool] = False,
    ) -> None:
        super().__init__()

        self._input_dimension = input_dimension
        self._hidden_dimension = hidden_dimension
        self._output_dimension = output_dimension
        self._num_heads = num_heads
        self._layer_dimensions = [self._input_dimension] + [self._hidden_dimension]
        self._bias = bias
        self._nonlinearity = nonlinearity
        self._initialisation_std = initialisation_std
        self._head_initialisation_std = head_initialisation_std
        self._normalise_weights = normalise_weights
        self._heads_one = heads_one
        self._unit_norm_head = unit_norm_head
        self._train_hidden_layer = train_hidden_layer
        self._train_head_layer = train_head_layer

        # set to 0 by default
        self._current_head: int = 0
        self._forward_hidden_scaling = 1 / np.sqrt(input_dimension)

        self._nonlinear_function = self._get_nonlinear_function()
        self._construct_layers()

        if freeze:
            self._freeze()

    @property
    def layers(self) -> nn.ModuleList:
        return self._layers

    @property
    def heads(self) -> nn.ModuleList:
        return self._heads

    @property
    def self_overlap(self):
        with torch.no_grad():
            layer = self._layers[0].weight.data
            overlap = layer.mm(layer.t()) / self._input_dimension
        return overlap

    @property
    def nonlinear_function(self):
        return self._nonlinear_function

    def _get_nonlinear_function(self) -> Callable:
        """Makes the nonlinearity function specified by the config.

        Returns:
            nonlinear_function: Callable object, the nonlinearity function

        Raises:
            ValueError: If nonlinearity provided in config is not recognised
        """
        if self._nonlinearity == constants.RELU:
            nonlinear_function = F.relu
        elif self._nonlinearity == constants.SCALED_ERF:
            nonlinear_function = custom_activations.ScaledErf()
        elif self._nonlinearity == constants.LINEAR:
            nonlinear_function = custom_activations.linear_activation
        else:
            raise ValueError(f"Unknown non-linearity: {self._nonlinearity}")
        return nonlinear_function

    def _construct_layers(self) -> None:
        """Instantiate layers (input, hidden and output) according to
        dimensions specified in configuration. Note this method makes a call to
        the abstract _construct_output_layers, which is implemented by the
        child.
        """
        self._layers = nn.ModuleList([])

        for layer_size, next_layer_size in zip(
            self._layer_dimensions[:-1], self._layer_dimensions[1:]
        ):
            layer = nn.Linear(layer_size, next_layer_size, bias=self._bias)
            self._initialise_weights(layer)
            if self._normalise_weights:
                with torch.no_grad():
                    for node_index, node in enumerate(layer.weight):
                        node_magnitude = torch.norm(node)
                        layer.weight[node_index] = (
                            np.sqrt(layer.weight.shape[1]) * node / node_magnitude
                        )
            self._layers.append(layer)

        self._construct_output_layers()

    def _initialise_weights(self, layer: nn.Module, value=None) -> None:
        """In-place weight initialisation for a given layer in accordance with configuration.

        Args:
            layer: the layer to be initialised.
        """
        if value is not None:
            layer.weight.data.fill_(value)
        else:
            if self._initialisation_std is not None:
                nn.init.normal_(layer.weight, std=self._initialisation_std)
                if self._bias:
                    nn.init.normal_(layer.bias, std=self._initialisation_std)

    def _initialise_head(self, layer: nn.Module, value=None) -> None:
        """In-place weight initialisation for a given layer in accordance with configuration.

        Args:
            layer: the layer to be initialised.
        """
        if value is not None:
            layer.weight.data.fill_(value)
        else:
            if self._initialisation_std is not None:
                nn.init.normal_(layer.weight, std=self._head_initialisation_std)
                if self._bias:
                    nn.init.normal_(layer.bias, std=self._head_initialisation_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method performs the forward pass. This implements the
        abstract method from the nn.Module base class.

        Args:
            x: input tensor

        Returns:
            y: output of network
        """
        for layer in self._layers:
            x = self._nonlinear_function(self._forward_hidden_scaling * layer(x))

        y = self._get_output_from_head(x)

        return y

    def _construct_output_layers(self):
        """Instantiate the output layers."""
        self._heads = nn.ModuleList([])
        for _ in range(self._num_heads):
            output_layer = nn.Linear(
                self._layer_dimensions[-1], self._output_dimension, bias=self._bias
            )
            if self._heads_one:
                output_layer.weight.data = torch.ones_like(output_layer.weight)
            elif self._unit_norm_head:
                head_norm = torch.norm(output_layer.weight)
                normalised_head = output_layer.weight / head_norm
                output_layer.weight.data = normalised_head
            else:
                self._initialise_head(output_layer)
            # freeze heads by default
            for param in output_layer.parameters():
                param.requires_grad = False
            self._heads.append(output_layer)

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through relevant head."""
        y = self._heads[self._current_head](x)
        return y

    def get_trainable_parameters(self):  # TODO: return type
        """Returns trainable (non-frozen) parameters of network.
        Return split into hidden and head parameter lists.
        """
        trainable_hidden_parameters = []
        if self._train_hidden_layer:
            trainable_hidden_parameters += [
                {"params": filter(lambda p: p.requires_grad, layer.parameters())}
                for layer in self._layers
            ]
        trainable_head_parameters = []
        if self._train_head_layer:
            trainable_head_parameters += [
                {
                    "params": head.parameters(),
                }
                for head in self._heads
            ]
        return trainable_hidden_parameters, trainable_head_parameters

    def signal_boundary(self, new_head: int) -> None:
        """Alert network to boundary. Freeze previous head, unfreeze new head.
        No restriction on new_head being different from current_head."""
        # freeze weights of head for previous task
        self._freeze_head(self._current_head)
        self._unfreeze_head(new_head)

        self._current_head = new_head

    def _freeze(self) -> None:
        for layer in self._layers:
            for param in layer.parameters():
                param.requires_grad = False

    def _freeze_head(self, head_index: int) -> None:
        """Freeze weights of head for task with index head index."""
        for param in self._heads[head_index].parameters():
            param.requires_grad = False

    def _unfreeze_head(self, head_index: int) -> None:
        """Unfreeze weights of head for task with index head index."""
        for param in self._heads[head_index].parameters():
            param.requires_grad = True

    def _freeze_hidden_layers(self) -> None:
        """Freeze weights in all but head weights
        (used for e.g. frozen feature model)."""
        for layer in self._layers:
            for param in layer.parameters():
                param.requires_grad = False
        self._frozen = True

    def _unfreeze_hidden_layers(self) -> None:
        """Unfreeze weights in all but head weights
        (used for e.g. frozen feature model)."""
        for layer in self._layers:
            for param in layer.parameters():
                param.requires_grad = True
        self._frozen = False

    def forward_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Makes call to forward, using all heads (used largely for evaluation)"""
        for layer in self._layers:
            x = self._nonlinear_function(self._forward_hidden_scaling * layer(x))
        task_outputs = [head(x) for head in self._heads]
        return task_outputs

    def forward_all_batches(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Call to forward of all network heads where each head gets a different input"""
        task_outputs = []

        if len(self._heads) == 1:
            for xi in x:
                for layer in self._layers:
                    ai = self._nonlinear_function(
                        self._forward_hidden_scaling * layer(xi)
                    )
                task_output = self._heads[0](ai)
                task_outputs.append(task_output)
        else:
            for xi, head in zip(x, self._heads):
                for layer in self._layers:
                    ai = self._nonlinear_function(
                        self._forward_hidden_scaling * layer(xi)
                    )
                task_output = head(ai)
                task_outputs.append(task_output)
        return task_outputs
