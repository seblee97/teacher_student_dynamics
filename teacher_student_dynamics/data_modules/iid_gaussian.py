from typing import Dict, Union

import numpy as np
import torch
import torch.distributions as tdist
from torch.utils.data import DataLoader

from teacher_student_dynamics import constants
from teacher_student_dynamics.data_modules import base_data_module


class IIDGaussianDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_size: int,
        distribution: torch.distributions.Normal,
        input_dimension,
    ) -> None:
        self._dataset_size = dataset_size
        self._data_distribution = distribution
        self._input_dimension = input_dimension

        self._fixed_dataset = self._data_distribution.sample(
            (self._dataset_size, self._input_dimension)
        )

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, idx):
        return self._fixed_dataset[idx]


class IIDGaussian(base_data_module.BaseData):
    """Class for generating data drawn i.i.d from unit normal Gaussian."""

    def __init__(
        self,
        device: str,
        train_batch_size: int,
        test_batch_size: int,
        input_dimension: int,
        mean: Union[int, float],
        variance: Union[int, float],
        dataset_size: Union[str, int],
        precompute_data: Union[None, int],
    ):

        if variance == 0:
            self._data_distribution = tdist.Categorical(torch.Tensor([1.0]))
        else:
            self._data_distribution = tdist.Normal(mean, np.sqrt(variance))

        self._dataset_size = dataset_size

        if self._dataset_size != constants.INF:
            self._dataset = IIDGaussianDataset(
                dataset_size=self._dataset_size,
                distribution=self._data_distribution,
                input_dimension=self._input_dimension,
            )
            self._reset_data_iterator()

        super().__init__(
            device=device,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            input_dimension=input_dimension,
            precompute_data=precompute_data,
        )

    @property
    def precompute_labels_on(self) -> Union[torch.Tensor, None]:
        if self._precompute_data is not None and self._precompute_labels:
            return self._precomputed_inputs

    def _get_precomputed_data(self):
        grouped_training_data = self._get_fixed_data(size=self._precompute_data)
        self._precomputed_inputs = grouped_training_data[constants.X]
        self._training_data = [{constants.X: i} for i in self._precomputed_inputs]
        self._precompute_labels = False

    def _get_fixed_data(self, size) -> Dict[str, torch.Tensor]:

        input_data = self._data_distribution.sample((size, self._input_dimension)).to(
            self._device
        )

        data_dict = {constants.X: input_data}

        return data_dict

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Returns batch of training data (input only)"""
        if self._dataset_size == constants.INF:
            if self._precompute_data is not None:
                if not self._training_data:
                    self._get_precomputed_data()
                return self._training_data.pop()
            batch = self._get_infinite_dataset_batch()
        else:
            batch = self._get_finite_dataset_batch()
        return {constants.X: batch}

    def _get_finite_dataset_batch(self) -> torch.Tensor:
        try:
            batch = next(iter(self._data_iterator))
        except StopIteration:
            self._reset_data_iterator()
            batch = next(iter(self._data_iterator))
        return batch

    def _get_infinite_dataset_batch(self) -> torch.Tensor:
        batch = self._data_distribution.sample(
            (self._train_batch_size, self._input_dimension)
        ).to(self._device)
        return batch

    def _reset_data_iterator(self):
        self._dataloader = DataLoader(
            self._dataset, batch_size=self._train_batch_size, shuffle=True
        )
        self._data_iterator = iter(self._dataloader)
