import abc
from typing import Dict, List, Union

import torch

from teacher_student_dynamics import constants


class BaseData(abc.ABC):
    """Class for handling data

    Abstract methods that need to be implemented:

    - get_test_data
    - get_batch
    - signal_task_bounary_to_data_generator
    """

    def __init__(
        self,
        device: str,
        train_batch_size: int,
        test_batch_size: int,
        input_dimension: int,
        precompute_data: Union[int, None],
    ):
        """Class constructor"""
        self._train_batch_size = train_batch_size
        self._test_batch_size = test_batch_size
        self._input_dimension = input_dimension
        self._precompute_data = precompute_data

        self._device = device

        if self._precompute_data is not None:
            self._get_precomputed_data()

    @abc.abstractproperty
    def precompute_labels_on(self) -> Union[None, torch.Tensor]:
        pass

    @abc.abstractmethod
    def _get_precomputed_data(self) -> None:
        pass

    @abc.abstractmethod
    def get_test_data(self) -> Dict[str, torch.Tensor]:
        """returns fixed test data sets (data and labels)"""
        pass

    @abc.abstractmethod
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """returns batch of training data (input data and label if relevant)"""
        pass

    @abc.abstractmethod
    def _get_fixed_data(self) -> Dict[str, torch.Tensor]:
        """returns batch of data (input data and label if relevant)"""
        pass

    def add_precomputed_labels(self, labels: List[float]) -> None:
        self._training_data = [
            {**i, **{constants.Y: j}} for i, j in zip(self._training_data, labels)
        ]
        self._precompute_labels = False

    def get_test_data(self) -> Dict[str, torch.Tensor]:
        """Give fixed test data set (input data only)."""
        return self._get_fixed_data(size=self._test_batch_size)
