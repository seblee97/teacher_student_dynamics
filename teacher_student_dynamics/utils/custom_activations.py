import numpy as np
import torch


def linear_activation(x):
    return x


class ScaledErf:

    SCALING = np.sqrt(2)

    @staticmethod
    def __call__(x):
        return torch.erf(x / ScaledErf.SCALING)
