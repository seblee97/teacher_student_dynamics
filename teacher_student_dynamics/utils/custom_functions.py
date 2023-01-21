from typing import Tuple, Union

import numpy as np
import torch


def generate_rotated_vectors(
    dimension: int, theta: float, normalisation: Union[float, int] = 1
) -> Tuple[np.ndarray]:
    """
    Generate 2 N-dimensional vectors that are rotated by an angle theta from each other.

    Args:
        dimension: desired dimension of two vectors.
        theta: angle to be rotated (in radians).
        normalisation: scaling of final two vectors.
            e.g. with normalisation 1, y_1 cdot y_2 = 1.

    Returns:
        rotated_vectors: tuple of two vectors appropriately rotated.
    """
    v_1 = np.random.normal(size=(dimension))
    v_2 = np.random.normal(size=(dimension))
    normal_1 = normalisation * v_1 / np.linalg.norm(v_1)
    normal_2 = normalisation * v_2 / np.linalg.norm(v_2)

    stacked_orthonormal = np.stack((normal_1, normal_2)).T

    # generate rotation vectors
    x_1 = np.array([0, 1])
    x_2 = np.array([np.sin(theta), np.cos(theta)])

    # generate rotated vectors
    y_1 = np.dot(stacked_orthonormal, x_1)
    y_2 = np.dot(stacked_orthonormal, x_2)

    return y_1, y_2


def generate_rotated_matrices(
    unrotated_weights: torch.Tensor,
    alpha: float,
    normalisation: Union[None, float, int] = None,
    orthogonalise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Method for generating two sets of matrices 'rotated' by a specified amount.
    Here 'rotation' corresponds to an interpolation between an original matrix, W_1 (provided),
    and an independently generated (orthogonal in the limit) matrix of the same dimension.
    This interpolation (alpha) is equivalent (in the limit) to the sum of diagonal elements
    of the overlap matrix of the two matrices W_1^TW_2, hence it can be interpreted as a rotation.

    Args:
        unrotated_weights: first weight matrix.
        alpha: interpolation parameter (between 0 and 1).
        normalisation: optional argument for the magnitude of the weights matrix.
        orthogonalise: whether to orthogonalise initial weight matrix.

    Returns:
        w_1: first weight matrix (equal to unrotated_weights argument
        in absence of additional normalisation and orthogonalisation).
        w_2: second weight matrix "rotated" by alpha.
    """
    if orthogonalise:
        normalisation = normalisation or 1
        first_matrix_self_overlap = (
            torch.mm(unrotated_weights, unrotated_weights.T) / normalisation
        )
        first_matrix_L = torch.cholesky(first_matrix_self_overlap)
        # orthonormal first matrix
        first_matrix = torch.mm(torch.inverse(first_matrix_L), unrotated_weights)
    else:
        first_matrix = unrotated_weights

    first_matrix_norms = torch.norm(first_matrix, dim=1)

    random_matrix = torch.randn(unrotated_weights.shape)

    second_matrix = alpha * first_matrix + np.sqrt(1 - alpha**2) * random_matrix

    for node_index, node in enumerate(second_matrix):
        node_norm = torch.norm(node)
        scaling = first_matrix_norms[node_index] / node_norm
        second_matrix[node_index] = scaling * node

    return first_matrix, second_matrix
