import os
import abc
from typing import Dict, List

import numpy as np

from teacher_student_dynamics import constants


class BaseNetworkConfiguration(abc.ABC):
    def __init__(self) -> None:
        pass

    @property
    @abc.abstractmethod
    def configuration_dictionary(self) -> Dict:
        pass

    @property
    @abc.abstractmethod
    def sub_dictionary(self) -> Dict:
        pass


class VanillaNetworkConfiguration(BaseNetworkConfiguration):
    """object to store configuration of student/teacher networks in unified way."""

    def __init__(
        self,
        student_head_weights: List[np.ndarray],
        teacher_head_weights: List[np.ndarray],
        student_self_overlap: np.ndarray,
        teacher_self_overlaps: List[np.ndarray],
        teacher_cross_overlaps: List[np.ndarray],
        student_teacher_overlaps: List[np.ndarray],
    ):
        self._student_head_weights = student_head_weights
        self._teacher_head_weights = teacher_head_weights
        self._student_self_overlap = student_self_overlap
        self._teacher_self_overlaps = teacher_self_overlaps
        self._teacher_cross_overlaps = teacher_cross_overlaps
        self._student_teacher_overlaps = student_teacher_overlaps

        self._num_teachers = len(self._teacher_head_weights)

        super().__init__()

    @property
    def student_head_weights(self) -> List[np.ndarray]:
        return self._student_head_weights

    @student_head_weights.setter
    def student_head_weights(self, student_head_weights) -> None:
        self._student_head_weights = student_head_weights

    @property
    def teacher_head_weights(self) -> List[np.ndarray]:
        return self._teacher_head_weights

    @property
    def student_self_overlap(self) -> np.ndarray:
        return self._student_self_overlap

    @student_self_overlap.setter
    def student_self_overlap(self, student_self_overlap) -> None:
        self._student_self_overlap = student_self_overlap

    @property
    def teacher_self_overlaps(self) -> List[np.ndarray]:
        return self._teacher_self_overlaps

    @property
    def teacher_cross_overlaps(self) -> List[np.ndarray]:
        return self._teacher_cross_overlaps

    @property
    def student_teacher_overlaps(self) -> List[np.ndarray]:
        return self._student_teacher_overlaps

    @student_teacher_overlaps.setter
    def student_teacher_overlaps(self, student_teacher_overlaps) -> None:
        self._student_teacher_overlaps = student_teacher_overlaps

    @property
    def configuration_dictionary(self):
        return {
            constants.STUDENT_SELF_OVERLAP: self._student_self_overlap,
            constants.STUDENT_TEACHER_OVERLAPS: self._student_teacher_overlaps,
            constants.STUDENT_HEAD_WEIGHTS: self._student_head_weights,
            constants.TEACHER_HEAD_WEIGHTS: self._teacher_head_weights,
            constants.TEACHER_CROSS_OVERLAPS: self._teacher_cross_overlaps,
            constants.TEACHER_SELF_OVERLAPS: self._teacher_self_overlaps,
        }

    @property
    def sub_dictionary(self):
        network_configuration_dictionary = {}

        for i, head in enumerate(self._student_head_weights):
            for j, weight in enumerate(head):
                network_configuration_dictionary[
                    f"{constants.STUDENT_HEAD}_{i}_{constants.WEIGHT}_{j}"
                ] = weight
        for i, head in enumerate(self._teacher_head_weights):
            for j, weight in enumerate(head):
                network_configuration_dictionary[
                    f"{constants.TEACHER_HEAD}_{i}_{constants.WEIGHT}_{j}"
                ] = weight
        for (i, j), overlap_value in np.ndenumerate(self._student_self_overlap):
            network_configuration_dictionary[
                f"{constants.STUDENT_SELF_OVERLAP}_{i}_{j}"
            ] = overlap_value
        for t, student_teacher_overlap in enumerate(self._student_teacher_overlaps):
            for (i, j), overlap_value in np.ndenumerate(student_teacher_overlap):
                network_configuration_dictionary[
                    f"{constants.STUDENT_TEACHER}_{t}_{constants.OVERLAP}_{i}_{j}"
                ] = overlap_value

        return network_configuration_dictionary


class HiddenManifoldNetworkConfiguration(VanillaNetworkConfiguration):
    """object to store configuration of student/teacher networks in unified way."""

    def __init__(
        self,
        student_head_weights: List[np.ndarray],
        teacher_head_weights: List[np.ndarray],
        student_self_overlap: np.ndarray,
        teacher_self_overlaps: List[np.ndarray],
        teacher_cross_overlaps: List[np.ndarray],
        student_teacher_overlaps: List[np.ndarray],
        rotated_student_teacher_overlaps: List[np.ndarray],
        student_weighted_feature_matrices: List[np.ndarray],
        student_local_field_covariances: List[np.ndarray],
        rotated_student_local_field_covariances: List[np.ndarray],
        student_weighted_feature_matrix_self_overlaps: List[np.ndarray],
        w_tilde_tau: List[np.ndarray],
        rotated_student_weighted_feature_matrix_self_overlaps: List[np.ndarray],
        feature_matrix_overlaps: List[np.ndarray],
        feature_matrix_overlap_eigenvalues: List[np.ndarray],
        feature_matrix_overlap_eigenvectors: List[np.ndarray],
        student_teacher_overlap_densities: List[np.ndarray],
        student_latent_self_overlap_densities: List[np.ndarray],
        projected_teacher_self_overlaps: List[np.ndarray],
    ):

        super().__init__(
            student_head_weights=student_head_weights,
            teacher_head_weights=teacher_head_weights,
            student_self_overlap=student_self_overlap,
            teacher_self_overlaps=teacher_self_overlaps,
            teacher_cross_overlaps=teacher_cross_overlaps,
            student_teacher_overlaps=student_teacher_overlaps,
        )

        self._student_weighted_feature_matrices = student_weighted_feature_matrices
        self._student_local_field_covariances = student_local_field_covariances
        self._rotated_student_local_field_covariances = (
            rotated_student_local_field_covariances
        )
        self._student_weighted_feature_matrix_self_overlaps = (
            student_weighted_feature_matrix_self_overlaps
        )
        self._w_tilde_tau = w_tilde_tau
        self._rotated_student_weighted_feature_matrix_self_overlaps = (
            rotated_student_weighted_feature_matrix_self_overlaps
        )
        self._rotated_student_teacher_overlaps = rotated_student_teacher_overlaps
        self._feature_matrix_overlaps = feature_matrix_overlaps
        self._feature_matrix_overlap_eigenvalues = feature_matrix_overlap_eigenvalues
        self._feature_matrix_overlap_eigenvectors = feature_matrix_overlap_eigenvectors
        self._student_teacher_overlap_densities = student_teacher_overlap_densities
        self._student_latent_self_overlap_densities = (
            student_latent_self_overlap_densities
        )
        self._projected_teacher_self_overlaps = projected_teacher_self_overlaps

    @property
    def student_weighted_feature_matrices(self) -> List[np.ndarray]:
        return self._student_weighted_feature_matrices

    @student_weighted_feature_matrices.setter
    def student_weighted_feature_matrices(
        self, student_weighted_feature_matrices
    ) -> np.ndarray:
        self._student_weighted_feature_matrices = student_weighted_feature_matrices

    @property
    def student_local_field_covariances(self) -> List[np.ndarray]:
        return self._student_local_field_covariances

    @student_local_field_covariances.setter
    def student_local_field_covariances(
        self, student_local_field_covariances
    ) -> List[np.ndarray]:
        self._student_local_field_covariances = student_local_field_covariances

    @property
    def rotated_student_local_field_covariances(self) -> List[np.ndarray]:
        return self._rotated_student_local_field_covariances

    @rotated_student_local_field_covariances.setter
    def rotated_student_local_field_covariances(
        self, student_local_field_covariances
    ) -> List[np.ndarray]:
        self._rotated_student_local_field_covariances = student_local_field_covariances

    @property
    def student_weighted_feature_matrix_self_overlaps(self) -> List[np.ndarray]:
        return self._student_weighted_feature_matrix_self_overlaps

    @property
    def w_tilde_tau(self) -> List[np.ndarray]:
        return self._w_tilde_tau

    @student_weighted_feature_matrix_self_overlaps.setter
    def student_weighted_feature_matrix_self_overlaps(
        self, student_weighted_feature_matrix_self_overlaps
    ) -> None:
        self._student_weighted_feature_matrix_self_overlaps = (
            student_weighted_feature_matrix_self_overlaps
        )

    @property
    def rotated_student_weighted_feature_matrix_self_overlaps(self) -> List[np.ndarray]:
        return self._rotated_student_weighted_feature_matrix_self_overlaps

    @rotated_student_weighted_feature_matrix_self_overlaps.setter
    def rotated_student_weighted_feature_matrix_self_overlaps(
        self, student_weighted_feature_matrix_self_overlaps
    ) -> None:
        self._rotated_student_weighted_feature_matrix_self_overlaps = (
            student_weighted_feature_matrix_self_overlaps
        )

    @property
    def rotated_student_teacher_overlaps(self) -> List[np.ndarray]:
        return self._rotated_student_teacher_overlaps

    @rotated_student_teacher_overlaps.setter
    def rotated_student_teacher_overlaps(
        self, rotated_student_teacher_overlaps
    ) -> None:
        self._rotated_student_teacher_overlaps = rotated_student_teacher_overlaps

    @property
    def feature_matrix_overlaps(self) -> List[np.ndarray]:
        return self._feature_matrix_overlaps

    @property
    def feature_matrix_overlap_eigenvalues(self) -> List[np.ndarray]:
        return self._feature_matrix_overlap_eigenvalues

    @property
    def feature_matrix_overlap_eigenvectors(self) -> List[np.ndarray]:
        return self._feature_matrix_overlap_eigenvectors

    @property
    def student_teacher_overlap_densities(self) -> List[np.ndarray]:
        return self._student_teacher_overlap_densities

    @student_teacher_overlap_densities.setter
    def student_teacher_overlap_densities(
        self, student_teacher_overlap_densities
    ) -> None:
        self._student_teacher_overlap_densities = student_teacher_overlap_densities

    @property
    def student_latent_self_overlap_densities(self) -> List[np.ndarray]:
        return self._student_latent_self_overlap_densities

    @student_latent_self_overlap_densities.setter
    def student_latent_self_overlap_densities(
        self, student_latent_self_overlap_densities
    ) -> None:
        self._student_latent_self_overlap_densities = (
            student_latent_self_overlap_densities
        )

    @property
    def projected_teacher_self_overlaps(self) -> List[np.ndarray]:
        return self._projected_teacher_self_overlaps

    # over-write base class
    @property
    def sub_dictionary(self):
        network_configuration_dictionary = {}

        for i, head in enumerate(self._student_head_weights):
            for j, weight in enumerate(head):
                network_configuration_dictionary[
                    f"{constants.STUDENT_HEAD}_{i}_{constants.WEIGHT}_{j}"
                ] = weight
        # for i, head in enumerate(self._teacher_head_weights):
        #     for j, weight in enumerate(head):
        #         network_configuration_dictionary[
        #             f"{constants.TEACHER_HEAD}_{i}_{constants.WEIGHT}_{j}"
        #         ] = weight

        for (i, j), overlap_value in np.ndenumerate(self._student_self_overlap):
            network_configuration_dictionary[
                f"{constants.AMBIENT}_{constants.STUDENT_SELF_OVERLAP}_{i}_{j}"
            ] = overlap_value
        for l, local_field_covariances in enumerate(
            self._rotated_student_local_field_covariances
        ):
            for (i, j), overlap_value in np.ndenumerate(local_field_covariances):
                network_configuration_dictionary[
                    f"{constants.AGGREGATE}_{constants.STUDENT_SELF_OVERLAP}_{l}_{i}_{j}"
                ] = overlap_value
        for l, latent_overlaps in enumerate(
            self._rotated_student_weighted_feature_matrix_self_overlaps
        ):
            for (i, j), overlap_value in np.ndenumerate(latent_overlaps):
                network_configuration_dictionary[
                    f"{constants.LATENT}_{constants.STUDENT_SELF_OVERLAP}_{l}_{i}_{j}"
                ] = overlap_value
        for t, student_teacher_overlap in enumerate(self._student_teacher_overlaps):
            for (i, j), overlap_value in np.ndenumerate(student_teacher_overlap):
                network_configuration_dictionary[
                    f"{constants.STUDENT_TEACHER}_{t}_{constants.OVERLAP}_{i}_{j}"
                ] = overlap_value
        for t, student_teacher_overlap in enumerate(
            self._rotated_student_teacher_overlaps
        ):
            for (i, j), overlap_value in np.ndenumerate(student_teacher_overlap):
                network_configuration_dictionary[
                    f"{constants.ROTATED}_{constants.STUDENT_TEACHER}_{t}_{constants.OVERLAP}_{i}_{j}"
                ] = overlap_value

        return network_configuration_dictionary
