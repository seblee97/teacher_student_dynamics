import os
from typing import Dict, List

import numpy as np

from teacher_student_dynamics import constants


class VanillaNetworkConfiguration:
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

    def save(self, path: str) -> Dict[str, str]:

        save_path_map = {}

        np.savetxt(
            os.path.join(path, f"{constants.STUDENT_SELF_OVERLAP}.csv"),
            self._student_self_overlap,
            delimiter=",",
        )
        save_path_map[constants.STUDENT_SELF_OVERLAP] = os.path.join(
            path, f"{constants.STUDENT_SELF_OVERLAP}.csv"
        )
        for i, student_teacher_overlap in enumerate(self._student_teacher_overlaps):
            op_name = f"{constants.STUDENT_TEACHER_OVERLAPS}_{i}"
            save_path = os.path.join(path, f"{op_name}.csv")
            np.savetxt(
                save_path,
                student_teacher_overlap,
                delimiter=",",
            )
            save_path_map[op_name] = save_path
        for i, student_head in enumerate(self._student_head_weights):
            op_name = f"{constants.STUDENT_HEAD_WEIGHTS}_{i}"
            save_path = os.path.join(path, f"{op_name}.csv")
            np.savetxt(
                save_path,
                student_head,
                delimiter=",",
            )
            save_path_map[op_name] = save_path
        for i in range(self._num_teachers):
            op_name = f"{constants.TEACHER_HEAD_WEIGHTS}_{i}"
            save_path = os.path.join(path, f"{op_name}.csv")
            np.savetxt(
                os.path.join(path, f"{constants.TEACHER_HEAD_WEIGHTS}_{i}.csv"),
                self._teacher_head_weights[i],
                delimiter=",",
            )
            save_path_map[op_name] = save_path
            op_name = f"{constants.TEACHER_SELF_OVERLAPS}_{i}"
            save_path = os.path.join(path, f"{op_name}.csv")
            np.savetxt(
                os.path.join(path, f"{constants.TEACHER_SELF_OVERLAPS}_{i}.csv"),
                self._teacher_self_overlaps[i],
                delimiter=",",
            )
            save_path_map[op_name] = save_path

        access_index = 0
        for i in range(self._num_teachers):
            for j in range(i, self._num_teachers):
                if i != j:
                    op_name = f"{constants.TEACHER_CROSS_OVERLAPS}_{i}_{j}"
                    save_path = os.path.join(path, f"{op_name}.csv")
                    np.savetxt(
                        os.path.join(
                            path, f"{constants.TEACHER_CROSS_OVERLAPS}_{i}_{j}.csv"
                        ),
                        self._teacher_cross_overlaps[access_index],
                        delimiter=",",
                    )
                    save_path_map[op_name] = save_path
                    access_index += 1

        return save_path_map

    @property
    def student_head_weights(self) -> List[np.ndarray]:
        return self._student_head_weights

    @property
    def teacher_head_weights(self) -> List[np.ndarray]:
        return self._teacher_head_weights

    @property
    def student_self_overlap(self) -> np.ndarray:
        return self._student_self_overlap

    @property
    def teacher_self_overlaps(self) -> List[np.ndarray]:
        return self._teacher_self_overlaps

    @property
    def teacher_cross_overlaps(self) -> List[np.ndarray]:
        return self._teacher_cross_overlaps

    @property
    def student_teacher_overlaps(self) -> List[np.ndarray]:
        return self._student_teacher_overlaps

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


class HiddenManifoldNetworkConfiguration:
    """object to store configuration of student/teacher networks in unified way."""

    def __init__(
        self,
        student_head_weights: List[np.ndarray],
        teacher_head_weights: List[np.ndarray],
        student_self_overlap: np.ndarray,
        teacher_self_overlaps: List[np.ndarray],
        teacher_cross_overlaps: List[np.ndarray],
        student_teacher_overlaps: List[np.ndarray],
        student_weighted_feature_matrices: List[np.ndarray],
        student_local_field_covariances: List[np.ndarray],
        student_weighted_feature_matrix_self_overlaps=List[np.ndarray],
    ):
        self._student_head_weights = student_head_weights
        self._teacher_head_weights = teacher_head_weights
        self._student_self_overlap = student_self_overlap
        self._teacher_self_overlaps = teacher_self_overlaps
        self._teacher_cross_overlaps = teacher_cross_overlaps
        self._student_teacher_overlaps = student_teacher_overlaps
        self._student_weighted_feature_matrices = student_weighted_feature_matrices
        self._student_local_field_covariances = student_local_field_covariances
        self._student_weighted_feature_matrix_self_overlaps = (
            student_weighted_feature_matrix_self_overlaps
        )

        self._num_teachers = len(self._teacher_head_weights)

    def save(self, path: str) -> Dict[str, str]:

        save_path_map = {}

        np.savetxt(
            os.path.join(path, f"{constants.STUDENT_SELF_OVERLAP}.csv"),
            self._student_self_overlap,
            delimiter=",",
        )
        save_path_map[constants.STUDENT_SELF_OVERLAP] = os.path.join(
            path, f"{constants.STUDENT_SELF_OVERLAP}.csv"
        )
        for i, student_teacher_overlap in enumerate(self._student_teacher_overlaps):
            op_name = f"{constants.STUDENT_TEACHER_OVERLAPS}_{i}"
            save_path = os.path.join(path, f"{op_name}.csv")
            np.savetxt(
                save_path,
                student_teacher_overlap,
                delimiter=",",
            )
            save_path_map[op_name] = save_path
        for i, student_head in enumerate(self._student_head_weights):
            op_name = f"{constants.STUDENT_HEAD_WEIGHTS}_{i}"
            save_path = os.path.join(path, f"{op_name}.csv")
            np.savetxt(
                save_path,
                student_head,
                delimiter=",",
            )
            save_path_map[op_name] = save_path
        for i in range(self._num_teachers):
            op_name = f"{constants.TEACHER_HEAD_WEIGHTS}_{i}"
            save_path = os.path.join(path, f"{op_name}.csv")
            np.savetxt(
                os.path.join(path, f"{constants.TEACHER_HEAD_WEIGHTS}_{i}.csv"),
                self._teacher_head_weights[i],
                delimiter=",",
            )
            save_path_map[op_name] = save_path
            op_name = f"{constants.TEACHER_SELF_OVERLAPS}_{i}"
            save_path = os.path.join(path, f"{op_name}.csv")
            np.savetxt(
                os.path.join(path, f"{constants.TEACHER_SELF_OVERLAPS}_{i}.csv"),
                self._teacher_self_overlaps[i],
                delimiter=",",
            )
            save_path_map[op_name] = save_path

        access_index = 0
        for i in range(self._num_teachers):
            for j in range(i, self._num_teachers):
                if i != j:
                    op_name = f"{constants.TEACHER_CROSS_OVERLAPS}_{i}_{j}"
                    save_path = os.path.join(path, f"{op_name}.csv")
                    np.savetxt(
                        os.path.join(
                            path, f"{constants.TEACHER_CROSS_OVERLAPS}_{i}_{j}.csv"
                        ),
                        self._teacher_cross_overlaps[access_index],
                        delimiter=",",
                    )
                    save_path_map[op_name] = save_path
                    access_index += 1

        return save_path_map

    @property
    def student_head_weights(self) -> List[np.ndarray]:
        return self._student_head_weights

    @property
    def teacher_head_weights(self) -> List[np.ndarray]:
        return self._teacher_head_weights

    @property
    def student_self_overlap(self) -> np.ndarray:
        return self._student_self_overlap

    @property
    def teacher_self_overlaps(self) -> List[np.ndarray]:
        return self._teacher_self_overlaps

    @property
    def teacher_cross_overlaps(self) -> List[np.ndarray]:
        return self._teacher_cross_overlaps

    @property
    def student_teacher_overlaps(self) -> List[np.ndarray]:
        return self._student_teacher_overlaps

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
