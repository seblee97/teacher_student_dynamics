from typing import Dict, List, Union

from config_manager import base_configuration

from teacher_student_dynamics import constants
from teacher_student_dynamics.experiments.config_template import ConfigTemplate

import numpy as np

class Config(base_configuration.BaseConfiguration):
    def __init__(self, config: Union[str, Dict], changes: List[Dict] = []) -> None:
        super().__init__(
            configuration=config,
            template=ConfigTemplate.base_config_template,
            changes=changes,
        )
        self._validate_configuration()

    def _validate_configuration(self): #TODO: int try catch in here for HMM
        """Method to check for non-trivial associations
        in the configuration.
        """
        num_tasks = len(self.feature_matrix_correlations)+1
        d = self.latent_dimension/(num_tasks - np.sum(self.feature_matrix_correlations))
        if self.run_ode:
            assert (
                self.implementation == constants.CPP
            ), "ODEs currently implemented in C++ only."
            assert self.input_source in [
                constants.IID_GAUSSIAN,
                constants.HIDDEN_MANIFOLD,
            ], "ODEs implemented for IID Gaussian inputs or the Hidden Manifold Model only."
            if self.input_source == constants.IID_GAUSSIAN:
                assert (
                    self.dataset_size == constants.INF
                ), "ODEs implemented for online learning (infinite dataset size) only."
            assert [
                len(i) == 0 or i[0] == 0.0 for i in self.noise_to_student_input
            ], "ODEs implemented for 0-centered noise on student inputs only."
            assert [
                len(i) == 0 or i[0] == 0.0 for i in self.noise_to_teacher_output
            ], "ODEs implemented for 0-centered noise on teachers only."
            assert (
                self.train_batch_size == 1
            ), "ODEs implemented for online learning (train batch size 1) only."
            assert (
                self.optimiser == constants.SGD
            ), "ODEs implemented for SGD optimiser only."
            assert (
                self.nonlinearity == constants.SCALED_ERF
            ), "ODEs implemented for scaled error function activation only."
            assert (
                self.output_dimension == 1
            ), "ODEs implemented for regression with unit output dimension only."
            assert (
                not self.student_bias and not self.teacher_bias
            ), "ODEs implemented for networks without bias only."
            assert self.schedule is None, "Interleaved replay not implemented for ODEs."
            assert (
               float(int(d)) - d == 0
            ), "Latent Dimension should be chosen such that task dimension is an integer."
