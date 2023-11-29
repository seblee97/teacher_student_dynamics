import os

import numpy as np
import pandas as pd

from teacher_student_dynamics import constants, experiments
from teacher_student_dynamics.ode import c_ode
from teacher_student_dynamics.runners import base_ode_runner
from teacher_student_dynamics.utils import cpp_utils, network_configurations


class VanillaODERunner(base_ode_runner.BaseODERunner):
    def __init__(
        self,
        config: experiments.config.Config,
        network_configuration: network_configurations.VanillaNetworkConfiguration,
        unique_id: str = "",
    ) -> None:

        self._ode_runner_path = os.path.join(
            os.path.dirname(c_ode.__file__), constants.VANILLA, constants.CPP_RUNNER
        )

        super().__init__(
            config=config,
            network_configuration=network_configuration,
            unique_id=unique_id,
        )

    def _get_data_columns(self):
        return []

    def _construct_ode_config(
        self,
        config: experiments.config.Config
    ):
        """Method to format/save subset of configuration relevant to ODE."""

        order_param_path = os.path.join(self._ode_file_path, "order_parameter.txt")

        assert all(
            [not len(n) or (n[0] == 0) for n in config.noise_to_teacher_output]
        ), "ODEs only hold for 0-centred Gaussian noise."
        teacher_output_noises = [
            n[1] if len(n) else 0.0 for n in config.noise_to_teacher_output
        ]

        assert all(
            [not len(n) or (n[0] == 0) for n in config.noise_to_student_input]
        ), "ODEs only hold for 0-centred Gaussian noise in student input."
        student_input_noises = [
            n[1] if len(n) else 0.0 for n in config.noise_to_student_input
        ]

        ode_config = {
            constants.NUM_STEPS: config.total_training_steps,
            constants.ODE_LOG_FREQUENCY: config.ode_log_frequency,
            constants.SWITCH_STEP: config.switch_steps[0],
            constants.HIDDEN_LEARNING_RATE: config.learning_rate,
            constants.HEAD_LEARNING_RATE: config.learning_rate,
            constants.INPUT_DIMENSION: config.input_dimension,
            constants.STUDENT_HIDDEN: config.student_hidden,
            constants.TEACHER_HIDDEN: config.teacher_hidden,
            constants.MULTI_HEAD: config.multi_head,
            constants.TIMESTEP: config.timestep,
            constants.TRAIN_HEAD_LAYER: config.train_head_layer,
            constants.TRAIN_HIDDEN_LAYER: config.train_hidden_layer,
            constants.INPUT_NOISE_STDS: student_input_noises,
            constants.NOISE_STDS: teacher_output_noises,
            constants.FREEZE_UNITS: config.freeze_units,
            constants.ORDER_PARAMETER_PATHS: order_param_path,
            constants.OUTPUT_PATH: self._ode_file_path,
            constants.OMP_NUM_THREADS: config.omp_num_threads,
            constants.STDOUT_FREQUENCY: config.stdout_frequency,
            constants.STDOUT_PATH: os.path.join(
                self._checkpoint_path, constants.ODE_LOG_FILE_NAME
            ),
        }

        cpp_utils.params_to_txt(params=ode_config, output_path=self._txt_config_path)

    def consolidate_outputs(self):

        df = pd.DataFrame()

        log_csvs_path = os.path.join(self._ode_file_path, constants.LOG_CSVS)

        for csv_path in os.listdir(log_csvs_path):
            if csv_path.endswith(".csv"):
                column = csv_path.split(".")[0]
                df[column] = np.genfromtxt(os.path.join(log_csvs_path, csv_path))

        df.to_csv(self._logfile_path, index=False)
