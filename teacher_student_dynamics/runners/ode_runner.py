import os
import subprocess
import time

import numpy as np
import pandas as pd
from run_modes import base_runner

from teacher_student_dynamics import constants, experiments
from teacher_student_dynamics.ode import c_ode
from teacher_student_dynamics.utils import cpp_utils, network_configuration


class ODERunner(base_runner.BaseRunner):
    """Runner for ode simulations.

    Class for orchestrating student teacher ODE equation solvers.
    """

    def __init__(
        self,
        config: experiments.config.Config,
        network_configuration: network_configuration.NetworkConfiguration,
        unique_id: str = "",
    ) -> None:
        super().__init__(config=config, unique_id=unique_id)

        self._logger.info("Setting up ODE runner...")

        self._student_hidden = config.student_hidden
        self._teacher_hidden = config.teacher_hidden

        self._num_teachers = config.num_teachers
        self._multi_head = config.multi_head

        if config.implementation == constants.CPP:

            self._omp = config.omp
            self._eigen_path = config.eigen_path

            self._ode_file_path = os.path.join(
                self._checkpoint_path, constants.ODE_FILES
            )
            os.mkdir(self._ode_file_path)

            self._cpp_out_path = os.path.join(self._ode_file_path, "ode_runner.out")
            self._txt_config_path = os.path.join(self._ode_file_path, "ode_config.txt")
            self._ode_runner_path = os.path.join(
                os.path.dirname(c_ode.__file__), "runner.cpp"
            )

            self._logger.info("Constructing ODE configuration...")
            self._construct_ode_config(
                config=config, network_configuration=network_configuration
            )
            self._logger.info("ODE configuration constucted.")

        else:
            raise NotImplementedError()

        self._logger.info("ODE runner setup.")

    def _get_data_columns(self):
        return []

    def _construct_ode_config(
        self,
        config: experiments.config.Config,
        network_configuration: network_configuration.NetworkConfiguration,
    ):
        """Method to format/save subset of configuration relevant to ODE."""

        order_params = {
            "Q.csv": network_configuration.student_self_overlap,
            "R.csv": network_configuration.student_teacher_overlaps[0],
            "U.csv": network_configuration.student_teacher_overlaps[1],
            "T.csv": network_configuration.teacher_self_overlaps[0],
            "S.csv": network_configuration.teacher_self_overlaps[1],
            "V.csv": network_configuration.teacher_cross_overlaps[0],
            "h1.csv": network_configuration.student_head_weights[0],
            "th1.csv": network_configuration.teacher_head_weights[0],
            "th2.csv": network_configuration.teacher_head_weights[1],
        }

        if config.multi_head:
            order_params["h2.csv"] = network_configuration.student_head_weights[1]

        order_param_path = os.path.join(self._ode_file_path, "order_parameter.txt")

        with open(order_param_path, "+w") as txt_file:
            for k, v in order_params.items():
                op_csv_path = os.path.join(self._ode_file_path, k)
                np.savetxt(op_csv_path, v, delimiter=",")
                param_name = k.split(".")[0]
                txt_file.write(f"{param_name},{op_csv_path}\n")

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

    def run(self):

        call_list = [
            "g++",
            "-pedantic",
            "-std=c++17",
            self._ode_runner_path,
            "-o",
            self._cpp_out_path,
        ]

        if self._eigen_path is not None:
            call_list.insert(3, "-I")
            call_list.insert(4, self._eigen_path)

        if self._omp:
            call_list.append("-fopenmp")

        pre_compile = time.time()
        self._logger.info("Compiling C++ ODE implementation...")
        subprocess.call(
            call_list,
            shell=False,
        )
        self._logger.info(
            f"C++ ODE implementation compiled in {round(time.time() - pre_compile, 3)}s."
        )

        subprocess.call([self._cpp_out_path, self._txt_config_path])

    def consolidate_outputs(self):

        df = pd.DataFrame()

        for i in range(self._student_hidden):
            for j in range(self._student_hidden):
                qij = np.genfromtxt(os.path.join(self._ode_file_path, f"q_{i}{j}.csv"))
                df[f"{constants.STUDENT_SELF_OVERLAP}_{i}_{j}"] = qij

        for i in range(self._student_hidden):
            for j in range(self._teacher_hidden):
                rij = np.genfromtxt(os.path.join(self._ode_file_path, f"r_{i}{j}.csv"))
                uij = np.genfromtxt(os.path.join(self._ode_file_path, f"u_{i}{j}.csv"))
                df[f"{constants.STUDENT_TEACHER}_{0}_{constants.OVERLAP}_{i}_{j}"] = rij
                df[f"{constants.STUDENT_TEACHER}_{1}_{constants.OVERLAP}_{i}_{j}"] = uij

        for i in range(self._num_teachers):

            ei = np.genfromtxt(os.path.join(self._ode_file_path, f"error_{i}.csv"))
            df[f"{constants.GENERALISATION_ERROR}_{i}"] = ei

            log_ei = np.log10(ei)
            df[f"{constants.LOG_GENERALISATION_ERROR}_{i}"] = log_ei

        if self._multi_head:
            for i in range(self._num_teachers):
                for j in range(self._student_hidden):
                    hij = np.genfromtxt(
                        os.path.join(self._ode_file_path, f"h_{i}{j}.csv")
                    )
                    df[f"{constants.STUDENT_HEAD}_{i}_{constants.WEIGHT}_{j}"] = hij
        else:
            for j in range(self._student_hidden):
                h0j = np.genfromtxt(os.path.join(self._ode_file_path, f"h_0{j}.csv"))
                df[f"{constants.STUDENT_HEAD}_0_{constants.WEIGHT}_{j}"] = h0j

        df.to_csv(self._logfile_path, index=False)
