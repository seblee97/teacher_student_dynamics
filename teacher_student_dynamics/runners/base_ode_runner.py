import abc
import os
import subprocess
import time

from run_modes import base_runner

from teacher_student_dynamics import constants, experiments
from teacher_student_dynamics.utils import network_configurations


class BaseODERunner(base_runner.BaseRunner):
    """Runner for ode simulations.

    Class for orchestrating student teacher ODE equation solvers.
    """

    def __init__(
        self,
        config: experiments.config.Config,
        network_configuration: network_configurations.VanillaNetworkConfiguration,
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
            os.makedirs(self._ode_file_path, exist_ok=True)

            self._cpp_out_path = os.path.join(self._ode_file_path, "ode_runner.out")
            self._txt_config_path = os.path.join(self._ode_file_path, "ode_config.txt")

            self._logger.info("Constructing ODE configuration...")
            self._construct_ode_config(
                config=config
            )
            self._logger.info("ODE configuration constucted.")

        else:
            raise NotImplementedError()

        self._logger.info("ODE runner setup.")

    def compile(self):
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

    def run(self):
        subprocess.call([self._cpp_out_path, self._txt_config_path])

    @abc.abstractmethod
    def _construct_ode_config(
        self,
        config: experiments.config.Config,
    ):
        pass

    @abc.abstractmethod
    def consolidate_outputs(self):
        pass
