import os

import pandas as pd

from teacher_student_dynamics import constants, experiments
from teacher_student_dynamics.runners import (
    base_network_runner,
    ode_runner,
    vanilla_multi_teacher_runner,
)
from teacher_student_dynamics.utils import plotting_functions


class CoreRunner:
    def __init__(
        self,
        config: experiments.config.Config,
        unique_id: str = "",
    ) -> None:
        """Class for orchestrating student teacher framework run.

        Initialise specific ODE simulation and network simulation runners.

        Args:
            config: configuration object specifying experiment setup.
        """
        self._run_network = config.run_network
        self._run_ode = config.run_ode

        self._checkpoint_path = config.checkpoint_path

        self._setup_runners(config=config, unique_id=unique_id)

    def _setup_runners(self, config: experiments.config.Config, unique_id: str = ""):
        # always generate initial configuration from torch networks,
        if unique_id == "":
            network_id = constants.NETWORK
            ode_id = constants.ODE
        else:
            network_id = f"{unique_id}_{constants.NETWORK}"
            ode_id = f"{unique_id}_{constants.ODE}"

        self._network_runner = vanilla_multi_teacher_runner.VanillaMultiTeacherRunner(
            config=config, unique_id=network_id
        )

        if self._run_ode:
            network_configuration = self._network_runner.get_network_configuration()
            self._ode_runner = ode_runner.ODERunner(
                config=config,
                unique_id=ode_id,
                network_configuration=network_configuration,
            )

    def run(self):
        if self._run_network:
            self._network_runner.train()
        if self._run_ode:
            self._ode_runner.run()
            self._ode_runner.consolidate_outputs()

    def post_process(self):
        plot_path = os.path.join(self._checkpoint_path, constants.PLOTS)
        os.mkdir(plot_path)
        if self._run_network:
            network_df = pd.read_csv(self._network_runner.logfile_path)
            plotting_functions.plot_all_scalars(
                df=network_df, path=plot_path, prefix=constants.NETWORK
            )
        if self._run_ode:
            ode_df = pd.read_csv(self._ode_runner.logfile_path)
            plotting_functions.plot_all_scalars(
                df=ode_df, path=plot_path, prefix=constants.ODE
            )
        if self._run_network and self._run_ode:
            plotting_functions.plot_all_common_scalars(
                dfs=[network_df, ode_df],
                path=plot_path,
                prefix=constants.OVERLAY,
                sub_prefixes=[constants.NETWORK, constants.ODE],
                styles=[
                    {constants.COLOR: "blue", constants.LINESTYLE: constants.DASHED},
                    {constants.COLOR: "blue", constants.LINESTYLE: constants.SOLID},
                ],
            )
