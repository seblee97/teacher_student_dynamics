from teacher_student_dynamics import experiments
from teacher_student_dynamics.runners import base_network_runner


class HMMMultiTeacher(base_network_runner.BaseNetworkRunner):
    """Implementation of hidden manifold model with multiple teachers.

    Extension of Goldt et. al (https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.041044).
    """

    def __init__(self, config: experiments.config.Config, unique_id: str = "") -> None:
        super().__init__(config, unique_id)

    def get_network_configuration(self):
        raise NotImplementedError()

    def _get_data_columns(self):
        raise NotImplementedError()

    def _setup_teachers(self, config: experiments.config.Config):
        """Initialise teacher object containing teacher networks."""
        raise NotImplementedError()

    def _setup_student(self, config: experiments.config.Config):
        """Initialise object containing student network."""
        raise NotImplementedError()

    def _setup_data(self, config: experiments.config.Config):
        """Initialise data module."""
        raise NotImplementedError()
