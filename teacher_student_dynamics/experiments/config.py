from typing import Dict
from typing import List
from typing import Union

from teacher_student_dynamics.experiments.config_template import ConfigTemplate
from config_manager import base_configuration


class Config(base_configuration.BaseConfiguration):
    def __init__(self, config: Union[str, Dict], changes: List[Dict] = []) -> None:
        super().__init__(
            configuration=config,
            template=ConfigTemplate.base_config_template,
            changes=changes,
        )
        self._validate_configuration()

    def _validate_configuration(self):
        """Method to check for non-trivial associations
        in the configuration.
        """
        pass
