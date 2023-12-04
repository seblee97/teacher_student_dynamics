import os
import tempfile
import unittest

from teacher_student_dynamics.experiments import config
from teacher_student_dynamics.runners import core_runner
from run_modes import single_run
from run_modes import utils

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
RUNNER_TEST_DIR = os.path.dirname(os.path.realpath(__file__))

TEST_CONFIG_PATH = os.path.join(MAIN_FILE_PATH, "base_test_config.yaml")


def _test_runner(changes):

    results_path = tempfile.mkdtemp()

    config_class = config.Config
    runner_class = core_runner.CoreRunner

    _, single_checkpoint_path = utils.setup_experiment(
        mode="single", results_folder=results_path, config_path=TEST_CONFIG_PATH
    )

    single_run.single_run(
        runner_class=runner_class,
        config_class=config_class,
        config_path=TEST_CONFIG_PATH,
        checkpoint_path=single_checkpoint_path,
        run_methods=["run", "post_process"],
        changes=changes,
    )


class TestRunner(unittest.TestCase):

    def test_base(self):
        _test_runner(changes=[])

    def test_vanilla_ode_only(self):
        changes = [{"runner": {"run_network": False}}]
        _test_runner(changes=changes)

    def test_network_only(self):
        changes=[{"runner": {"run_ode": False}}]
        _test_runner(changes=changes)