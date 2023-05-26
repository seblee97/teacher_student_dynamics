import argparse
import os

import submitit
from run_modes import cluster_run, parallel_run, serial_run, single_run, utils

from teacher_student_dynamics import constants
from teacher_student_dynamics.experiments import config
from teacher_student_dynamics.runners import core_runner

# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"


MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

STOCHASTIC_PACKAGES = ["numpy", "torch", "random"]
RUN_METHODS = ["run", "post_process"]

parser = argparse.ArgumentParser()

parser.add_argument("--mode", metavar="-M", help="run experiment.", default="single")
parser.add_argument(
    "--config_path",
    metavar="-C",
    type=str,
    default="config.yaml",
    help="path to configuration yaml file.",
)
parser.add_argument(
    "--seeds", metavar="-S", default="[0]", help="list of seeds to run."
)
parser.add_argument("--config_changes", metavar="-CC", default="config_changes.py")
parser.add_argument(
    "--results_folder", default=constants.RESULTS, type=str, help="path to all results."
)
parser.add_argument("--cluster_debug", action="store_true")
parser.add_argument("--cluster_debug_run", action="store_true")

# cluster config
parser.add_argument("--scheduler", type=str, help="univa or slurm", default="univa")
parser.add_argument("--num_cpus", type=int, default=4)
parser.add_argument("--num_gpus", type=int, default=0)
parser.add_argument("--mem", type=int, default=16)
parser.add_argument("--timeout", type=str, default="")


if __name__ == "__main__":

    args = parser.parse_args()

    results_folder = os.path.join(MAIN_FILE_PATH, args.results_folder)

    config_class_name = "Config"
    config_module_name = "config"
    config_module_path = os.path.join(MAIN_FILE_PATH, "config.py")
    config_class = config.Config
    config_instance = config_class(args.config_path)

    runner_class_name = "CoreRunner"
    runner_module_name = "core_runner"
    runner_module_path = os.path.join(MAIN_FILE_PATH, "..", "runners", "core_runner.py")
    runner_class = core_runner.CoreRunner

    if args.mode == constants.SINGLE:

        _, single_checkpoint_path = utils.setup_experiment(
            mode=constants.SINGLE,
            results_folder=results_folder,
            config_path=args.config_path,
        )

        single_run.single_run(
            runner_class=runner_class,
            config_class=config_class,
            config_path=args.config_path,
            checkpoint_path=single_checkpoint_path,
            run_methods=RUN_METHODS,
            stochastic_packages=STOCHASTIC_PACKAGES,
        )

    elif args.mode in [
        constants.SINGLE_CLUSTER,
        constants.PARALLEL,
        constants.SERIAL,
        constants.CLUSTER,
        constants.SUBMITIT,
    ]:

        if constants.SINGLE in args.mode:
            config_changes_path = None
        else:
            config_changes_path = args.config_changes

        try:
            if args.seeds is not None:
                seeds_arg = int(args.seeds)
            else:
                seeds_arg = args.seeds
        except ValueError:
            seeds_arg = args.seeds

        seeds = utils.process_seed_arguments(seeds_arg)

        experiment_path, checkpoint_paths = utils.setup_experiment(
            mode="multi",
            results_folder=results_folder,
            config_path=args.config_path,
            config_changes_path=config_changes_path,
            seeds=seeds,
        )

        if args.mode == constants.PARALLEL:

            parallel_run.parallel_run(
                runner_class=runner_class,
                config_class=config_class,
                config_path=args.config_path,
                checkpoint_paths=checkpoint_paths,
                run_methods=RUN_METHODS,
                stochastic_packages=STOCHASTIC_PACKAGES,
            )

        elif args.mode == constants.SERIAL:

            serial_run.serial_run(
                runner_class=runner_class,
                config_class=config_class,
                config_path=args.config_path,
                checkpoint_paths=checkpoint_paths,
                run_methods=RUN_METHODS,
                stochastic_packages=STOCHASTIC_PACKAGES,
            )

        elif constants.CLUSTER in args.mode:

            cluster_run.cluster_run(
                runner_class_name=runner_class_name,
                runner_module_name=runner_module_name,
                runner_module_path=runner_module_path,
                config_class_name=config_class_name,
                config_module_name=config_module_name,
                config_module_path=config_module_path,
                config_path=args.config_path,
                checkpoint_paths=checkpoint_paths,
                run_methods=RUN_METHODS,
                stochastic_packages=STOCHASTIC_PACKAGES,
                env_name="ts_dyn",
                scheduler=args.scheduler,
                num_cpus=args.num_cpus,
                num_gpus=args.num_gpus,
                memory=args.mem,
                walltime=args.timeout,
                cluster_debug=args.cluster_debug,
                cluster_debug_run=args.cluster_debug_run,
            )

        elif args.mode == constants.SUBMITIT:

            # _, single_checkpoint_path = utils.setup_experiment(
            #     mode=constants.SINGLE,
            #     results_folder=results_folder,
            #     config_path=args.config_path,
            # )

            # single_run.single_run(
            #     runner_class=runner_class,
            #     config_class=config_class,
            #     config_path=args.config_path,
            #     checkpoint_path=single_checkpoint_path,
            #     run_methods=RUN_METHODS,
            #     stochastic_packages=STOCHASTIC_PACKAGES,
            # )

            executor = submitit.AutoExecutor(folder=experiment_path)

            if args.num_gpus > 0:
                partition = "gpu"
            else:
                partition = "cpu"

            executor.update_parameters(
                timeout_min=args.timeout,
                mem_gb=args.mem,
                gpus_per_node=args.num_gpus,
                cpus_per_task=args.num_cpus,
                slurm_partition=partition,
            )

            jobs = []

            for checkpoint_path in checkpoint_paths:
                executor.update_parameters(folder=checkpoint_path)
                changes = utils.json_to_config_changes(
                    os.path.join(checkpoint_path, constants.CONFIG_CHANGES_JSON)
                )
                job = executor.submit(
                    single_run.single_run,
                    runner_class,
                    config_class,
                    RUN_METHODS,
                    args.config_path,
                    checkpoint_path,
                    changes,
                    STOCHASTIC_PACKAGES,
                )

    else:
        raise ValueError(f"run mode {args.mode} not recognised.")
