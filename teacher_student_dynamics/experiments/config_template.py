from config_manager import config_field, config_template

from teacher_student_dynamics import constants


class ConfigTemplate:

    _runner_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.RUN_ODE,
                types=[bool],
            ),
            config_field.Field(
                name=constants.RUN_NETWORK,
                types=[bool],
            ),
        ],
        level=[constants.RUNNER],
    )

    _logging_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.CHECKPOINT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.STDOUT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.OVERLAP_FREQUENCY, types=[int, type(None)]
            ),
            config_field.Field(
                name=constants.ODE_LOG_FREQUENCY, types=[int, type(None)]
            ),
        ],
        level=[constants.LOGGING],
    )

    _ode_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.IMPLEMENTATION,
                types=[str],
                requirements=[lambda x: x in [constants.CPP, constants.PYTHON]],
            ),
            config_field.Field(name=constants.OMP, types=[bool]),
            config_field.Field(name=constants.OMP_NUM_THREADS, types=[int]),
            config_field.Field(name=constants.EIGEN_PATH, types=[str, type(None)]),
            config_field.Field(
                name=constants.TIMESTEP, types=[float], requirements=[lambda x: x > 0.0]
            ),
        ],
        level=[constants.ODE],
        dependent_variables=[constants.RUN_ODE],
        dependent_variables_required_values=[[True]],
    )

    _iid_gaussian_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.MEAN,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.VARIANCE,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.DATASET_SIZE,
                types=[str, int],
                requirements=[lambda x: x == constants.INF or x > 0],
            ),
        ],
        level=[constants.DATA, constants.IID_GAUSSIAN],
        dependent_variables=[constants.INPUT_SOURCE],
        dependent_variables_required_values=[[constants.IID_GAUSSIAN]],
    )

    _hidden_manifold_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.MEAN,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.VARIANCE,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.LATENT_DIMENSION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.ACTIVATION,
                types=[str],
                requirements=[lambda x: x in [constants.SCALED_ERF]],
            ),
            config_field.Field(
                name=constants.FEATURE_MATRIX_CORRELATIONS,
                types=[list],
                requirements=[
                    lambda x: all(
                        [isinstance(y, float) and y >= 0.0 and y <= 1.0 for y in x]
                    )
                ],
            ),
        ],
        level=[constants.DATA, constants.HIDDEN_MANIFOLD],
        dependent_variables=[constants.INPUT_SOURCE],
        dependent_variables_required_values=[[constants.HIDDEN_MANIFOLD]],
    )

    _data_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.INPUT_SOURCE,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.IID_GAUSSIAN,
                        constants.HIDDEN_MANIFOLD,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.PRECOMPUTE_DATA,
                types=[int, type(None)],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.NOISE_TO_STUDENT_INPUT,
                types=[list],
                requirements=[
                    lambda x: all(
                        [
                            isinstance(y, list)
                            and (
                                not bool(y)  # empty list
                                or all(
                                    [
                                        isinstance(z, float) or isinstance(z, int)
                                        for z in y
                                    ]
                                )
                            )
                            for y in x
                        ]
                    )
                ],
            ),
            config_field.Field(
                name=constants.NOISE_TO_TEACHER_OUTPUT,
                types=[list],
                requirements=[
                    lambda x: all(
                        [
                            isinstance(y, list)
                            and (
                                not bool(y)  # empty list
                                or all(
                                    [
                                        isinstance(z, float) or isinstance(z, int)
                                        for z in y
                                    ]
                                )
                            )
                            for y in x
                        ]
                    )
                ],
            ),
        ],
        nested_templates=[_iid_gaussian_template, _hidden_manifold_template],
        level=[constants.DATA],
    )

    _training_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TRAIN_BATCH_SIZE,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.TOTAL_TRAINING_STEPS,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.OPTIMISER,
                types=[str],
                requirements=[lambda x: x in [constants.SGD]],
            ),
            config_field.Field(
                name=constants.LEARNING_RATE,
                types=[float],
                requirements=[lambda x: x > 0.0],
            ),
            config_field.Field(
                name=constants.LOSS_FUNCTION,
                types=[str],
                requirements=[lambda x: x in [constants.MSE]],
            ),
            config_field.Field(
                name=constants.TRAIN_HIDDEN_LAYER,
                types=[bool],
            ),
            config_field.Field(
                name=constants.TRAIN_HEAD_LAYER,
                types=[bool],
            ),
            config_field.Field(
                name=constants.COPY_HEAD_AT_SWITCH,
                types=[bool],
            ),
            config_field.Field(
                name=constants.FREEZE_UNITS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) for y in x)],
            ),
        ],
        level=[constants.TRAINING],
    )

    _testing_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TEST_BATCH_SIZE,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.TEST_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.TESTING],
    )

    _rotation_teachers_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.FEATURE_ROTATION_ALPHA,
                types=[float],
                requirements=[lambda x: x <= 0.0 and x >= 0],
            ),
            config_field.Field(
                name=constants.READOUT_ROTATION_ALPHA,
                types=[float],
                requirements=[lambda x: x <= 0.0 and x >= 0],
            ),
        ],
        level=[constants.NETWORKS, constants.ROTATION_TEACHERS],
        dependent_variables=[constants.TEACHER_CONFIGURATION],
        dependent_variables_required_values=[[constants.ROTATION]],
    )

    _node_sharing_teachers_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.NUM_SHARED_NODES,
                types=[int],
                requirements=[lambda x: x >= 1],
            ),
            config_field.Field(
                name=constants.FEATURE_ROTATION_ALPHA,
                types=[float],
                requirements=[lambda x: x <= 0.0 and x >= 0],
            ),
        ],
        level=[constants.NETWORKS, constants.NODE_SHARING_TEACHERS],
        dependent_variables=[constants.TEACHER_CONFIGURATION],
        dependent_variables_required_values=[[constants.NODE_SHARING]],
    )

    _network_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.INPUT_DIMENSION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.OUTPUT_DIMENSION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.NONLINEARITY,
                types=[str],
                requirements=[
                    lambda x: x
                    in [constants.SCALED_ERF, constants.LINEAR, constants.RELU]
                ],
            ),
            config_field.Field(
                name=constants.STUDENT_HIDDEN,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(name=constants.STUDENT_BIAS, types=[bool]),
            config_field.Field(
                name=constants.STUDENT_INITIALISATION_STD,
                types=[float],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(name=constants.MULTI_HEAD, types=[bool]),
            config_field.Field(
                name=constants.NUM_TEACHERS, types=[int], requirements=[lambda x: x > 0]
            ),
            config_field.Field(
                name=constants.TEACHER_HIDDEN,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(name=constants.TEACHER_BIAS, types=[bool]),
            config_field.Field(name=constants.UNIT_NORM_TEACHER_HEAD, types=[bool]),
            config_field.Field(name=constants.NORMALISE_TEACHERS, types=[bool]),
            config_field.Field(
                name=constants.TEACHER_INITIALISATION_STD,
                types=[float],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.TEACHER_CONFIGURATION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.ROTATION,
                        constants.IDENTICAL,
                        constants.NODE_SHARING,
                    ]
                ],
            ),
        ],
        level=[constants.NETWORKS],
        nested_templates=[_rotation_teachers_template, _node_sharing_teachers_template],
    )

    _curriculum_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.STOPPING_CONDITION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.FIXED_PERIOD,
                        constants.LOSS_THRESHOLDS,
                        constants.SWITCH_STEPS,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.FIXED_PERIOD,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.SWITCH_STEPS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) and y >= 0 for y in x)],
            ),
            config_field.Field(
                name=constants.LOSS_THRESHOLDS,
                types=[list],
                requirements=[
                    lambda x: all(
                        (isinstance(y, int) or isinstance(y, float)) and y > 0
                        for y in x
                    )
                ],
            ),
        ],
        level=[constants.CURRICULUM],
    )

    _periodic_replay_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.INTERLEAVE_PERIOD, types=[int]),
            config_field.Field(name=constants.INTERLEAVE_DURATION, types=[int]),
        ],
        dependent_variables=[constants.SCHEDULE],
        dependent_variables_required_values=[[constants.PERIODIC]],
        level=[constants.REPLAY, constants.PERIODIC_REPLAY],
    )

    _gamma_replay_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.GAMMA,
                types=[float],
                requirements=[lambda x: x >= 0 and x <= 1],
            )
        ],
        dependent_variables=[constants.STRATEGY],
        dependent_variables_required_values=[[constants.GAMMA]],
        level=[constants.REPLAY, constants.GAMMA_REPLAY],
    )

    _replay_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.SCHEDULE,
                types=[str, type(None)],
                requirements=[
                    lambda x: x is None
                    or x
                    in [
                        constants.PERIODIC,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.STRATEGY,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.UNIFORM,
                        constants.GAMMA,
                    ]
                ],
            ),
        ],
        level=[constants.REPLAY],
        nested_templates=[_periodic_replay_template, _gamma_replay_template],
    )

    base_config_template = config_template.Template(
        fields=[config_field.Field(name=constants.SEED, types=[int])],
        nested_templates=[
            _runner_template,
            _logging_template,
            _ode_template,
            _data_template,
            _training_template,
            _testing_template,
            _network_template,
            _curriculum_template,
            _replay_template,
        ],
    )
