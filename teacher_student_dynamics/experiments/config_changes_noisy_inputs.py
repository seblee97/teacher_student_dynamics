import itertools

param_config_changes = {
    f"param_{h}": [
        {"networks": {"student_hidden": h, "teacher_hidden": h}},
        {"data": {"noise_to_student_input": [[0.0, n], []]}},
        {"curriculum": {"switch_steps": [s]}},
    ]
    for h, n, s in itertools.product(
        [1, 2, 4], [1.0, 0.1, 0.01, 0.001], [1000, 10000, 100000, 1000000, 10000000]
    )
}
oparam_config_changes = {
    f"oparam_{h}": [
        {"networks": {"student_hidden": 2 * h, "teacher_hidden": h}},
        {"data": {"noise_to_student_input": [[0.0, n], []]}},
        {"curriculum": {"switch_steps": [s]}},
    ]
    for h, n, s in itertools.product(
        [1, 2, 4], [1.0, 0.1, 0.01, 0.001], [1000, 10000, 100000, 1000000, 10000000]
    )
}
CONFIG_CHANGES = {**param_config_changes, **oparam_config_changes}
