import itertools

oparam_baseline_config_changes = {
    f"oparam_{h}_baseline": [
        {"networks": {"student_hidden": 2 * h, "teacher_hidden": h}},
        {"data": {"noise_to_student_input": [[], []]}},
        {"curriculum": {"switch_steps": [30000000000]}},
    ]
    for h in [1, 2, 4]
}

param_baseline_config_changes = {
    f"param_{h}_baseline": [
        {"networks": {"student_hidden": h, "teacher_hidden": h}},
        {"data": {"noise_to_student_input": [[], []]}},
        {"curriculum": {"switch_steps": [30000000000]}},
    ]
    for h in [1, 2, 4]
}

param_config_changes = {
    f"param_{h}_{n}_{s}": [
        {"networks": {"student_hidden": h, "teacher_hidden": h}},
        {"data": {"noise_to_student_input": [[0.0, n], []]}},
        {"curriculum": {"switch_steps": [s]}},
    ]
    for h, n, s in itertools.product(
        [1, 2, 4], [10.0, 5.0, 1.0, 0.1, 0.01], [100000, 1000000, 5000000, 10000000, 15000000]
    )
}
oparam_config_changes = {
    f"oparam_{h}_{n}_{s}": [
        {"networks": {"student_hidden": 2 * h, "teacher_hidden": h}},
        {"data": {"noise_to_student_input": [[0.0, n], []]}},
        {"curriculum": {"switch_steps": [s]}},
    ]
    for h, n, s in itertools.product(
        [1, 2], [10.0, 5.0, 1.0, 0.1, 0.01], [100000, 1000000, 5000000, 10000000, 15000000]
    )
}

CONFIG_CHANGES = {
    **oparam_baseline_config_changes, 
    **param_baseline_config_changes, 
    **param_config_changes, 
    **oparam_config_changes
}
