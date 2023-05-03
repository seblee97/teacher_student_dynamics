import itertools

oparam_baseline_config_changes = {
    f"oparam_{h}_baseline": [
        {"networks": {"student_hidden": 2 * h, "teacher_hidden": h}},
        {"training": {"freeze_units": [0, 0]}},
        {"curriculum": {"switch_steps": [30000000000]}},
    ]
    for h in [1, 2, 4]
}

param_baseline_config_changes = {
    f"param_{h}_baseline": [
        {"networks": {"student_hidden": h, "teacher_hidden": h}},
        {"training": {"freeze_units": [0, 0]}},
        {"curriculum": {"switch_steps": [30000000000]}},
    ]
    for h in [1, 2, 4]
}

param_config_changes_1 = {
    f"param_{h}_{f}_{s}": [
        {"networks": {"student_hidden": h, "teacher_hidden": h}},
        {"training": {"freeze_units": [f, 0]}},
        {"curriculum": {"switch_steps": [s]}},
    ]
    for h, f, s in itertools.product(
        [1], [1], [100000, 1000000, 5000000, 10000000, 15000000]
    )
}
param_config_changes_2 = {
    f"param_{h}_{f}_{s}": [
        {"networks": {"student_hidden": h, "teacher_hidden": h}},
        {"training": {"freeze_units": [f, 0]}},
        {"curriculum": {"switch_steps": [s]}},
    ]
    for h, f, s in itertools.product(
        [2], [1, 2], [100000, 1000000, 5000000, 10000000, 15000000]
    )
}
param_config_changes_4 = {
    f"param_{h}_{f}_{s}": [
        {"networks": {"student_hidden": h, "teacher_hidden": h}},
        {"training": {"freeze_units": [f, 0]}},
        {"curriculum": {"switch_steps": [s]}},
    ]
    for h, f, s in itertools.product(
        [4], [1, 2, 4], [100000, 1000000, 5000000, 10000000, 15000000]
    )
}

oparam_config_changes_1 = {
    f"oparam_{h}_{f}_{s}": [
        {"networks": {"student_hidden": 2 * h, "teacher_hidden": h}},
        {"training": {"freeze_units": [f, 0]}},
        {"curriculum": {"switch_steps": [s]}},
    ]
    for h, f, s in itertools.product(
        [1], [1, 2], [100000, 1000000, 5000000, 10000000, 15000000]
    )
}
oparam_config_changes_2 = {
    f"oparam_{h}_{f}_{s}": [
        {"networks": {"student_hidden": 2 * h, "teacher_hidden": h}},
        {"training": {"freeze_units": [f, 0]}},
        {"curriculum": {"switch_steps": [s]}},
    ]
    for h, f, s in itertools.product(
        [2], [1, 2, 3, 4], [100000, 1000000, 5000000, 10000000, 15000000]
    )
}

CONFIG_CHANGES = {
    **oparam_baseline_config_changes,
    **param_baseline_config_changes,
    **param_config_changes_1, 
    **param_config_changes_2, 
    **param_config_changes_4, 
    **oparam_config_changes_1, 
    **oparam_config_changes_2
}
