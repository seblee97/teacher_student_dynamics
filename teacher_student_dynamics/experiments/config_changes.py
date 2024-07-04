import itertools
import numpy as np

param_config_changes = {
    f"param_{h}": [{"networks": {"student_hidden": h, "teacher_hidden": h}}]
    for h in [1, 2, 4]
}
oparam_config_changes = {
    f"oparam_{h}": [{"networks": {"student_hidden": 2 * h, "teacher_hidden": h}}]
    for h in [1, 2, 4]
}
# CONFIG_CHANGES = {**param_config_changes, **oparam_config_changes}

CONFIG_CHANGES = {
    f"init_{a}_{b}": [
        {"networks": {"student_head_initialisation_std": [float(a), float(b)]}}
    ]
    for a, b in itertools.product(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
}
