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
#CONFIG_CHANGES = {**param_config_changes, **oparam_config_changes}

CONFIG_CHANGES = {
    f"replay_gamma_{a}": [
        {
            "replay": {"gamma_replay": {"gamma": [float(a)]}},
        }
    ]
    for a in np.linspace(0, 1, 21)
}
