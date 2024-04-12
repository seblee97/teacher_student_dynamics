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
    f"replay_gamma_{a}_{b}": [
{           "data": {"hidden_manifold": {"feature_matrix_correlations": [float(b)]}},
            "replay": {"gamma_replay": {"gamma": float(a)}},
        }
    ]
    for a, b, in itertools.product(
        [0.0,1.0],[0.0,0.1,0.2,]
    )
}

