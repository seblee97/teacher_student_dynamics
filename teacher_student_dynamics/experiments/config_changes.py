import itertools
import numpy as np

CONFIG_CHANGES = {
    f"steps_{s}_alpha_{a}": [
        {
            "curriculum": {"switch_steps": [int(s)]},
            "networks": {"rotation_teachers": {"feature_rotation_alpha": a}},
        }
    ]
    for s, a in itertools.product(np.linspace(200000, 400000, 3), [1.0])
}
