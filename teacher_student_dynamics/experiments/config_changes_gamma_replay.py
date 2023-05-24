import itertools

CONFIG_CHANGES = {
    f"gamma_{g}_{p}": [
        {
            "replay": {
                "periodic_replay": {"interleave_period": p},
                "gamma_replay": {"gamma": g},
            }
        },
    ]
    for (g, p) in itertools.product([0.0, 0.25, 0.5, 0.75, 1.0], [1, 5, 10])
}
