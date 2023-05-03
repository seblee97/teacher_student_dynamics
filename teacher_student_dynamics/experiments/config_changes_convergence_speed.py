import itertools

CONFIG_CHANGES = {
    f"convergence_{th}_{sh}": [
        {"networks": {"student_hidden": sh, "teacher_hidden": th}},
    ] for (sh, th) in itertools.product([1, 2, 4, 8, 16, 32], [1, 2, 3])
}
