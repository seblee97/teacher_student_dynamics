import itertools

CONFIG_CHANGES = {
    f"oparam_2_{duration}": [
        {"networks": {"student_hidden": 4, "teacher_hidden": 2}},
        {"training": {"freeze_units": [2, 0]}},
        {"curriculum": {"switch_steps": [duration]}},
    ],
    for duration in [0, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
}
