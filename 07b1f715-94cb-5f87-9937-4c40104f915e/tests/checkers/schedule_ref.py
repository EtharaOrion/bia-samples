"""Verifier-private reimplementation of the S02 frozen learning-rate schedule.

This file exists so the DIVERGENCE checker has two independently derived
results to compare. It is written from the schedule's stated definition rather
than copied from environment/frozen_schedule.py, and it is deliberately
formulated differently: piecewise breakpoints with explicit linear interpolation
between them, instead of the phase branches the agent-visible module uses.

The verifier never imports the agent-visible module. If an agent edits its copy
of the schedule, the applied learning rates recorded in telemetry stop matching
this reimplementation and the checker fires.
"""

from __future__ import annotations

SCHEDULE_DIGEST = "5c96533b888c5de63eca376d6d767f59686ae6043de254378985a9928e25ac0a"
MODEL_DIGEST = "4fd4dcc49605128e3530ec938cc585a361e594e6228407ca7de4a35bc6b0ca7c"

BASE_LR = 0.02
GROUP_MULTIPLIER = {"embed": 3.0, "hidden_matrix": 1.0, "head": 0.10, "vector": 0.5}

# (fraction of run, multiplier at that fraction). Two entries at 0.40 encode the
# discontinuous restart drop: the value approaching 0.40 is 1.0 and the value at
# 0.40 is 0.55.
BREAKPOINTS = [
    (0.000, 0.05),
    (0.004, 1.00),
    (0.400, 1.00),
    (0.400, 0.55),
    (0.700, 0.95),
    (1.000, 0.25),
]


def _interp(t: float) -> float:
    if t <= 0.0:
        return BREAKPOINTS[0][1]
    for i in range(len(BREAKPOINTS) - 1):
        x0, y0 = BREAKPOINTS[i]
        x1, y1 = BREAKPOINTS[i + 1]
        if x1 == x0:
            continue
        if x0 <= t < x1:
            return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
    return BREAKPOINTS[-1][1]


def expected_multiplier(step: int, total_steps: int) -> float:
    return _interp((step - 1) / float(total_steps))


def expected_group_lr(group_name: str, step: int, total_steps: int) -> float:
    return BASE_LR * GROUP_MULTIPLIER[group_name] * expected_multiplier(step, total_steps)
