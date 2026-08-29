"""BIA S02 frozen learning-rate schedule.

PUBLISHED AND FROZEN. This module is agent-visible on purpose: you are meant to
read it, reason about it, and design an update rule that works under it. It is
NOT modifiable. The verifier recomputes this schedule from its own private copy
and compares the result against the learning rates the runner actually applied,
so an edited copy in the agent workspace changes nothing except the score, which
becomes zero.

The schedule is a pure function of (step, total_steps) and a per-group
multiplier table. The runner is the only caller. Your submitted update rule
never sees total_steps and never sees this module.
"""

from __future__ import annotations

BASE_LR = 0.02

GROUP_MULTIPLIER = {
    "embed": 3.0,
    "hidden_matrix": 1.0,
    "head": 0.10,
    "vector": 0.5,
}

WARMUP_FRAC = 0.004
WARMUP_FLOOR = 0.05
STABLE_END_FRAC = 0.40
RESTART_DROP = 0.55
RESTART_END_FRAC = 0.70
RESTART_PEAK = 0.95
TAIL_FLOOR = 0.25


def schedule_multiplier(step: int, total_steps: int) -> float:
    """Return the global learning-rate multiplier applied at optimizer step `step`.

    Steps are one-indexed: the first optimizer step of a run is step 1 and the
    last is step total_steps. Four phases, in order:

    1. Linear warmup from WARMUP_FLOOR to 1.0 across the first WARMUP_FRAC of
       the run. The warmup is deliberately short.
    2. Stable at 1.0 until STABLE_END_FRAC.
    3. An instantaneous drop to RESTART_DROP at STABLE_END_FRAC, then a linear
       rise back to RESTART_PEAK by RESTART_END_FRAC.
    4. A linear tail from RESTART_PEAK down to TAIL_FLOOR at the final step. The
       tail floors at TAIL_FLOOR and never reaches zero, so this schedule has no
       terminal anneal.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 1 or step > total_steps:
        raise ValueError("step out of range for total_steps")
    t = (step - 1) / float(total_steps)
    if t < WARMUP_FRAC:
        u = t / WARMUP_FRAC
        return WARMUP_FLOOR + (1.0 - WARMUP_FLOOR) * u
    if t < STABLE_END_FRAC:
        return 1.0
    if t < RESTART_END_FRAC:
        u = (t - STABLE_END_FRAC) / (RESTART_END_FRAC - STABLE_END_FRAC)
        return RESTART_DROP + (RESTART_PEAK - RESTART_DROP) * u
    u = (t - RESTART_END_FRAC) / (1.0 - RESTART_END_FRAC)
    return RESTART_PEAK + (TAIL_FLOOR - RESTART_PEAK) * u


def group_lr(group_name: str, step: int, total_steps: int) -> float:
    """Return the exact learning rate the runner writes into a parameter group."""
    if group_name not in GROUP_MULTIPLIER:
        raise KeyError("unknown parameter group: %s" % group_name)
    return BASE_LR * GROUP_MULTIPLIER[group_name] * schedule_multiplier(step, total_steps)


def lr_trace(group_name: str, total_steps: int):
    """Return the full learning-rate trace for one group, for offline inspection."""
    return [group_lr(group_name, s, total_steps) for s in range(1, total_steps + 1)]
