"""The frozen surrogate optimizer surface. Verifier-owned, held out from the agent.

This module is the measurement device. It is the only place the graded quantity
is computed, and it is never handed to the solving agent: the agent surface
Harbor assembles for a non-oracle agent carries task.toml, instruction.md and
environment/ only, so these constants live on the far side of that boundary.

What it models. A session tunes three optimizer directions, a, b and c, whose
combined effect sets how many optimizer steps a run needs before its validation
loss falls below the bound target and stays there. Direction a saturates inside
the session budget: past its plateau, further movement along it changes nothing
at all, and nothing in the observation stream announces that it has saturated.
Directions b and c do not saturate within reach. One attempt moves exactly one
direction by at most the bound per-attempt cap, so the target operating point is
reachable only by a session that spends its whole attempt budget and stops
feeding the saturated direction once it stops paying.

Why every quantity is an integer until the last division. Repeated addition of
0.05 in binary floating point does not land on 0.5, and two sessions holding the
same amount of progress split differently across b and c would otherwise differ
in the last bits and be graded as different operating points. Coordinates are
integer thousandths and gains are integer milli-steps, so the surface is exactly
split-invariant: what a session holds is what it is graded on.

Determinism. Every number below is a pure function of the frozen constants, the
session state and the seed. There is no clock, no random source, no network and
no locale-dependent formatting anywhere in this file.
"""

import hashlib

SURFACE_ID = "oer06-tri-axis-plateau/v1"

# --- surrogate surface constants -------------------------------------------
# These are the geometry of a synthetic surface, NOT this family's published
# anchors. F15 anchors are unmeasured and are declared absent in task.toml under
# gap-oer-per-family-anchors-unmeasured. No number in this block may be read as
# a baseline_metric or a target_metric for the family.
BASE_STEPS = 4096
BASE_MILLI = BASE_STEPS * 1000
AXIS_ORDER = ("a", "b", "c")

# Milli-steps bought per thousandth of movement, and where each axis stops
# paying. Axis a is worth 500 * 768 = 384000 milli-steps and then nothing; b and
# c are worth 150 milli-steps per unit and their plateaus sit at 3000 units,
# which fifty attempts of fifty units cannot reach.
AXIS_MILLI_PER_UNIT = {"a": 768, "b": 150, "c": 150}
AXIS_PLATEAU_UNITS = {"a": 500, "b": 3000, "c": 3000}
FLATTENING_AXIS = "a"

UNIT_SCALE = 1000
DELTA_CAP_UNITS = 50

# Bound session shape, transcribed from the batch optimization bindings.
# max_timeout bounds the session ACROSS attempts; budget_hours bounds ONE
# attempt. They are never aliases.
ATTEMPT_BUDGET = 50
MAX_TIMEOUT_HOURS = 6.0
BUDGET_HOURS = 0.12
FINAL_SELECTION = "best"

SEED_LADDER = (11, 29, 47, 83, 101)
JITTER_SPAN = 12
SEPARATION_MARGIN = 0.05
SEED_ARTIFACT_BAND = 0.25

# How many attempts a session may still spend on the flattened axis after that
# axis reaches its plateau. One confirmation probe is how a session learns the
# direction went flat, so one is owed; a session still feeding it well past that
# never noticed.
PROBE_ALLOWANCE = 2

TARGET_LOSS = 3.28
EVAL_EVERY = 8
SUSTAIN_POINTS = 3
CURVE_MAX_STEPS = 5120
DIP_OFFSET_POINTS = 3
DIP_DEPTH = 0.004

ZERO_STATE = {"a": 0, "b": 0, "c": 0}

# The operating point a session reaches by spending ten attempts taking axis a
# to its plateau, one attempt confirming that it went flat, and the remaining
# thirty-nine on a direction that is still paying. It is split-invariant, so any
# division of those thirty-nine attempts between b and c reaches it.
REFERENCE_STATE = {"a": 500, "b": 1950, "c": 0}

# The coordinates the reference session is actually carrying when its fiftieth
# attempt closes: ten attempts taking a to its plateau, one confirming probe past
# it, and the remaining thirty-nine spread over directions that still pay. It is
# not the same triple as REFERENCE_STATE above, which is the canonical
# split-invariant spelling of the same operating point, and that is exactly what
# reference_operating_point_matches_final_state grades.
#
# This is NOT a second derivation source. solution/recompute.py asserts it equals
# reference_policy.expected_final_state in solution/grounding.yaml on every run,
# alongside every other constant in this file, so a drift between the two is a
# hard failure of the generator rather than a silent disagreement.
REFERENCE_FINAL_STATE = {"a": 550, "b": 250, "c": 1700}


def zero_state():
    return dict(ZERO_STATE)


def clamp_delta(units):
    try:
        value = int(units)
    except (TypeError, ValueError):
        return 0
    if value < 0:
        return 0
    return min(value, DELTA_CAP_UNITS)


def apply_delta(state, axis, units):
    """Advance one axis and return the applied movement in integer thousandths.

    Movement past a plateau is applied to the coordinate but buys nothing, which
    is exactly the silence this task is built around: the state still moves, and
    the measured step count does not.
    """
    if axis not in AXIS_MILLI_PER_UNIT:
        return 0
    applied = clamp_delta(units)
    state[axis] = state.get(axis, 0) + applied
    return applied


def effective_units(axis, units):
    return min(int(units), AXIS_PLATEAU_UNITS[axis])


def gain_milli(state):
    total = 0
    for axis in AXIS_ORDER:
        total += effective_units(axis, state.get(axis, 0)) * AXIS_MILLI_PER_UNIT[axis]
    return total


def steps_milli(state):
    return BASE_MILLI - gain_milli(state)


def steps_at(state):
    return steps_milli(state) / 1000.0


def marginal_gain(before, after):
    """What one attempt actually bought, measured here and reported to nobody else."""
    return (steps_milli(before) - steps_milli(after)) / 1000.0


def _digest(*parts):
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def jitter(state, seed):
    """Per-seed dispersion in steps, keyed on what the session holds and the seed.

    Keying on held progress rather than on the raw coordinates is what makes the
    surface split-invariant: two sessions holding the same progress are the same
    operating point and must not be separated by a hash of how they got there.
    """
    raw = _digest(SURFACE_ID, "jitter", seed, steps_milli(state))
    return int(raw[:4], 16) % (2 * JITTER_SPAN + 1) - JITTER_SPAN


def seed_steps(state, seed):
    return steps_at(state) + jitter(state, seed)


def eval_points(halted_at=None):
    """The evaluation cadence the VERIFIER schedules. A submission never sets it."""
    limit = CURVE_MAX_STEPS
    if halted_at is not None:
        try:
            limit = min(limit, int(halted_at))
        except (TypeError, ValueError):
            limit = CURVE_MAX_STEPS
    return list(range(EVAL_EVERY, limit + 1, EVAL_EVERY))


def _first_point_above(value):
    return (int(value) // EVAL_EVERY + 1) * EVAL_EVERY


def dip_point(state, seed):
    """The single favourable evaluation a noise dip puts before the real crossing."""
    return _first_point_above(seed_steps(state, seed)) - DIP_OFFSET_POINTS * EVAL_EVERY


def raw_loss(state, seed, step):
    """The unsmoothed evaluation this module computes at one step. No filter, ever."""
    reference = seed_steps(state, seed)
    if reference <= 0:
        return TARGET_LOSS
    if step == dip_point(state, seed):
        return TARGET_LOSS - DIP_DEPTH
    if step <= reference:
        return TARGET_LOSS + 0.9 * (reference - step) / reference
    return TARGET_LOSS - 0.35 * (step - reference) / reference


def sustained_crossing(state, seed, halted_at=None):
    """The graded crossing: below target here and at SUSTAIN_POINTS further points.

    A single favourable evaluation is not a crossing. The dip this surface plants
    three points early is below target at exactly one point, so a run that stops
    there, or blends its readout until the dip survives averaging, produces no
    crossing at all rather than an early one.
    """
    points = eval_points(halted_at)
    available = set(points)
    for point in points:
        if raw_loss(state, seed, point) >= TARGET_LOSS:
            continue
        held = True
        for step in range(1, SUSTAIN_POINTS + 1):
            follow = point + step * EVAL_EVERY
            if follow not in available or raw_loss(state, seed, follow) >= TARGET_LOSS:
                held = False
                break
        if held:
            return point
    return None


def weights_digest(state, seed, step):
    if step is None:
        return ""
    return _digest(SURFACE_ID, "weights", seed, steps_milli(state), step)[:32]


def mean_steps(state, halted_at=None):
    per_seed = {}
    for seed in SEED_LADDER:
        per_seed[str(seed)] = sustained_crossing(state, seed, halted_at)
    values = [value for value in per_seed.values() if value is not None]
    if len(values) != len(SEED_LADDER):
        return None, per_seed
    return sum(values) / float(len(values)), per_seed


def control_detail():
    """Both in-run operating-point controls, with their per-seed values.

    These are measurements of the frozen surrogate surface taken by the verifier
    at grading time, not published anchors. The family's baseline_metric and
    target_metric remain absent under gap-oer-per-family-anchors-unmeasured, and
    nothing here may be transcribed into them.
    """
    baseline, baseline_seeds = mean_steps(dict(ZERO_STATE))
    target, target_seeds = mean_steps(dict(REFERENCE_STATE))
    return baseline, baseline_seeds, target, target_seeds


def control_means():
    baseline, _, target, _ = control_detail()
    return baseline, target


def reference_final_state_detail():
    """The operating point REFERENCE_FINAL_STATE produces, measured on this surface.

    Measured here, at grading time, in exactly the way every other control is
    measured, so a checker can compare it against the reference-operating-point
    control the reward formula divides by instead of taking split invariance on
    trust. It is a measurement of the frozen surrogate surface and never an anchor.
    """
    mean, per_seed = mean_steps(dict(REFERENCE_FINAL_STATE))
    return mean, per_seed, gain_milli(REFERENCE_FINAL_STATE)


def raw_reward(baseline, target, agent):
    """The bound lower-is-better formula, clipped to the closed interval."""
    if baseline is None or target is None or agent is None:
        return 0.0
    span = baseline - target
    if span <= 0:
        return 0.0
    return min(max((baseline - agent) / span, 0.0), 1.0)


def margin_taper(score):
    """Continuous degradation below the separation margin. No cliff at the margin.

    At the margin the taper is exactly one, so the two branches agree there and
    the reward is continuous. Below the margin it falls off smoothly rather than
    snapping to zero, which is what keeps the score from being binary at the one
    place a binary score would be easiest to hide.
    """
    if score >= SEPARATION_MARGIN:
        return score
    return score * (score / SEPARATION_MARGIN)
