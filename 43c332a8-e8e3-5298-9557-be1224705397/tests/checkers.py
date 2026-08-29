"""The nine graded checkers. Pure, deterministic, and blind to the submission.

Every function here is a total function of one mapping: the verifier telemetry
record that `tests/runner.py` produced in the verifier's own process. Nothing in
this module opens a socket, reads a clock, consults a random source, reads an
environment variable, reads a planted file, or imports the submission. The import
list is the AST allowlist, in full: json, math, hashlib, pathlib, dataclasses,
typing. `pathlib` and `json` are used to load a telemetry record from disk in the
one entry point that takes a path; every check itself takes the already-loaded
mapping and touches no filesystem.

The rule those constraints encode: if a checker needs a number, that number comes
from a record the verifier's own process produced. A number the submission printed
is reconciliation input, never reward input.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TELEMETRY_SCHEMA = "forge.verifier_telemetry/v1"

# The three outcome states. AR7 lives here: the middle state is a first-class
# value, not a shading of either neighbour.
STATE_ESTABLISHED = "significance-established"
STATE_UNPROVEN = "significance-unestablished-at-ceiling"
STATE_NOT_APPLICABLE = "significance-not-applicable"


@dataclass
class Verdict:
    """One checker's answer: a value in [0,1], and the reason it is not 1.0."""

    ident: str
    value: float
    reason: str = ""
    detail: str = ""
    state: str = ""
    numbers: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.value >= 1.0


def clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return max(0.0, min(1.0, number))


def load_telemetry(path: Path) -> Dict[str, Any]:
    """Read the verifier's own telemetry record. The only filesystem read here."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _bound(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    block = telemetry.get("bound")
    return block if isinstance(block, dict) else {}


def _seeds(telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = telemetry.get("seeds")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sustained_crossing(seed: Dict[str, Any], target: float, sustain: int) -> Optional[int]:
    """The first grid point below target that stays below it for `sustain` more points.

    Returns None when no such point exists. A single dip below target is not a
    crossing and is never reported as one; that distinction is the whole reason the
    verifier schedules the extra points itself.
    """
    points = seed.get("evaluation_points") or []
    losses = seed.get("raw_val_loss") or []
    if len(points) != len(losses) or not points:
        return None
    for index in range(len(points) - sustain):
        window = losses[index: index + sustain + 1]
        if all(float(value) < target for value in window):
            return int(points[index])
    return None


def dipped_below(seed: Dict[str, Any], target: float) -> bool:
    """True when any grid point is below target, sustained or not."""
    return any(float(value) < target for value in (seed.get("raw_val_loss") or []))


def first_dip_index(seed: Dict[str, Any], target: float) -> Optional[int]:
    """The index of the first grid point below target, sustained or not."""
    for index, value in enumerate(seed.get("raw_val_loss") or []):
        if float(value) < target:
            return index
    return None


def per_seed_scores(telemetry: Dict[str, Any]) -> List[float]:
    """The clipped anchor-normalised score of every re-executed seed."""
    bound = _bound(telemetry)
    target = float(bound.get("target_loss", 3.28))
    sustain = int(bound.get("sustain_points_required", 2))
    baseline = float(bound.get("baseline_metric", 3250))
    goal = float(bound.get("target_metric", 2690))
    span = baseline - goal
    scores: List[float] = []
    for seed in _seeds(telemetry):
        crossing = sustained_crossing(seed, target, sustain)
        if crossing is None or span <= 0:
            scores.append(0.0)
            continue
        scores.append(clamp((baseline - crossing) / span))
    return scores


def graded_step(telemetry: Dict[str, Any]) -> Optional[int]:
    """The graded quantity: the worst sustained crossing across re-executed seeds.

    A run is only as fast as its slowest seed, so this is a max and never a mean or
    a best-of. If any seed never sustained a crossing, there is no graded step.
    """
    bound = _bound(telemetry)
    target = float(bound.get("target_loss", 3.28))
    sustain = int(bound.get("sustain_points_required", 2))
    crossings = [sustained_crossing(seed, target, sustain) for seed in _seeds(telemetry)]
    if not crossings or any(item is None for item in crossings):
        return None
    return max(int(item) for item in crossings if item is not None)


def separation(telemetry: Dict[str, Any]) -> Optional[float]:
    """`sep = mean(s_i) - spread / (2 * sqrt(n))`, or None when n is zero.

    The spread term is what makes an improvement inside the seed-to-seed noise band
    fail to separate. It is subtracted rather than tested against, so the quantity
    is continuous in every input and the reward derived from it cannot step.
    """
    scores = per_seed_scores(telemetry)
    count = len(scores)
    if count == 0:
        return None
    mean = sum(scores) / count
    spread = max(scores) - min(scores)
    return mean - spread / (2.0 * math.sqrt(count))


def base_score(telemetry: Dict[str, Any]) -> float:
    """The anchor-normalised improvement at the graded step, clipped to [0,1]."""
    bound = _bound(telemetry)
    baseline = float(bound.get("baseline_metric", 3250))
    goal = float(bound.get("target_metric", 2690))
    step = graded_step(telemetry)
    if step is None or baseline - goal <= 0:
        return 0.0
    return clamp((baseline - step) / (baseline - goal))


# --------------------------------------------------------------------------- #
# The nine checkers. Exactly one reduction each, declared in tests/checkers.yaml.
# --------------------------------------------------------------------------- #


def check_target_reached_on_reexecuted_seeds(telemetry: Dict[str, Any]) -> Verdict:
    """EFFECT. Re-execution drove the verifier's own held-out loss below target."""
    ident = "target_reached_on_reexecuted_seeds"
    seeds = _seeds(telemetry)
    target = float(_bound(telemetry).get("target_loss", 3.28))
    if not seeds:
        return Verdict(ident, 0.0, "screen-target-not-reached",
                       "no seed was re-executed, so nothing was measured",
                       STATE_NOT_APPLICABLE)
    missed = [str(seed.get("seed")) for seed in seeds if not dipped_below(seed, target)]
    if missed:
        return Verdict(ident, 0.0, "screen-target-not-reached",
                       "seeds never below " + repr(target) + ": " + ", ".join(missed),
                       STATE_NOT_APPLICABLE,
                       {"seeds_missing_target": len(missed), "seeds": len(seeds)})
    return Verdict(ident, 1.0, "", str(len(seeds)) + " re-executed seeds reached below target",
                   "", {"seeds": len(seeds)})


def check_graded_readout_is_unsmoothed(telemetry: Dict[str, Any]) -> Verdict:
    """VALUE. The graded readout is the verifier's raw evaluation, unfiltered.

    A submission may EMA-blend its own reported loss for its own use. The quantity
    graded here is recomputed unsmoothed by the verifier, so a shaped readout buys
    nothing and is reported rather than silently ignored.
    """
    ident = "graded_readout_is_unsmoothed"
    bound = _bound(telemetry)
    if bool(bound.get("readout_smoothing_admitted", False)):
        return Verdict(ident, 0.0, "graded-readout-smoothed",
                       "the bound parameters admit smoothing on the graded path",
                       STATE_NOT_APPLICABLE)
    offenders = []
    for seed in _seeds(telemetry):
        if str(seed.get("readout_source", "")) != "verifier-raw":
            offenders.append(str(seed.get("seed")) + ":source=" + str(seed.get("readout_source")))
        elif str(seed.get("graded_readout_key", "")) != "raw_val_loss":
            offenders.append(str(seed.get("seed")) + ":key=" + str(seed.get("graded_readout_key")))
    if offenders:
        return Verdict(ident, 0.0, "graded-readout-smoothed",
                       "graded readout is not the raw verifier evaluation: " + ", ".join(offenders),
                       STATE_NOT_APPLICABLE)
    return Verdict(ident, 1.0, "", "every graded readout is the raw verifier evaluation")


def check_early_stop_is_not_a_crossing(telemetry: Dict[str, Any]) -> Verdict:
    """ABSENCE. No graded crossing is credited from a run that halted inside the window.

    Halting at a favourable evaluation is graded as not having crossed. It is scored
    zero with a reason rather than recorded as an absent result, because an absent
    result is indistinguishable from a harness that never ran.

    This checker owns exactly the case where the sustain window has NO DATA: the run
    exited nonzero, or the grid stops before the window could close. The neighbouring
    case, where the window has data and the loss went back above target, belongs to
    `crossing_is_sustained`. Splitting them is the point: a run that stopped and a run
    that regressed are two different findings and they carry two different reasons.
    """
    ident = "early_stop_is_not_a_crossing"
    bound = _bound(telemetry)
    target = float(bound.get("target_loss", 3.28))
    sustain = int(bound.get("sustain_points_required", 2))
    period = int(bound.get("evaluation_period_steps", 25))
    halted = []
    for seed in _seeds(telemetry):
        dip = first_dip_index(seed, target)
        if dip is None:
            continue
        points = [int(value) for value in (seed.get("evaluation_points") or [])]
        if not bool(seed.get("run_completed", False)):
            halted.append(str(seed.get("seed")) + ":run-incomplete")
            continue
        if dip + sustain >= len(points):
            halted.append(str(seed.get("seed")) + ":grid-stops-inside-window")
            continue
        if int(seed.get("final_step", -1)) < points[dip] + sustain * period:
            halted.append(str(seed.get("seed")) + ":final_step="
                          + str(seed.get("final_step")) + "<"
                          + str(points[dip] + sustain * period))
    if halted:
        return Verdict(ident, 0.0, "early-stop-not-a-crossing",
                       "runs halted before the sustain window closed: " + ", ".join(halted),
                       STATE_NOT_APPLICABLE)
    return Verdict(ident, 1.0, "", "no crossing rests on a run that halted inside its window")


def check_crossing_is_sustained(telemetry: Dict[str, Any]) -> Verdict:
    """INVARIANT. The target holds at the graded step and at the bound follow-on points."""
    ident = "crossing_is_sustained"
    bound = _bound(telemetry)
    target = float(bound.get("target_loss", 3.28))
    sustain = int(bound.get("sustain_points_required", 2))
    seeds = _seeds(telemetry)
    if not seeds:
        return Verdict(ident, 0.0, "crossing-not-sustained", "no seed to check",
                       STATE_NOT_APPLICABLE)
    unsustained = []
    for seed in seeds:
        if sustained_crossing(seed, target, sustain) is None and dipped_below(seed, target):
            unsustained.append(str(seed.get("seed")))
    if unsustained:
        return Verdict(ident, 0.0, "crossing-not-sustained",
                       "seeds dipped below target without holding for " + str(sustain)
                       + " further scheduled points: " + ", ".join(unsustained),
                       STATE_NOT_APPLICABLE, {"sustain_points_required": sustain})
    if graded_step(telemetry) is None:
        return Verdict(ident, 0.0, "crossing-not-sustained",
                       "no seed established a sustained crossing", STATE_NOT_APPLICABLE)
    return Verdict(ident, 1.0, "", "every seed held the target across the sustain window",
                   "", {"sustain_points_required": sustain})


def check_evaluation_grid_is_verifier_scheduled(telemetry: Dict[str, Any]) -> Verdict:
    """ORDERING. The grid ascends at the verifier's own fixed period, unselected."""
    ident = "evaluation_grid_is_verifier_scheduled"
    period = int(_bound(telemetry).get("evaluation_period_steps", 25))
    offenders = []
    for seed in _seeds(telemetry):
        points = [int(value) for value in (seed.get("evaluation_points") or [])]
        if str(seed.get("grid_source", "")) != "verifier":
            offenders.append(str(seed.get("seed")) + ":source=" + str(seed.get("grid_source")))
            continue
        if len(points) < 2:
            offenders.append(str(seed.get("seed")) + ":grid-too-short")
            continue
        gaps = [points[i + 1] - points[i] for i in range(len(points) - 1)]
        if any(gap != period for gap in gaps):
            offenders.append(str(seed.get("seed")) + ":gaps=" + str(sorted(set(gaps))))
    if offenders:
        return Verdict(ident, 0.0, "evaluation-grid-not-verifier-scheduled",
                       "grids that are not the verifier's ascending fixed-period grid: "
                       + ", ".join(offenders), STATE_NOT_APPLICABLE,
                       {"evaluation_period_steps": period})
    return Verdict(ident, 1.0, "", "every grid ascends at the bound verifier period",
                   "", {"evaluation_period_steps": period})


def check_graded_weights_are_harness_owned(telemetry: Dict[str, Any]) -> Verdict:
    """VALUE. The weights evaluated at the graded step are the ones the harness held."""
    ident = "graded_weights_are_harness_owned"
    bound = _bound(telemetry)
    if str(bound.get("checkpoint_ledger_owner", "")) != "harness":
        return Verdict(ident, 0.0, "graded-weights-not-harness-owned",
                       "the checkpoint ledger is owned by "
                       + repr(bound.get("checkpoint_ledger_owner")) + ", not by the harness",
                       STATE_NOT_APPLICABLE)
    target = float(bound.get("target_loss", 3.28))
    sustain = int(bound.get("sustain_points_required", 2))
    offenders = []
    for seed in _seeds(telemetry):
        crossing = sustained_crossing(seed, target, sustain)
        if crossing is None:
            continue
        ledger = {int(row.get("step")): row for row in (seed.get("checkpoint_ledger") or [])
                  if isinstance(row, dict)}
        evaluated = {int(row.get("step")): row for row in (seed.get("evaluated_weights") or [])
                     if isinstance(row, dict)}
        held = ledger.get(crossing)
        used = evaluated.get(crossing)
        if held is None or used is None:
            offenders.append(str(seed.get("seed")) + ":no-ledger-entry-at-" + str(crossing))
        elif str(held.get("owner")) != "harness":
            offenders.append(str(seed.get("seed")) + ":owner=" + str(held.get("owner")))
        elif str(held.get("weights_sha256")) != str(used.get("weights_sha256")):
            offenders.append(str(seed.get("seed")) + ":digest-mismatch-at-" + str(crossing))
    if offenders:
        return Verdict(ident, 0.0, "graded-weights-not-harness-owned",
                       "graded weights did not come from the harness state: " + ", ".join(offenders),
                       STATE_NOT_APPLICABLE)
    return Verdict(ident, 1.0, "", "graded weights are the harness state at the graded step")


def check_frozen_axes_survive_reexecution(telemetry: Dict[str, Any]) -> Verdict:
    """EFFECT. Re-execution ran under the axes the harness locked, not others."""
    ident = "frozen_axes_survive_reexecution"
    observed = telemetry.get("frozen_axes")
    locked = telemetry.get("frozen_axes_locked")
    if not isinstance(observed, dict) or not isinstance(locked, dict):
        return Verdict(ident, 0.0, "frozen-axis-moved",
                       "the run recorded no comparable frozen-axis pair", STATE_NOT_APPLICABLE)
    moved = sorted(key for key in set(locked) | set(observed)
                   if observed.get(key) != locked.get(key))
    if moved:
        return Verdict(ident, 0.0, "frozen-axis-moved",
                       "axes moved under re-execution: " + ", ".join(moved),
                       STATE_NOT_APPLICABLE, {"moved": moved})
    fingerprint = _digest(json.dumps(locked, sort_keys=True, separators=(",", ":")))
    return Verdict(ident, 1.0, "", "every frozen axis survived re-execution",
                   "", {"frozen_axis_fingerprint": fingerprint})


def check_claim_reconciles_with_reexecution(telemetry: Dict[str, Any]) -> Verdict:
    """DIVERGENCE. The claimed step does not beat what re-execution produced.

    The claim is reconciliation input. It can lower an outcome and can never raise
    one, so the only thing checked is that the claim did not overstate. The reward
    uses the verifier's number regardless of what the claim said.
    """
    ident = "claim_reconciles_with_reexecution"
    bound = _bound(telemetry)
    tolerance = int(bound.get("claim_tolerance_steps", 0))
    submission = telemetry.get("submission")
    submission = submission if isinstance(submission, dict) else {}
    claimed = submission.get("claimed_step")
    if claimed is None:
        return Verdict(ident, 0.0, "claim-exceeds-reexecution",
                       "the submission carried no claimed_step to reconcile",
                       STATE_NOT_APPLICABLE)
    step = graded_step(telemetry)
    if step is None:
        return Verdict(ident, 0.0, "claim-exceeds-reexecution",
                       "re-execution produced no graded step to reconcile against",
                       STATE_NOT_APPLICABLE)
    if step > int(claimed) + tolerance:
        return Verdict(ident, 0.0, "claim-exceeds-reexecution",
                       "claimed_step " + str(claimed) + " is better than the re-executed graded step "
                       + str(step) + " beyond the bound tolerance " + str(tolerance),
                       STATE_NOT_APPLICABLE,
                       {"claimed_step": int(claimed), "graded_step": int(step)})
    return Verdict(ident, 1.0, "", "the claim is not better than re-execution",
                   "", {"claimed_step": int(claimed), "graded_step": int(step)})


def check_significance_separation_cleared(telemetry: Dict[str, Any]) -> Verdict:
    """DIVERGENCE. The multi-seed separation clears the bound margin at the ceiling.

    This is the AR7 seam. Three outcomes leave this function and they are never
    collapsed into two:

      established   `sep >= margin`. value 1.0, no reason.
      unproven      the run completed and reached the target, the seed budget was
                    the binding limit, and `sep < margin`. value ramps continuously
                    from 1.0 down to 0.0 as `sep` falls to zero, and the reason is
                    `significance-unestablished-at-ceiling`. This is NOT a failure.
      not applicable no crossing exists, so significance was never at issue. The
                    required gates own that case and name it with their own reasons.

    The value is `clip(sep / margin, 0, 1)`, which is continuous everywhere. It does
    not flip at the margin, because a binary flip there would let a submission at
    `sep = 0.0499` and one at `sep = 0.0` receive the same verdict, and those two
    submissions are not in the same condition.
    """
    ident = "significance_separation_cleared"
    bound = _bound(telemetry)
    margin = float(bound.get("separation_margin", 0.05))
    floor = int(bound.get("graded_seed_floor", 20))
    ceiling = int(bound.get("seed_ceiling", 20))
    count = len(_seeds(telemetry))
    step = graded_step(telemetry)

    if step is None:
        return Verdict(ident, 0.0, "", "no sustained crossing, so significance is not at issue",
                       STATE_NOT_APPLICABLE, {"seeds": count})

    if count < floor:
        return Verdict(
            ident, 0.0, "significance-unestablished-at-ceiling",
            "the seed budget was the binding limit: " + str(count)
            + " seeds re-executed against a graded floor of " + str(floor)
            + " and a ceiling of " + str(ceiling)
            + ". The run completed and reached the target; separation is undetermined.",
            STATE_UNPROVEN,
            {"seeds": count, "graded_seed_floor": floor, "seed_ceiling": ceiling,
             "separation": None, "separation_margin": margin, "binding_limit": "seed-budget"},
        )

    sep = separation(telemetry)
    if sep is None:
        return Verdict(ident, 0.0, "significance-unestablished-at-ceiling",
                       "no per-seed score was measurable", STATE_UNPROVEN,
                       {"seeds": count, "separation": None, "separation_margin": margin})

    value = clamp(sep / margin) if margin > 0 else 0.0
    numbers = {
        "seeds": count,
        "graded_seed_floor": floor,
        "seed_ceiling": ceiling,
        "separation": sep,
        "separation_margin": margin,
        "significance_factor": value,
    }
    if sep >= margin:
        return Verdict(ident, 1.0, "",
                       "separation " + format(sep, ".6f") + " clears the bound margin "
                       + format(margin, ".6f"), STATE_ESTABLISHED, dict(numbers, significance_factor=1.0))
    numbers["binding_limit"] = "seed-budget" if count >= ceiling else "graded-seed-floor"
    return Verdict(
        ident, value, "significance-unestablished-at-ceiling",
        "the run completed and reached the target at graded step " + str(step)
        + ", the seed budget was the binding limit at " + str(count)
        + " of a ceiling of " + str(ceiling) + ", and separation " + format(sep, ".6f")
        + " did not clear the bound margin " + format(margin, ".6f")
        + ". Unproven, not failed.",
        STATE_UNPROVEN, numbers,
    )


# The bound gate order, by checker id. Read as a tuple so no caller reorders it in
# place. `significance_separation_cleared` is deliberately absent: it is not a gate,
# it is the continuous term, and putting it here would turn the unproven band into a
# binary rejection, which is exactly the collapse this slot exists to prevent.
GATES: Tuple[str, ...] = (
    "target_reached_on_reexecuted_seeds",
    "graded_readout_is_unsmoothed",
    "early_stop_is_not_a_crossing",
    "crossing_is_sustained",
    "evaluation_grid_is_verifier_scheduled",
    "graded_weights_are_harness_owned",
    "frozen_axes_survive_reexecution",
    "claim_reconciles_with_reexecution",
)

# checker id -> the selector this bundle declares in tests/checkers.yaml.
SELECTOR_NAMES: Dict[str, str] = {
    "target_reached_on_reexecuted_seeds": "check_target_reached_on_reexecuted_seeds",
    "graded_readout_is_unsmoothed": "check_graded_readout_is_unsmoothed",
    "crossing_is_sustained": "check_crossing_is_sustained",
    "early_stop_is_not_a_crossing": "check_early_stop_is_not_a_crossing",
    "evaluation_grid_is_verifier_scheduled": "check_evaluation_grid_is_verifier_scheduled",
    "graded_weights_are_harness_owned": "check_graded_weights_are_harness_owned",
    "frozen_axes_survive_reexecution": "check_frozen_axes_survive_reexecution",
    "claim_reconciles_with_reexecution": "check_claim_reconciles_with_reexecution",
    "significance_separation_cleared": "check_significance_separation_cleared",
}
