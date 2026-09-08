"""The two endpoints of the reward scale, both MEASURED here, neither authored.

`checkers.normalized_reward` scales the graded crossing between a baseline and a
target:

    raw = (baseline_metric - graded_step) / (baseline_metric - target_metric)

Until this module existed those two numbers were the literals 3250 and 2690,
carried in tests/golden.json under `"state": "measured"` and an `authority`
pointing at a requirements document. Neither was a measurement of anything this
bundle runs:

  * 3250 is the SCHEDULE LENGTH of `environment/train_locked.py`. A schedule
    length is not a crossing step. The handed baseline was never run and its
    crossing was never read, so the low end of the scale was the number of steps
    somebody intended to take rather than the step at which the loss went under
    the bar.
  * 2690 is an upstream nanogpt speedrun record, measured on a 124M model over
    FineWeb10B at a batch of 480x1024 -- a different substrate from the one this
    verifier now trains.

Both are re-derived on every graded run, on THIS substrate, at THIS operating
point, by running two verifier-owned probe scripts through the SAME chain the
submission goes through: `runner.launch` into a verifier-owned run directory,
the pinned harness snapshotting on its own cadence, `tests/evaluator.py`
computing an unsmoothed held-out cross-entropy per snapshot, and
`checkers.first_sustained_below` reading the crossing off those evaluations. One
chain measures the bar and the submission, so they are commensurable by
construction rather than by assertion.

  tests/probe_handed.py    the free axes environment/train_locked.py hands the
                           agent, given the full schedule the envelope allows.
                           Its crossing is `baseline_metric`.
  tests/probe_refined.py   one fixed declared refinement of them. Its crossing is
                           `target_metric`, where the reward saturates.

Neither probe reads `solution/`, which tests/Dockerfile does not copy into this
image, so the reference has to reach the bar on its own.

WHEN THE SCALE CANNOT BE BUILT. If either probe fails to cross, or the two
crossings leave no positive span, this module returns null endpoints and a
machine-readable reason. `grade.py` turns that into an attributed zero. It does
not substitute a number, because substituting a number is the defect this module
was written to remove.
"""

from __future__ import annotations

import pathlib
import sys
import time
from typing import Any, Dict, Optional

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkers  # noqa: E402
import runner  # noqa: E402

HANDED_PROBE = HERE / "probe_handed.py"
REFINED_PROBE = HERE / "probe_refined.py"

HANDED_SOURCE = "verifier-rerun-of-the-handed-baseline"
REFINED_SOURCE = "verifier-refined-probe"

_CACHE: Dict[str, Any] = {}


def _measure_one(probe: pathlib.Path, run_root: pathlib.Path, timeout: float,
                 state: Dict[str, Any], evaluate) -> Dict[str, Any]:
    """Run one probe exactly as a submission is run and read its crossing back."""
    started = time.monotonic()
    record = runner.launch(probe, run_root, timeout,
                           schedule_hint=int(state.get("max_schedule_steps", 0) or 0))
    rows = evaluate(record, state)
    crossing: Optional[int] = checkers.first_sustained_below(
        checkers.evaluations({"evaluations": rows}),
        float(state["target_loss"]), int(state["sustain_points"]))
    return {
        "probe": probe.name,
        "crossing": None if crossing is None else int(crossing),
        "status": record.status,
        "exit_code": record.exit_code,
        "steps_executed": record.steps_executed,
        "evaluations": len(rows),
        "losses": [round(float(row["loss"]), 5) for row in rows],
        "stderr_tail": record.stderr_tail[-300:],
        "seconds": round(time.monotonic() - started, 3),
    }


def measure(run_root: pathlib.Path, timeout: float, state: Dict[str, Any],
            evaluate) -> Dict[str, Any]:
    """Both endpoints, measured now, on this substrate, in this image.

    `evaluate` is `grade.evaluate_ledger`, passed in rather than imported, so
    this module cannot drift onto a second evaluation path. There is exactly one
    way a loss becomes a crossing in this bundle and every consumer uses it.
    """
    if _CACHE:
        return _CACHE

    handed = _measure_one(HANDED_PROBE, run_root / "anchor-handed", timeout, state, evaluate)
    refined = _measure_one(REFINED_PROBE, run_root / "anchor-refined", timeout, state, evaluate)
    baseline = handed["crossing"]
    target = refined["crossing"]

    if baseline is None:
        anchors_state, gap = "unscalable", "handed-probe-never-crossed"
    elif target is None:
        anchors_state, gap = "unscalable", "refined-probe-never-crossed"
    elif baseline - target <= 0:
        anchors_state, gap = "unscalable", "scaling-span-nonpositive"
    else:
        anchors_state, gap = "measured", ""

    _CACHE.update({
        "baseline_metric": baseline,
        "target_metric": target,
        "direction": "lower",
        "pass_threshold": 0.65,
        "state": anchors_state,
        "gap": gap,
        "baseline_source": HANDED_SOURCE,
        "target_source": REFINED_SOURCE,
        "authority": "measured at grading time by tests/anchors.py on this substrate",
        "handed_probe": handed,
        "refined_probe": refined,
        "seconds": round(handed["seconds"] + refined["seconds"], 3),
    })
    return _CACHE
