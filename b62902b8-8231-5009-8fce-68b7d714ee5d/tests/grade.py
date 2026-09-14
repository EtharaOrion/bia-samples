"""The grading path for slot OER-03. Imports the checkers, never the submission.

Flow, and every step of it happens inside the verifier's own process:

  1. tests/runner.py launches the submission out-of-process and returns a run
     record. Nothing is imported from it.
  2. This module reads the harness-owned weight ledger out of the verifier-owned
     run directory and evaluates each recorded state on the frozen held-out split
     the agent environment never sees. Those evaluations are the ONLY loss
     numbers that reach grading.
  3. tests/checkers.py derives the outcome document and runs the gate chain.
  4. The reward is written. Every exit path writes one, and every zero carries a
     machine-readable reason.

The graded step is the crossing point. A schedule extended past the crossing
adds evaluation points that change nothing, and a run that never sustains a
crossing is graded a failure with a reason rather than reported as an absent
result. That obligation applies to this file too: if this module raises, the
module-level handler still writes an attributed reward, because a grader that
dies quietly is the same defect one layer up.
"""

from __future__ import annotations

import json
import pathlib
import sys
import traceback
from typing import Any, Dict, List, Optional

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import anchors as anchors_module  # noqa: E402  the grade-time anchor measurement
import checkers  # noqa: E402  the checker module, imported by name
import runner  # noqa: E402  the out-of-process launcher

# The bound runtime paths. seed/forge/verifier.py binds the bare-float carrier at
# /logs/verifier/reward.txt and audits this bundle against it. The lane brief
# binds the attributed document at /logs/verifier/reward.json. Both are written on
# every exit path, so neither contract is silently dropped. The divergence is
# declared in solution/grounding.yaml under gap-oer-03-reward-path-name-collides.
LOGS_DIR = pathlib.Path("/logs/verifier")
REWARD_VALUE_PATH = "/logs/verifier/reward.txt"
REWARD_DOCUMENT_PATH = "/logs/verifier/reward.json"
SCORE_DOCUMENT_PATH = "/logs/verifier/score.json"

# The submission, and the verifier-owned scratch root.
#
# THIS USED TO BE `/app/submission.py` AND NOTHING EVER WROTE IT. The agent
# container and the verifier container are different containers; they share
# /workspace and /logs and nothing else. `/app` is inside each image, so the file
# the solving agent wrote at /app/submission.py did not exist in the container
# that grades it, `runner.launch` returned `submission-absent` on every run, and
# the slot scored a constant zero for the reference and for a wreck alike. That
# is not a grading result, it is a grading path that never reached a submission.
#
# The bound path is now the shared surface, and the roots are ordered rather than
# single so a runtime that stages the submission at either place is read. The
# first entry is what task.toml, instruction.md and solution/solve.sh all name.
SUBMISSION_CANDIDATES = (
    pathlib.Path("/workspace/submission.py"),
    pathlib.Path("/workspace/app/submission.py"),
    pathlib.Path("/app/submission.py"),
)
RUN_ROOT = pathlib.Path("/tmp/bia-oer03-runs")


def submission_path() -> pathlib.Path:
    """The first candidate that exists, else the bound one so the reason names it."""
    for candidate in SUBMISSION_CANDIDATES:
        if candidate.is_file():
            return candidate
    return SUBMISSION_CANDIDATES[0]

# The generated fixture carrying the bound state and the reference binding. It is
# derived from solution/grounding.yaml by solution/recompute.py and is verifier
# side only; the agent surface Harbor assembles carries task.toml, instruction.md
# and environment/ and nothing else.
GOLDEN = HERE / "golden.json"
CHECKERS_YAML = HERE / "checkers.yaml"
RUBRICS_JSONL = HERE / "rubrics.jsonl"

# Reasons this module can emit on its own behalf, distinct from the checker
# reasons in tests/checkers.py ZERO_REASONS.
GRADER_REASONS = (
    "grader-internal-error",
    "bound-state-unreadable",
    "no-improvement-over-baseline",
)

# Every selector the gate chain reaches, named here because tests/checkers.yaml
# declares this file as the carrier that reaches each one. Reading this tuple is
# how an outside auditor confirms the manifest's reachability claim without
# trusting it.
GATE_SELECTORS = (
    "absence_of_ungraded_non_convergence",
    "run_produced_verifier_owned_state",
    "frozen_axes_held",
    "evaluated_weights_owned_by_the_harness",
    "evaluation_points_ascend_on_the_verifier_schedule",
    "graded_readout_unsmoothed",
    "early_stop_is_not_a_crossing",
    "crossing_sustained_across_verifier_evaluations",
    "graded_step_is_the_crossing_not_the_schedule",
    "reported_crossing_matches_recomputed_crossing",
)

# Per-RUN bound, and the verifier now makes three runs: the handed anchor probe,
# the refined anchor probe, and the submission. It was 10800 s, which is three
# hours for one run and nine for three, against an operating point nobody had
# timed. Re-scaled with the point: measured in this image, a full
# max_schedule_steps run of the gradeable point costs well under a minute, and
# this leaves a wide margin over that while still bounding a submission that
# never terminates.
VERIFIER_RUN_TIMEOUT_SEC = 240.0


def logs_dir() -> pathlib.Path:
    """The reward directory. Overridable by argv only, never by the submission."""
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return pathlib.Path(sys.argv[1].strip())
    return LOGS_DIR


def emit(reward: float, reason: str, metric: Dict[str, Any], verdicts: Optional[List] = None) -> float:
    """Write the reward. One float in [0, 1], and a reason whenever it is zero.

    The write is the last thing that happens on any path through this module, and
    tests/test.sh registers an EXIT trap that writes a reward if this module never
    reached here at all.
    """
    value = float(min(max(float(reward), 0.0), 1.0))
    if value <= 0.0 and not reason:
        reason = "grader-internal-error"
    document = {"reward": value, "reason": reason, "metric": dict(metric)}
    base = logs_dir()
    try:
        base.mkdir(parents=True, exist_ok=True)
        (base / "reward.txt").write_text(repr(value) + "\n", encoding="utf-8")
        (base / "reward.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (base / "score.json").write_text(
            json.dumps(
                {
                    "reward": value,
                    "reason": reason,
                    "metric": dict(metric),
                    "checkers": [item.as_dict() for item in (verdicts or [])],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        # An unwritable reward directory is an infrastructure fault. It is
        # reported on stderr rather than swallowed, and the exit status carries it.
        sys.stderr.write("reward directory unwritable: " + str(base) + "\n")
    sys.stdout.write(json.dumps(document, sort_keys=True) + "\n")
    return value


def bound_state() -> Dict[str, Any]:
    """The verifier-owned bound values, DERIVED from the executed operating point.

    These used to be a copy of tests/golden.json's `bound_state` block, which was
    a second authored statement of numbers the pinned harness also states. Two
    statements of one thing is how a re-scale moves one and not the other, and
    the grader would then have read a cadence the harness never snapshotted on.
    Everything below now comes off the same declaration the harness, the loader,
    the model and the evaluator resolve: environment/operating_point.json,
    through environment/bia_corpus.py, carried into this image at /app/env.

    The two anchors are deliberately ABSENT from this document. They are not
    bound state; they are measurements, and tests/anchors.py takes them.
    """
    sys.path.insert(0, "/app/env")
    try:
        import bia_corpus  # type: ignore
        import bia_harness  # type: ignore
    except ImportError:
        return {}
    axes = bia_corpus.bound_axes()
    return {
        "eval_stride": int(bia_harness.EVAL_STRIDE),
        "sustain_points": int(bia_harness.SUSTAIN_POINTS),
        "target_loss": float(bia_harness.TARGET_LOSS),
        "max_schedule_steps": int(bia_harness.MAX_SCHEDULE_STEPS),
        "readout_filter_admitted": str(bia_harness.READOUT_FILTER_ADMITTED),
        "weights_digest_domain": str(bia_harness.WEIGHTS_DIGEST_DOMAIN),
        "graded_quantity": "crossing_step",
        "operating_point": str(bia_corpus.operating_point()["name"]),
        "frozen_batch_size": int(axes["batch_size"]),
        "frozen_fwd_bwd_per_step": int(axes["fwd_bwd_per_step"]),
        "frozen_dataset_digest": str(axes["dataset_digest"]),
        "frozen_architecture_digest": str(axes["architecture_digest"]),
    }


def evaluate_ledger(run: runner.RunRecord, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The verifier's own unsmoothed evaluation of harness-owned states.

    The ledger names the steps at which the pinned harness owned a parameter set
    and the digest of that set. This function evaluates each one on the frozen
    held-out split and emits a raw loss. It reads no loss the submission printed,
    and it applies no filter: `filter` is always the admitted raw readout, so an
    EMA-blended number can never enter here.

    The evaluation grid is the verifier's. Ledger entries off the bound stride are
    dropped rather than admitted, because the cadence is not a submission choice.
    """
    ledger = runner.read_ledger(pathlib.Path(run.run_dir))
    stride = int(state.get("eval_stride", 0) or 0)
    domain = str(state.get("weights_digest_domain", ""))
    rows: List[Dict[str, Any]] = []
    entries = ledger.get("states")
    if not isinstance(entries, list) or stride <= 0:
        return rows
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        try:
            step = int(raw.get("step"))
        except (TypeError, ValueError):
            continue
        if step <= 0 or step % stride != 0:
            continue
        params_digest = str(raw.get("params_digest", ""))
        loss = _held_out_loss(pathlib.Path(run.run_dir), step, params_digest)
        if loss is None:
            continue
        rows.append(
            {
                "step": step,
                "loss": float(loss),
                "filter": str(state.get("readout_filter_admitted", "raw")),
                "weights_digest": checkers.harness_weights_digest(domain, step, params_digest),
                "weights_source": "harness",
                "params_digest": params_digest,
                "scheduled_by": "verifier",
            }
        )
    rows.sort(key=lambda item: item["step"])
    return rows


def _held_out_loss(run_dir: pathlib.Path, step: int, params_digest: str) -> Optional[float]:
    """Evaluate one harness-owned state on the frozen held-out split.

    The split lives in the verifier image only and never enters the agent
    environment. A state the harness recorded but did not persist yields None,
    which drops the point rather than inventing one.
    """
    state_path = run_dir / ("state_" + str(step) + ".pt")
    if not state_path.is_file():
        return None
    try:
        import evaluator  # type: ignore  # verifier-image module, GPU side
    except ImportError:
        return None
    return float(evaluator.held_out_loss(state_path, params_digest))


def build_telemetry(run: runner.RunRecord, state: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the record every checker reads. Live handles only."""
    return {
        "bound": dict(state),
        "run": {
            "status": run.status,
            "exit_code": run.exit_code,
            "stopped_by": run.stopped_by,
            "schedule_length": run.schedule_length,
            "steps_executed": run.steps_executed,
        },
        "evaluations": evaluate_ledger(run, state),
        "frozen_axes_observed": _observed_axes(pathlib.Path(run.run_dir)),
        "submission_report": _submission_report(pathlib.Path(run.run_dir)),
        "submission_sha256": run.submission_sha256,
    }


def _observed_axes(run_dir: pathlib.Path) -> Dict[str, Any]:
    """What the pinned harness recorded about the frozen axes during the run."""
    ledger = runner.read_ledger(run_dir)
    axes = ledger.get("frozen_axes")
    return dict(axes) if isinstance(axes, dict) else {}


def _submission_report(run_dir: pathlib.Path) -> Dict[str, Any]:
    """What the submission SAID. Carried only so a divergence can be graded."""
    ledger = runner.read_ledger(run_dir)
    report = ledger.get("submission_report")
    return dict(report) if isinstance(report, dict) else {}


def grade_telemetry(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """The gate chain and the reward, over one telemetry record. Pure.

    This is the whole graded decision, factored so that the feasibility bundle in
    seed/tasks/OER-03/adequacy.py drives the SAME chain the runtime drives rather
    than a restatement of it.
    """
    telemetry = dict(telemetry)
    if "outcome" not in telemetry:
        telemetry["outcome"] = checkers.derive_outcome(telemetry)
    verdicts = checkers.run_all(telemetry)
    bound_block = telemetry.get("bound", {}) or {}
    baseline = bound_block.get("baseline_metric")
    target = bound_block.get("target_metric")
    graded = telemetry["outcome"].get("graded_step")
    metric = {
        "graded_step": graded,
        "baseline": None if baseline is None else int(baseline),
        "target": None if target is None else int(target),
        "anchors_state": bound_block.get("anchors_state"),
        "baseline_source": bound_block.get("baseline_source"),
        "target_source": bound_block.get("target_source"),
        "anchor_seconds": bound_block.get("anchor_seconds"),
        "operating_point": bound_block.get("operating_point"),
        "schedule_length": telemetry.get("run", {}).get("schedule_length"),
        "graded_quantity": "crossing_step",
        "verdict_digest": checkers.verdict_digest(verdicts),
    }
    failure = checkers.first_failure(verdicts)
    if failure is not None:
        return {
            "reward": 0.0,
            "reason": failure.reason,
            "metric": metric,
            "verdicts": verdicts,
            "telemetry": telemetry,
        }
    # `fixture-calibrated` is admitted here and ONLY here. tests/test_output.py
    # drives this same function over the recorded telemetry documents in
    # tests/golden.json, which carry their own self-consistent scale. A live run
    # cannot reach that value: main() sets `anchors_state` from
    # tests/anchors.py, which returns `measured` or `unscalable` and nothing
    # else, and no submission can put a byte into this telemetry.
    if (bound_block.get("anchors_state") not in ("measured", "fixture-calibrated")
            or baseline is None or target is None):
        # The verifier could not build a scale on this substrate. A fact about
        # this run, not about the submission, so it carries the probe's own
        # machine-readable reason instead of being reported as the submission
        # failing to improve. Substituting an endpoint here is exactly the defect
        # tests/anchors.py was written to remove.
        return {
            "reward": 0.0,
            "reason": bound_block.get("anchors_gap") or "anchors-unmeasurable",
            "metric": metric,
            "verdicts": verdicts,
            "telemetry": telemetry,
        }
    score = checkers.normalized_reward(graded, int(baseline), int(target))
    reason = "" if score > 0.0 else "no-improvement-over-baseline"
    return {
        "reward": score,
        "reason": reason,
        "metric": metric,
        "verdicts": verdicts,
        "telemetry": telemetry,
    }


def main() -> int:
    state = bound_state()
    if not state:
        emit(
            0.0,
            "bound-state-unreadable",
            {"graded_step": None, "baseline": None, "target": None, "graded_quantity": "crossing_step"},
        )
        return 1
    state = dict(state)

    # BOTH ENDS OF THE SCALE, MEASURED, BEFORE ANYTHING IS GRADED. Two verifier-
    # owned probes are trained and evaluated through the same chain the
    # submission goes through. Nothing here reads solution/, which is not in this
    # image, and nothing here reads a stored constant.
    anchor = anchors_module.measure(RUN_ROOT, VERIFIER_RUN_TIMEOUT_SEC, state, evaluate_ledger)
    state["baseline_metric"] = anchor["baseline_metric"]
    state["target_metric"] = anchor["target_metric"]
    state["anchors_state"] = anchor["state"]
    state["anchors_gap"] = anchor["gap"]
    state["baseline_source"] = anchor["baseline_source"]
    state["target_source"] = anchor["target_source"]
    state["anchor_seconds"] = anchor["seconds"]
    sys.stdout.write(json.dumps({"anchors": {
        "baseline_metric": anchor["baseline_metric"],
        "target_metric": anchor["target_metric"],
        "state": anchor["state"],
        "handed": anchor["handed_probe"],
        "refined": anchor["refined_probe"],
    }}, sort_keys=True) + "\n")
    sys.stdout.flush()

    submission = submission_path()
    run = runner.launch(
        submission,
        RUN_ROOT,
        VERIFIER_RUN_TIMEOUT_SEC,
        schedule_hint=int(state.get("max_schedule_steps", 0) or 0),
    )
    telemetry = build_telemetry(run, state)
    result = grade_telemetry(telemetry)
    result["metric"]["submission_path"] = str(submission)
    emit(result["reward"], result["reason"], result["metric"], result["verdicts"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001  a grader that dies quietly is the defect
        traceback.print_exc()
        emit(
            0.0,
            "grader-internal-error",
            {
                "graded_step": None,
                "baseline": None,
                "target": None,
                "graded_quantity": "crossing_step",
            },
        )
        raise SystemExit(1)
