"""The refinement loop, run by the VERIFIER rather than reported by the agent.

The bound loop is: the harness lets the agent solve and submit, grades the
submission, summarizes the trajectory, and starts the next iteration with that
summary. The running context carries every prior approach and its reward. The
session ends when `max_timeout` (6.0 h, across attempts) or `max_attempts` (50)
trips, whichever binds first, and one attempt must complete inside
`budget_hours` (0.12 h). `max_timeout` and `budget_hours` are never aliases.

This module is the only thing that writes the session record, and it writes only
what it measured itself. The submission's `report` block is copied across
verbatim and is never read for a graded number; it exists so the DIVERGENCE
checker has something to reconcile against.

TERMINATION. Three things in this module were unbounded and are now bound, and
each bound is enforced against the value `task.toml` already binds rather than
against a number invented here:

1. The graded work itself. `budget_hours` is the field task.toml names the
   "per-attempt completion bound", and `checkers.check_bound_envelope_respected`
   GRADES it. It used to be handed only to `runner.launch`, which bounds the
   submission's proposal subprocess -- a few tens of milliseconds -- and never to
   `train_eval.train_and_evaluate`, which is where every second of an attempt
   actually goes. The bound now reaches the work it names: training carries the
   same deadline and raises `train_eval.BudgetExceeded` when it passes, so an
   over-budget attempt ends as an attributed refusal instead of running forever.
2. The operating point. `load_operating_point` reads `tests/operating_point.json`,
   which is beside this file and therefore resolves inside the verifier image
   that `tests/Dockerfile` actually builds. The previous fallback reached
   `HERE.parent / "environment" / "shape.json"`, and `tests/Dockerfile` copies
   only `tests/` into `/verifier/tests/`, so that path does not exist in the
   pinned image and the verifier crashed into `grader-internal-error` there.
3. The failed-launch loop. The `while` condition counts COMPLETED attempts, so a
   submission that never yields a proposal used to spin until `max_timeout`
   tripped six hours later. Launches are now counted and bounded too.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bia_recipe  # noqa: E402
import checkers  # noqa: E402
import runner  # noqa: E402
import train_eval  # noqa: E402

SCHEMA = "bia.session/v1"
SLOT = "OER-02"

BINDINGS = {
    "max_timeout_hours": 6.0,          # refinement-loop terminator across attempts
    "budget_hours": 0.12,              # per-attempt completion bound, 7.2 minutes
    "max_attempts": 50,
    "final_selection": "best",
    "final_selection_k": 50,
}

CONSOLIDATION = {
    "multiplicity": 3,
    "min_frontier_improvements": 3,
    "min_inherited_components": 2,
    "min_determining_index": 3,
}

ANCHORS = {
    "baseline_metric": 3250,
    "target_metric": 2690,
    "target_loss": 3.28,
    "pass_threshold": 0.65,
    "anchors_state": "measured",
    "authority": "requirements/bia-environment-spec.md lines 297 to 321",
}

# The bound operating points, as bytes beside this file. See the module docstring
# item 2 for why this is not read out of environment/.
OPERATING_POINT_PATH = HERE / "operating_point.json"

PUBLIC_SHAPE_KEYS = (
    "vocab_size", "num_layers", "model_dim", "head_dim", "seq_len",
    "batch_tokens_per_step", "max_steps", "eval_stride", "sustain_window",
)


def load_operating_point() -> dict:
    document = json.loads(OPERATING_POINT_PATH.read_text(encoding="utf-8"))
    name = os.environ.get("BIA_OPERATING_POINT") or document["selected"]
    if name not in document["points"]:
        raise KeyError("no operating point named {!r} in {}".format(name, OPERATING_POINT_PATH))
    point = dict(document["points"][name])
    point["name"] = name
    point["window_ceiling_seconds"] = float(
        document["verifier_robustness"]["window_ceiling_seconds"])
    return point


def load_shape(point: dict | None = None) -> dict:
    path = os.environ.get("BIA_SHAPE")
    if path:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    point = point or load_operating_point()
    shape = dict(point["shape"])
    if point.get("device"):
        shape["device"] = point["device"]
    return shape


def load_anchors(shape: dict, point: dict | None = None) -> dict:
    anchors = dict(ANCHORS)
    if point is not None:
        anchors.update(point["anchors"])
    anchors["target_loss"] = float(shape["target_loss"])
    for key in ("baseline_metric", "target_metric", "pass_threshold"):
        override = os.environ.get("BIA_" + key.upper())
        if override is not None:
            anchors[key] = float(override) if key == "pass_threshold" else int(float(override))
    return anchors


def _inherited(recipe: dict, claims, prior: list) -> dict:
    """Verify an inheritance claim against the components themselves.

    A claim is only inheritance when the named earlier attempt actually carried
    the same value for that component. An unbacked claim is dropped rather than
    counted, so `inherits_from` cannot manufacture carried state.
    """
    by_index = {int(row["index"]): row for row in prior}
    out = {}
    for claim in claims or []:
        try:
            index = int(claim)
        except (TypeError, ValueError):
            continue
        source = by_index.get(index)
        if source is None:
            continue
        shared = sorted(key for key, value in recipe.items()
                        if key in source["recipe"] and source["recipe"][key] == value)
        if shared:
            out[str(index)] = shared
    return out


def run_session(submission: pathlib.Path, shape: dict | None = None,
                max_attempts: int | None = None) -> dict:
    """Drive the whole session and return the record the checkers grade."""
    point = load_operating_point()
    shape = shape or load_shape(point)
    anchors = load_anchors(shape, point)
    train_eval.pin_threads(point.get("torch_threads"))
    bindings = dict(BINDINGS)
    if max_attempts is not None:
        bindings["max_attempts"] = int(max_attempts)
        bindings["final_selection_k"] = int(max_attempts)
    per_attempt_bound = float(bindings["budget_hours"]) * 3600.0
    session_bound = float(bindings["max_timeout_hours"]) * 3600.0
    substrate = train_eval.Substrate(shape)
    public_shape = {key: shape[key] for key in PUBLIC_SHAPE_KEYS if key in shape}

    # A launch that yields no proposal appends nothing to `attempts`, so the loop
    # condition below cannot retire it. Counting launches is what stops a
    # submission that always fails from spinning until max_timeout trips.
    launch_ceiling = int(bindings["max_attempts"]) * 2

    attempts: list = []
    failures: list = []
    opened = time.monotonic()
    index = 0
    while len(attempts) < int(bindings["max_attempts"]):
        if time.monotonic() - opened > session_bound:
            failures.append({"why": "max-timeout-tripped-before-max-attempts",
                             "completed": len(attempts)})
            break
        if index >= launch_ceiling:
            failures.append({"why": "launch-ceiling-tripped-before-max-attempts",
                             "completed": len(attempts), "launches": index,
                             "launch_ceiling": launch_ceiling})
            break
        index += 1
        prior = [
            {
                "index": row["index"],
                "recipe": row["recipe"],
                "recipe_fingerprint": row["recipe_fingerprint"],
                "sustained_crossing_step": row["sustained_crossing_step"],
                "attempt_reward": row["attempt_reward"],
                "reported": row["reported"],
            }
            for row in attempts
        ]
        digest = checkers.summary_digest(attempts)
        state = {
            "index": len(attempts) + 1,
            "shape": public_shape,
            "anchors": {key: anchors[key] for key in
                        ("baseline_metric", "target_metric", "target_loss", "pass_threshold")},
            "bindings": bindings,
            "consolidation": dict(CONSOLIDATION),
            "attempts": prior,
            "summary_digest": digest,
        }
        started = time.monotonic()
        launched = runner.launch(submission, state, per_attempt_bound)
        if launched.proposal is None:
            failures.append({"attempt_ordinal": index, "why": launched.error})
            continue
        proposal = launched.proposal
        normalized = bia_recipe.normalize(proposal.get("recipe"), shape)
        recipe = normalized["recipe"]
        seed = train_eval.seed_for(normalized["fingerprint"])
        try:
            measured = train_eval.train_and_evaluate(
                substrate, recipe, seed=seed, index=len(attempts) + 1,
                deadline=started + per_attempt_bound)
        except train_eval.BudgetExceeded as exceeded:
            failures.append({"attempt_ordinal": index, "why": "attempt-budget-exceeded",
                             "budget_seconds": per_attempt_bound,
                             "halted_at_step": exceeded.step,
                             "of_steps": exceeded.total})
            break
        crossing = checkers.sustained_crossing(
            measured["verifier_evals"], float(anchors["target_loss"]), int(shape["sustain_window"]))
        elapsed = time.monotonic() - started
        span = float(anchors["baseline_metric"]) - float(anchors["target_metric"])
        contribution = 0.0 if crossing is None or span <= 0 else min(max(
            (float(anchors["baseline_metric"]) - float(crossing)) / span, 0.0), 1.0)
        attempts.append({
            "index": len(attempts) + 1,
            "run_digest": measured["run_digest"],
            "recipe": recipe,
            "recipe_fingerprint": normalized["fingerprint"],
            "frozen_axis_writes": normalized["frozen_axis_writes"],
            "unknown_keys": normalized["unknown_keys"],
            "summary_digest_expected": digest,
            "summary_digest_echoed": str(proposal.get("summary_digest", "")),
            "carried_attempt_count": len(prior),
            "inherits_from": list(proposal.get("inherits_from") or []),
            "inherited_components": _inherited(recipe, proposal.get("inherits_from"), prior),
            "attempt_seconds": round(elapsed, 6),
            "halted_at_step": measured["halted_at_step"],
            "forward_passes_per_step": measured["forward_passes_per_step"],
            "backward_passes_per_step": measured["backward_passes_per_step"],
            "verifier_evals": measured["verifier_evals"],
            "sustained_crossing_step": crossing,
            "attempt_reward": round(contribution, 6),
            "reported": proposal.get("report") if isinstance(proposal.get("report"), dict) else {},
        })
    return {
        "schema": SCHEMA,
        "slot": SLOT,
        "bindings": bindings,
        "anchors": anchors,
        "consolidation": dict(CONSOLIDATION),
        "eval_schedule": {
            "owner": "verifier",
            "stride": int(shape["eval_stride"]),
            "sustain_window": int(shape["sustain_window"]),
            "smoothing": "none",
        },
        "session_seconds": round(time.monotonic() - opened, 6),
        "attempts": attempts,
        "failed_launches": failures,
    }
