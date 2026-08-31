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

PUBLIC_SHAPE_KEYS = (
    "vocab_size", "num_layers", "model_dim", "head_dim", "seq_len",
    "batch_tokens_per_step", "max_steps", "eval_stride", "sustain_window",
)


def load_shape() -> dict:
    path = os.environ.get("BIA_SHAPE")
    if path:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return json.loads((HERE.parent / "environment" / "shape.json").read_text(encoding="utf-8"))


def load_anchors(shape: dict) -> dict:
    anchors = dict(ANCHORS)
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
    shape = shape or load_shape()
    anchors = load_anchors(shape)
    bindings = dict(BINDINGS)
    if max_attempts is not None:
        bindings["max_attempts"] = int(max_attempts)
        bindings["final_selection_k"] = int(max_attempts)
    per_attempt_bound = float(bindings["budget_hours"]) * 3600.0
    session_bound = float(bindings["max_timeout_hours"]) * 3600.0
    substrate = train_eval.Substrate(shape)
    public_shape = {key: shape[key] for key in PUBLIC_SHAPE_KEYS if key in shape}

    attempts: list = []
    failures: list = []
    opened = time.monotonic()
    index = 0
    while len(attempts) < int(bindings["max_attempts"]):
        if time.monotonic() - opened > session_bound:
            failures.append({"why": "max-timeout-tripped-before-max-attempts",
                             "completed": len(attempts)})
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
        measured = train_eval.train_and_evaluate(substrate, recipe, seed=seed, index=len(attempts) + 1)
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
