"""The refinement loop, run by the VERIFIER rather than reported by the agent.

The bound loop is: the harness lets the agent solve and submit, grades the
submission, summarizes the trajectory, and starts the next iteration with that
summary. The running context carries every prior approach and its reward. The
session ends when `max_timeout` or `max_attempts` trips, whichever binds first,
and one attempt must complete inside `budget_hours`. `max_timeout` and
`budget_hours` are never aliases.

The three numbers are deliberately not repeated in this sentence, and not
repeated in `BINDINGS` below, because repeating them is what broke this module.
It carried 50 attempts at 0.12 h while `task.toml [metadata.optimization]` bound
20 at 0.3 h, and `checkers.check_bound_envelope_respected` charges every measured
attempt against whichever of the two it is handed, so the restatement decided
whether an attempt could be graded at all. The manifest is the one declaration
and this module now reads it.

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
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import anchors as anchors_module  # noqa: E402
import bia_recipe  # noqa: E402
import checkers  # noqa: E402
import runner  # noqa: E402
import train_eval  # noqa: E402

SCHEMA = "bia.session/v1"
SLOT = "OER-02"

MANIFEST = HERE.parent / "task.toml"

BOUND_FIELDS = (
    ("max_timeout_hours", float),
    ("budget_hours", float),
    ("max_attempts", int),
    ("final_selection", str),
    ("final_selection_k", int),
)


def bound_optimization() -> dict:
    """The optimization bindings, READ from `task.toml [metadata.optimization]`.

    That table is the canonical declaration: its own header says so, and the
    values it carries are the ones `requirements/bia-environment-spec.md` binds
    under the twenty-attempt election that supersedes the earlier fifty. Reading
    them is the whole point, so every field is a strict subscript and a missing
    one raises here rather than defaulting to a number this module chose. A
    default would be a restatement wearing a fallback's clothes, and it is the
    restatement that drifted.

    The manifest travels into the verifier image beside `tests/`, so this path
    resolves both in the delivered container and in the bundle tree on a host.
    """
    with MANIFEST.open("rb") as handle:
        table = tomllib.load(handle)["metadata"]["optimization"]
    return {name: cast(table[name]) for name, cast in BOUND_FIELDS}


BINDINGS = bound_optimization()

CONSOLIDATION = {
    "multiplicity": 3,
    "min_frontier_improvements": 3,
    "min_inherited_components": 2,
    "min_determining_index": 3,
}

# The reward scale used to be anchored HERE, by the literals 3250 and 2690 under
# `anchors_state: "measured"`. They were measured, but on the upstream nanogpt
# speedrun rather than on this substrate, and at the operating point this bundle
# binds no run of any recipe comes within an order of magnitude of either. The
# endpoints are now re-derived on every graded run by `tests/anchors.py`, which
# trains probe recipes through the same loop an attempt goes through. There is no
# anchor literal left in this module to drift.
PASS_THRESHOLD = 0.65

PUBLIC_SHAPE_KEYS = (
    "vocab_size", "num_layers", "model_dim", "head_dim", "seq_len",
    "batch_tokens_per_step", "max_steps", "eval_stride", "sustain_window",
)


def load_shape() -> dict:
    path = os.environ.get("BIA_SHAPE")
    if path:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return json.loads((HERE.parent / "environment" / "shape.json").read_text(encoding="utf-8"))


def measure_anchors(substrate, shape: dict) -> dict:
    """Both endpoints of the reward scale, measured on THIS substrate, right now.

    There is deliberately no environment override and no default. An override
    would be an authored anchor wearing a fallback's clothes, and it is the
    authored anchor this repair removes. If the probes cannot build a scale the
    returned document says so in `anchors_state` and carries null endpoints, and
    `grade.py` turns that into an attributed zero rather than into a number.
    """
    measured = anchors_module.measure(substrate, shape)
    measured["target_loss"] = float(shape["target_loss"])
    measured["pass_threshold"] = PASS_THRESHOLD
    return measured


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
    bindings = dict(BINDINGS)
    if max_attempts is not None:
        bindings["max_attempts"] = int(max_attempts)
        bindings["final_selection_k"] = int(max_attempts)
    per_attempt_bound = float(bindings["budget_hours"]) * 3600.0
    session_bound = float(bindings["max_timeout_hours"]) * 3600.0
    substrate = train_eval.Substrate(shape)

    # The anchors are MEASURED, so measuring them costs real training. That cost
    # is charged to the verifier and not to the session: `opened` is started
    # after this returns, because `max_timeout_hours` bounds the refinement loop
    # across attempts and a submission must not lose attempts to the verifier
    # building its own scale. `anchors["seconds"]` carries what it cost.
    anchors = measure_anchors(substrate, shape)

    public_shape = {key: shape[key] for key in PUBLIC_SHAPE_KEYS if key in shape}

    attempts: list = []
    failures: list = []
    opened = time.monotonic()
    index = 0
    # `spent` counts LAUNCHES, not successful ones. `len(attempts)` counted only
    # the launches that came back with a proposal, which is why a submission that
    # never returned one drove this loop forever. See the failed-launch branch.
    spent = 0
    while spent < int(bindings["max_attempts"]):
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
                        ("baseline_metric", "target_metric", "target_loss",
                         "pass_threshold", "anchors_state")},
            "bindings": bindings,
            "consolidation": dict(CONSOLIDATION),
            "attempts": prior,
            "summary_digest": digest,
        }
        started = time.monotonic()
        launched = runner.launch(submission, state, per_attempt_bound)
        if launched.proposal is None:
            # A FAILED LAUNCH SPENDS AN ATTEMPT. It used to `continue` without
            # touching `attempts`, and `attempts` is the loop variable, so a
            # submission that never produced a proposal never advanced the loop
            # at all: measured at 5299 failed launches in 60 s, which projects to
            # about 1.9 million iterations before `max_timeout_hours` could stop
            # it. A submission that returns nothing could therefore not score
            # low, only hang, and hanging is not a score. `spent` is what the
            # loop now counts, so a launch that produced nothing costs exactly
            # what a launch that produced something costs: one attempt out of
            # `max_attempts`. The failure is still recorded in `failures` with
            # its reason, and it still contributes no attempt row, so it can
            # never manufacture a crossing -- it only stops the session running
            # forever.
            spent += 1
            failures.append({"attempt_ordinal": index, "spent": spent,
                             "why": launched.error,
                             "seconds": round(time.monotonic() - started, 6)})
            continue
        spent += 1
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
        "attempts_spent": spent,
        "attempts": attempts,
        "failed_launches": failures,
    }
