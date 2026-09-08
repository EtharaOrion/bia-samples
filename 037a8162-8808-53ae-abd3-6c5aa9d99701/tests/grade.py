#!/usr/bin/env python3
"""The grader for OER-01. Trains three models and divides two measurements it made itself.

WHAT IS GRADED HERE IS NOT A LOSS. IT IS A NUMBER OF TOKENS.

    target         -- the shipped environment/default_recipe.json's OWN evaluation
                      loss after it has spent the entire budget ceiling. Measured on
                      this run. There is no literal for it anywhere in this bundle.

    <recipe>_micro -- the FIRST point on the evaluation grid at which that recipe's
                      evaluation loss is at or below the target. Measured, never
                      interpolated: it is one of the 24 points that were actually
                      evaluated.

    reward = clamp( (control_micro - agent_micro) / (control_micro - reference_micro), 0, 1 )

Both endpoints of that scale are measured here, on this run:

    control_micro   -- where the shipped default first reaches its own final loss.
                       The floor. A submission that ties it scores 0.0.
    reference_micro -- where tests/private/reference_recipe.json first reaches the
                       same target. The bar. Reaching it scores 1.0, beating it also
                       scores 1.0.

Nothing is stored, cached or defaulted. If either anchor fails to produce a finite
curve, or the reference does not actually get there sooner than the control, the run
REFUSES with a named reason rather than substituting anything.

WHY THIS IS A DIFFERENT QUESTION FROM "REACH THE LOWEST LOSS"
    A schedule that holds a high peak for the whole ceiling ends lower than one that
    decays early, and crosses a fixed bar LATER. The two objectives disagree, and this
    slot grades the second one.

THE RECORD-DISPLACEMENT GATE
    A recipe that restates a recipe already in environment/published_records.json is
    refused with `record-replayed`. That is a gate and not a grade: it never scales the
    float and never contributes a partial credit. See records.py for what counts as the
    same record.

WHAT IS NOT GRADED
    Nothing the submission reports about itself. `notes` is free text, copied into the
    score document for a human reader and never parsed.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harness            # noqa: E402  the verifier's OWN copy
import recipe_schema      # noqa: E402  the verifier's OWN copy
import records            # noqa: E402  the verifier's OWN copy

SUBMISSION_PATHS = (
    Path(os.environ.get("OER01_SUBMISSION", "/workspace/submission/recipe.json")),
    Path("/workspace/recipe.json"),
)
DEFAULT_RECIPE = HERE / "default_recipe.json"
REFERENCE_RECIPE = HERE / "private" / "reference_recipe.json"
RECORDS = HERE / "published_records.json"
TRAIN_SLICE = HERE / "corpus" / "train_slice.bin"
HOLDOUT_SLICE = HERE / "holdout" / "val_slice.bin"

LOGDIR = Path(os.environ.get("OER01_LOGDIR", "/logs/verifier"))


def emit(reward: float, reason: str, document: dict) -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    reward = float(reward)
    document = dict(document)
    document["reward"] = reward
    document["reason"] = reason
    (LOGDIR / "score.json").write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    (LOGDIR / "reward.json").write_text(
        json.dumps({"reward": reward, "reason": reason}) + "\n", encoding="utf-8")
    (LOGDIR / "reward.txt").write_text(f"{reward}\n", encoding="utf-8")
    print(f"\n[reward] {reward}  [reason] {reason}", flush=True)


def refuse(reason: str, detail: str, document: dict | None = None) -> int:
    doc = dict(document or {})
    doc["refusal_detail"] = detail
    emit(0.0, reason, doc)
    print(f"[refused] {reason}: {detail}", flush=True)
    return 0


def locate_submission():
    for path in SUBMISSION_PATHS:
        if path.is_file():
            return path
    return None


def main() -> int:
    started = time.time()
    document = {
        "slot": "OER-01",
        "graded_quantity": "micro-batches consumed before the target validation loss is first reached",
        "metric": "tokens to target, lower is better",
        "metric_direction": "lower is better",
        "reward_formula": "clamp((control_micro - agent_micro) / (control_micro - reference_micro), 0, 1)",
        "target_provenance": "the shipped default recipe's own end-of-ceiling loss, measured on this run",
        "anchor_provenance": "both endpoints measured from scratch by this verifier on this run",
    }

    submitted = locate_submission()
    if submitted is None:
        return refuse("submission-absent",
                      "no recipe at " + " or ".join(str(p) for p in SUBMISSION_PATHS), document)
    document["submission_path"] = str(submitted)
    try:
        agent_recipe = recipe_schema.load(submitted)
    except recipe_schema.Refusal as exc:
        return refuse(exc.reason, exc.detail, document)
    document["agent_recipe"] = {k: v for k, v in agent_recipe.items() if k != "notes"}
    document["agent_notes"] = str(agent_recipe.get("notes", ""))[:2000]

    # --- the record-displacement gate. Refuses, never scales. --------------
    try:
        corpus = records.load_records(RECORDS)
    except Exception as exc:
        return refuse("verifier-substrate-unavailable",
                      f"the published-record corpus could not be read: {exc}", document)
    document["published_records"] = len(corpus)
    document["agent_recipe_digest"] = records.digest(agent_recipe)
    replay = records.find_replay(agent_recipe, corpus)
    if replay is not None:
        return refuse("record-replayed",
                      f"the submitted recipe restates published record "
                      f"{replay.get('record_id')!r} on every graded field. A published "
                      f"record is not a displacement of the record.", document)

    try:
        default_recipe = recipe_schema.load(DEFAULT_RECIPE)
        reference_recipe = recipe_schema.load(REFERENCE_RECIPE)
    except recipe_schema.Refusal as exc:
        return refuse("verifier-anchor-recipe-invalid", f"{exc.reason}: {exc.detail}", document)
    if records.find_replay(reference_recipe, corpus) is not None:
        return refuse("verifier-anchor-recipe-invalid",
                      "the verifier's own reference recipe is in the published corpus, "
                      "so the bar it sets is already excluded from the submission space",
                      document)

    import torch
    if not torch.cuda.is_available():
        return refuse("accelerator-absent", "no CUDA device is visible to the verifier", document)
    document["device"] = torch.cuda.get_device_name(0)

    harness.configure_backends()
    spec = harness.load_spec()
    document["substrate"] = {"architecture": spec["architecture"], "budget": spec["budget"],
                             "evaluation": spec["evaluation"], "init": spec["init"]}

    try:
        train_tokens = harness.to_device_tokens(harness.load_shard(TRAIN_SLICE), "cuda")
        holdout_tokens = harness.to_device_tokens(harness.load_shard(HOLDOUT_SLICE), "cuda")
        runner = harness.Runner("cuda", spec)
    except Exception as exc:
        traceback.print_exc()
        return refuse("verifier-substrate-unavailable", f"{type(exc).__name__}: {exc}", document)
    document["parameters"] = sum(p.numel() for p in runner.model.parameters())
    print(f"[setup] {document['parameters'] / 1e6:.1f}M parameters, "
          f"{time.time() - started:.1f}s", flush=True)

    runs = {}
    for label, recipe in (("control", default_recipe),
                          ("reference", reference_recipe),
                          ("agent", agent_recipe)):
        print(f"\n[train:{label}] ceiling {spec['budget']['micro_steps']} micro-batches, "
              f"grid every {spec['evaluation']['grid_micro_steps']}", flush=True)
        run_started = time.time()
        try:
            report = runner.run_curve(recipe, train_tokens, holdout_tokens, progress=6)
        except Exception as exc:
            traceback.print_exc()
            if label == "agent":
                return refuse("agent-run-failed",
                              f"the submitted recipe raised {type(exc).__name__}: {exc}",
                              {**document, "runs": runs})
            return refuse("anchor-run-failed",
                          f"the {label} anchor raised {type(exc).__name__}: {exc}",
                          {**document, "runs": runs})
        report["wall_seconds"] = round(time.time() - run_started, 1)
        runs[label] = report
        print(f"[train:{label}] final_eval_loss={report['final_eval_loss']:.6f} "
              f"({report['wall_seconds']}s)", flush=True)
    document["runs"] = runs
    document["grading_seconds"] = round(time.time() - started, 1)

    for label in ("control", "reference"):
        if not math.isfinite(runs[label]["final_eval_loss"]):
            return refuse("anchor-diverged",
                          f"the {label} anchor produced no finite curve, so the reward scale "
                          f"has no {label} endpoint on this run; refusing rather than "
                          f"substituting a stored value", document)

    # --- the target IS the control's own end-of-ceiling loss ---------------
    target = runs["control"]["final_eval_loss"]
    document["target"] = target

    crossings = {}
    for label in ("control", "reference", "agent"):
        crossings[label] = harness.first_crossing(runs[label]["curve"], target)
    document["crossings"] = crossings
    print(f"\n[target] {target:.6f}   crossings {crossings}", flush=True)

    if crossings["control"] is None:
        return refuse("anchor-diverged",
                      "the control never reached its own final loss on the grid, which "
                      "means the grid did not observe the point that defines the target",
                      document)
    if crossings["reference"] is None:
        return refuse("verifier-anchor-recipe-invalid",
                      "the reference recipe never reached the target, so this run has no "
                      "upper endpoint; refusing rather than substituting one", document)

    control_micro = crossings["control"]
    reference_micro = crossings["reference"]
    span = control_micro - reference_micro
    document["control_micro"] = control_micro
    document["reference_micro"] = reference_micro
    document["span_micro"] = span

    if span <= 0:
        return refuse("calibration-span-nonpositive",
                      f"the control crossed at {control_micro} micro-batches and the "
                      f"reference at {reference_micro}, leaving no positive span to scale "
                      f"between. This is a property of this run's measurements, not of "
                      f"the submission.", document)

    if crossings["agent"] is None:
        return refuse("target-not-reached",
                      f"the submitted recipe never reached the target {target:.6f} within "
                      f"the budget ceiling. Its final evaluation loss was "
                      f"{runs['agent']['final_eval_loss']:.6f}.", document)

    agent_micro = crossings["agent"]
    document["agent_micro"] = agent_micro
    document["agent_tokens"] = agent_micro * spec["budget"]["micro_batch"] * \
        spec["architecture"]["seq_len"]

    raw = (control_micro - agent_micro) / span
    reward = min(max(raw, 0.0), 1.0)
    document["raw_score"] = raw
    document["gap_closed_percent"] = round(100.0 * raw, 3)

    if raw <= 0.0:
        reason = "no-displacement-of-the-control"
    elif raw >= 1.0:
        reason = "reference-reached"
    else:
        reason = "partial-displacement"
    emit(reward, reason, document)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        emit(0.0, "grading-chain-failed", {"traceback": traceback.format_exc()[-4000:]})
        raise SystemExit(0)
