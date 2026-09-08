#!/usr/bin/env python3
"""The grader. Runs the frozen training run ONCE and divides two measurements it made itself.

THE REWARD SCALE
    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both endpoints of that scale are MEASURED HERE, on this run, from the snapshots this
verifier's own frozen training run just produced:

    default_loss   -- the shipped environment/default_selection.json, which keeps the
                      FINAL snapshot and nothing else. The floor. A submission that ties
                      it scores 0.0.
    reference_loss -- tests/private/reference_selection.json, a stronger retention
                      selection that exists only inside the verifier image. The bar. A
                      submission that reaches it scores 1.0, and beating it also scores
                      1.0.

Neither number is stored, cached, defaulted or read from the bundle. There is no literal
anywhere in this file or in this bundle for either of them. If either anchor fails to
produce a finite loss the run REFUSES with a named reason rather than substituting
anything, and if the two coincide it refuses with calibration-span-nonpositive.

WHY THIS IS CHEAP DESPITE MEASURING BOTH ENDS
    All three selections are scored against the SAME snapshots from the SAME single
    training run, because no selection can influence training. Grading therefore pays
    for one training run and three evaluations, not three training runs.

WHAT IS NOT GRADED
    Nothing the submission reports about itself. The submission is a retention selection
    -- a list of snapshot ids and a list of weights -- and the only things read out of it
    are those. `selection.notes` is free text, is copied into the score document for a
    human reader, and is never parsed.
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
import selection_schema   # noqa: E402  the verifier's OWN copy

SUBMISSION_PATHS = (
    Path(os.environ.get("OER_SUBMISSION", "/workspace/submission/selection.json")),
    Path("/workspace/selection.json"),
)
DEFAULT_SELECTION = HERE / "default_selection.json"
REFERENCE_SELECTION = HERE / "private" / "reference_selection.json"
TRAIN_SLICE = HERE / "corpus" / "train_slice.bin"
HOLDOUT_SLICE = HERE / "holdout" / "val_slice.bin"

LOGDIR = Path(os.environ.get("OER_LOGDIR", "/logs/verifier"))


# ---------------------------------------------------------------------------
# Reward carriers
# ---------------------------------------------------------------------------

def emit(reward: float, reason: str, document: dict) -> None:
    """Write every carrier. reward.txt is the bare float, reward.json the pair."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    reward = float(reward)
    document = dict(document)
    document["reward"] = reward
    document["reason"] = reason
    (LOGDIR / "score.json").write_text(json.dumps(document, indent=2, default=str),
                                       encoding="utf-8")
    (LOGDIR / "reward.json").write_text(
        json.dumps({"reward": reward, "reason": reason}) + "\n", encoding="utf-8")
    (LOGDIR / "reward.txt").write_text(f"{reward}\n", encoding="utf-8")
    print(f"\n[reward] {reward}  [reason] {reason}", flush=True)


def refuse(reason: str, detail: str, document: dict | None = None) -> int:
    """Reward 0.0 with a machine-readable reason. Never a scaled value."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    started = time.time()
    document = {
        "slot": os.environ.get("OER_SLOT", "OER-22"),
        "metric": "validation cross-entropy in nats on the verifier's held-out FineWeb slice",
        "metric_direction": "lower is better",
        "reward_formula": "clamp((default_loss - agent_loss) / (default_loss - reference_loss), 0, 1)",
        "anchor_provenance": "both endpoints measured by this verifier on this run, from the "
                             "snapshots of the single frozen training run it performed here",
    }

    # --- hygiene gate. Refuses, never scales. ------------------------------
    submitted = locate_submission()
    if submitted is None:
        return refuse("submission-absent",
                      "no selection at " + " or ".join(str(p) for p in SUBMISSION_PATHS), document)
    document["submission_path"] = str(submitted)
    try:
        agent_selection = selection_schema.load(submitted)
    except selection_schema.Refusal as exc:
        return refuse(exc.reason, exc.detail, document)
    document["agent_selection"] = {"keep": agent_selection["keep"],
                                   "weights": agent_selection["weights"]}
    document["agent_notes"] = str(agent_selection.get("notes", ""))[:2000]

    # --- the verifier's own inputs ----------------------------------------
    try:
        default_selection = selection_schema.load(DEFAULT_SELECTION)
        reference_selection = selection_schema.load(REFERENCE_SELECTION)
    except selection_schema.Refusal as exc:
        return refuse("verifier-anchor-selection-invalid",
                      f"{exc.reason}: {exc.detail}", document)

    import torch
    if not torch.cuda.is_available():
        return refuse("accelerator-absent",
                      "no CUDA device is visible to the verifier", document)
    document["device"] = torch.cuda.get_device_name(0)

    harness.configure_backends()
    spec = harness.load_spec()
    document["substrate"] = {"architecture": spec["architecture"], "budget": spec["budget"],
                             "snapshots": spec["snapshots"], "evaluation": spec["evaluation"],
                             "training_recipe": spec["training_recipe"], "init": spec["init"]}

    try:
        train_tokens = harness.to_device_tokens(harness.load_shard(TRAIN_SLICE), "cuda")
        holdout_tokens = harness.to_device_tokens(harness.load_shard(HOLDOUT_SLICE), "cuda")
        runner = harness.Runner("cuda", spec)
    except Exception as exc:
        return refuse("verifier-substrate-unavailable",
                      f"{type(exc).__name__}: {exc}", document)
    document["parameters"] = sum(p.numel() for p in runner.model.parameters())
    print(f"[setup] {document['parameters'] / 1e6:.1f}M parameters, "
          f"{time.time() - started:.1f}s", flush=True)

    # --- the ONE frozen training run. No selection can influence it. -------
    print(f"\n[train] the frozen run: {runner.micro_steps} micro-batches, "
          f"{runner.snapshot_count} snapshots every {runner.snapshot_every} steps", flush=True)
    try:
        run = runner.train_snapshots(train_tokens, progress=True)
    except Exception as exc:
        traceback.print_exc()
        return refuse("frozen-run-failed",
                      f"the frozen training run raised {type(exc).__name__}: {exc}", document)
    snapshots = run.pop("snapshots")
    document["frozen_run"] = run
    print(f"[train] done in {run['train_seconds']}s", flush=True)

    # --- the three evaluations. Anchors first, so a crash in the submission's
    # --- combination still leaves both measured endpoints in the document. --
    results = {}
    for label, selection in (("default", default_selection),
                             ("reference", reference_selection),
                             ("agent", agent_selection)):
        print(f"\n[eval:{label}] keeping {selection['keep']}", flush=True)
        try:
            report = runner.evaluate_selection(selection, snapshots, holdout_tokens)
        except Exception as exc:
            traceback.print_exc()
            if label == "agent":
                return refuse("agent-selection-failed",
                              f"the submitted selection raised {type(exc).__name__}: {exc}",
                              {**document, "evaluations": results})
            return refuse("anchor-evaluation-failed",
                          f"the {label} anchor raised {type(exc).__name__}: {exc}",
                          {**document, "evaluations": results})
        results[label] = report
        print(f"[eval:{label}] val_loss={report['val_loss']:.6f} "
              f"({report['eval_seconds']}s)", flush=True)
    document["evaluations"] = results
    document["grading_seconds"] = round(time.time() - started, 1)

    default_loss = results["default"]["val_loss"]
    reference_loss = results["reference"]["val_loss"]
    agent_loss = results["agent"]["val_loss"]

    for label in ("default", "reference"):
        if not math.isfinite(results[label]["val_loss"]):
            return refuse("anchor-diverged",
                          f"the {label} anchor produced no finite loss, so the reward scale "
                          f"has no {label} endpoint on this run; refusing rather than "
                          f"substituting a stored value",
                          document)

    span = default_loss - reference_loss
    document["default_loss"] = default_loss
    document["reference_loss"] = reference_loss
    document["agent_loss"] = agent_loss if math.isfinite(agent_loss) else None
    document["span"] = span

    if span <= 0.0:
        return refuse("calibration-span-nonpositive",
                      f"the default anchor measured {default_loss!r} and the reference anchor "
                      f"measured {reference_loss!r}, leaving no positive span to scale between. "
                      f"This is a property of this run's measurements, not of the submission.",
                      document)

    if not math.isfinite(agent_loss):
        return refuse("agent-selection-diverged",
                      "the submitted selection produced no finite validation loss", document)

    raw = (default_loss - agent_loss) / span
    reward = min(max(raw, 0.0), 1.0)
    document["raw_score"] = raw
    document["gap_closed_percent"] = round(100.0 * raw, 3)

    if raw <= 0.0:
        reason = "no-improvement-over-final-snapshot"
    elif raw >= 1.0:
        reason = "reference-reached"
    else:
        reason = "partial-improvement"
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
