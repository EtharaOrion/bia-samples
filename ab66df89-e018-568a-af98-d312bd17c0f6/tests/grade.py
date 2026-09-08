#!/usr/bin/env python3
"""The grader. Trains three models under three compute allocations and divides two
measurements it made itself.

THE REWARD SCALE
    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both endpoints of that scale are MEASURED HERE, on this run, by real training runs of
this verifier's own harness:

    default_loss   -- the shipped environment/default_allocation.json, retrained from
                      scratch. The floor. A submission that ties it scores 0.0.
    reference_loss -- tests/private/reference_allocation.json, a stronger allocation
                      that exists only inside the verifier image. The bar. A
                      submission that reaches it scores 1.0, and beating it also
                      scores 1.0.

Neither number is stored, cached, defaulted or read from the bundle. There is no
literal anywhere in this file or in this bundle for either of them, and if either
anchor fails to produce a finite loss the run REFUSES with a named reason rather than
substituting anything.

WHY THE THREE RUNS ARE COMPARABLE, AND WHY THEY COST THE SAME
    Every admissible allocation spends at most the same declared FLOP budget, checked
    by the same arithmetic on the agent surface and here. Allocations that name the
    same menu entry share one built model, one compiled graph and one pristine set of
    initial tensors; allocations that name different entries necessarily build
    different models, which is the axis being graded. Every run is evaluated on the
    same window.

WHAT IS NOT GRADED
    Nothing the submission reports about itself. The allocation declares a model, a
    micro-step count and a gradient accumulation, and nothing else; `notes` is free
    text copied into the score document without being parsed.
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

import harness              # noqa: E402  the verifier's OWN copy
import allocation_schema    # noqa: E402  the verifier's OWN copy

SUBMISSION_PATHS = (
    Path(os.environ.get("OER16_SUBMISSION", "/workspace/submission/allocation.json")),
    Path("/workspace/allocation.json"),
)
DEFAULT_ALLOCATION = HERE / "default_allocation.json"
REFERENCE_ALLOCATION = HERE / "private" / "reference_allocation.json"
TRAIN_SLICE = HERE / "corpus" / "train_slice.bin"
HOLDOUT_SLICE = HERE / "holdout" / "val_slice.bin"

LOGDIR = Path(os.environ.get("OER16_LOGDIR", "/logs/verifier"))


def emit(reward: float, reason: str, document: dict) -> None:
    """Write every carrier. reward.txt is the bare float, reward.json the pair."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    reward = float(reward)
    document = dict(document)
    document["reward"] = reward
    document["reason"] = reason
    (LOGDIR / "score.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
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


def main() -> int:
    started = time.time()
    document = {
        "slot": os.environ.get("OER16_SLOT", "OER-16"),
        "metric": "validation cross-entropy in nats on the verifier's held-out FineWeb slice",
        "metric_direction": "lower is better",
        "reward_formula": "clamp((default_loss - agent_loss) / (default_loss - reference_loss), 0, 1)",
        "anchor_provenance": "both endpoints trained from scratch by this verifier on this run",
    }

    try:
        spec = harness.load_spec()
        recipe = harness.load_train_recipe()
    except Exception as exc:
        return refuse("verifier-substrate-unavailable", f"{type(exc).__name__}: {exc}", document)

    # --- hygiene gate. Refuses, never scales. ------------------------------
    submitted = locate_submission()
    if submitted is None:
        return refuse("submission-absent",
                      "no allocation at " + " or ".join(str(p) for p in SUBMISSION_PATHS),
                      document)
    document["submission_path"] = str(submitted)
    try:
        agent_alloc = allocation_schema.load(submitted, spec, harness)
    except allocation_schema.Refusal as exc:
        return refuse(exc.reason, exc.detail, document)
    document["agent_allocation"] = agent_alloc
    document["agent_notes"] = str(agent_alloc.get("notes", ""))[:2000]

    try:
        default_alloc = allocation_schema.load(DEFAULT_ALLOCATION, spec, harness)
        reference_alloc = allocation_schema.load(REFERENCE_ALLOCATION, spec, harness)
    except allocation_schema.Refusal as exc:
        return refuse("verifier-anchor-allocation-invalid", f"{exc.reason}: {exc.detail}", document)

    import torch
    if not torch.cuda.is_available():
        return refuse("accelerator-absent", "no CUDA device is visible to the verifier", document)
    document["device"] = torch.cuda.get_device_name(0)

    harness.configure_backends()
    document["substrate"] = {"architecture_common": spec["architecture_common"],
                             "model_menu": spec["model_menu"], "budget": spec["budget"],
                             "accounting": spec["accounting"], "evaluation": spec["evaluation"],
                             "init": spec["init"]}

    try:
        train_tokens = harness.to_device_tokens(harness.load_shard(TRAIN_SLICE), "cuda")
        holdout_tokens = harness.to_device_tokens(harness.load_shard(HOLDOUT_SLICE), "cuda")
    except Exception as exc:
        traceback.print_exc()
        return refuse("verifier-substrate-unavailable", f"{type(exc).__name__}: {exc}", document)

    # --- the three runs. Anchors first, so a crash in the submission's run
    # --- still leaves both measured endpoints in the score document. -------
    runners = {}
    runs = {}
    for label, alloc in (("default", default_alloc),
                         ("reference", reference_alloc),
                         ("agent", agent_alloc)):
        model = alloc["model"]
        spent = alloc["micro_steps"] * harness.micro_batch_flops(spec, model)
        print(f"\n[train:{label}] {model}  micro_steps={alloc['micro_steps']}  "
              f"grad_accum={alloc['grad_accum']}  "
              f"{100.0 * spent / int(spec['budget']['flop_budget']):.1f}% of the FLOP budget",
              flush=True)
        run_started = time.time()
        try:
            if model not in runners:
                runners[model] = harness.Runner(spec, model, "cuda")
            report = runners[model].run(alloc, recipe, train_tokens, holdout_tokens, progress=None)
        except Exception as exc:
            traceback.print_exc()
            if label == "agent":
                return refuse("agent-run-failed",
                              f"the submitted allocation raised {type(exc).__name__}: {exc}",
                              {**document, "runs": runs})
            return refuse("anchor-run-failed",
                          f"the {label} anchor raised {type(exc).__name__}: {exc}",
                          {**document, "runs": runs})
        report["wall_seconds"] = round(time.time() - run_started, 1)
        report["spent_flops"] = spent
        report["flop_budget"] = int(spec["budget"]["flop_budget"])
        report["allocation"] = alloc
        runs[label] = report
        print(f"[train:{label}] val_loss={report['val_loss']:.6f} "
              f"({report['wall_seconds']}s, {report['epochs']} epochs of the corpus)", flush=True)
    document["runs"] = runs
    document["grading_seconds"] = round(time.time() - started, 1)

    default_loss = runs["default"]["val_loss"]
    reference_loss = runs["reference"]["val_loss"]
    agent_loss = runs["agent"]["val_loss"]

    for label in ("default", "reference"):
        if not math.isfinite(runs[label]["val_loss"]):
            return refuse("anchor-diverged",
                          f"the {label} anchor produced no finite loss, so the reward "
                          f"scale has no {label} endpoint on this run; refusing rather "
                          f"than substituting a stored value", document)

    span = default_loss - reference_loss
    document["default_loss"] = default_loss
    document["reference_loss"] = reference_loss
    document["agent_loss"] = agent_loss if math.isfinite(agent_loss) else None
    document["span"] = span

    if span <= 0.0:
        return refuse("calibration-span-nonpositive",
                      f"the default anchor measured {default_loss!r} and the reference "
                      f"anchor measured {reference_loss!r}, leaving no positive span to "
                      f"scale between. This is a property of this run's measurements, "
                      f"not of the submission.", document)

    if not math.isfinite(agent_loss):
        return refuse("agent-allocation-diverged",
                      "the submitted allocation produced no finite validation loss", document)

    raw = (default_loss - agent_loss) / span
    reward = min(max(raw, 0.0), 1.0)
    document["raw_score"] = raw
    document["gap_closed_percent"] = round(100.0 * raw, 3)

    if raw <= 0.0:
        reason = "no-improvement-over-default"
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
