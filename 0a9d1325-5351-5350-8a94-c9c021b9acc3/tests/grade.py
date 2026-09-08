#!/usr/bin/env python3
"""The grader for OER-09. Trains three models and divides two measurements it made itself.

THE REWARD SCALE
    reward = clamp( (control_loss - agent_loss) / (control_loss - reference_loss), 0, 1 )

Both endpoints of that scale are MEASURED HERE, on this run, by real training runs
of this verifier's own harness:

    control_loss   -- the shipped environment/default_filter.json, retrained from
                      scratch. Every predicate in it names a field the register
                      does not carry, so it resolves nothing, admits the pool
                      unchanged, and trains on the first 3072 blocks INCLUDING the
                      degraded ones. That is the uncurated floor. A submission that
                      ties it scores 0.0.
    reference_loss -- tests/private/reference_filter.json, a chain that exists only
                      inside the verifier image. The bar. A submission that reaches
                      it scores 1.0, and beating it also scores 1.0.

Neither number is stored, cached, defaulted or read from the bundle. There is no
literal anywhere in this file or in this bundle for either of them, and if either
anchor fails to produce a finite loss the run REFUSES with a named reason rather
than substituting anything. The cost of that discipline is that grading trains
three models instead of one; the model was sized so that fits the envelope.

WHAT IS NOT GRADED
    Nothing the submission reports about itself. The submission is a filter chain
    -- a declaration of predicates -- and the only thing read out of it is which
    blocks it admits. It carries no loss, no metric and no claim that this file
    looks at. `notes` is free text and is copied into the score document for a
    human reader without ever being parsed. The runner's own report
    (`rules_unresolved` and the rest) is recorded for the reader and never scaled
    into the reward: a chain that resolved nothing is not penalised by a rule, it
    simply trains on the uncurated pool and therefore measures like the control.

WHY THE THREE RUNS ARE COMPARABLE
    harness.Runner builds ONE network under the frozen seed and restores those exact
    initial tensors before each run, and one compiled graph serves all three. The
    frozen optimizer and schedule come from the frozen spec. The anchors and the
    submission differ by WHICH BLOCKS they consumed and by nothing else.
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
import filter_schema      # noqa: E402  the verifier's OWN copy

SUBMISSION_PATHS = (
    Path(os.environ.get("OER09_SUBMISSION", "/workspace/submission/filter.json")),
    Path("/workspace/filter.json"),
)
DEFAULT_FILTER = HERE / "default_filter.json"
REFERENCE_FILTER = HERE / "private" / "reference_filter.json"
REGISTER = HERE / "pool" / "source_register.jsonl"
POOL_BLOCKS = HERE / "pool" / "pool_blocks.bin"
HOLDOUT_SLICE = HERE / "holdout" / "val_slice.bin"

LOGDIR = Path(os.environ.get("OER09_LOGDIR", "/logs/verifier"))


def emit(reward: float, reason: str, document: dict) -> None:
    """Write every carrier. reward.txt is the bare float, reward.json the pair."""
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
        "slot": "OER-09",
        "graded_quantity": "curation filter chain over the source pool",
        "metric": "validation cross-entropy in nats on the verifier's held-out FineWeb slice",
        "metric_direction": "lower is better",
        "reward_formula": "clamp((control_loss - agent_loss) / (control_loss - reference_loss), 0, 1)",
        "anchor_provenance": "both endpoints trained from scratch by this verifier on this run",
    }

    # --- hygiene gate. Refuses, never scales. ------------------------------
    submitted = locate_submission()
    if submitted is None:
        return refuse("submission-absent",
                      "no filter chain at " + " or ".join(str(p) for p in SUBMISSION_PATHS),
                      document)
    document["submission_path"] = str(submitted)
    try:
        agent_doc = filter_schema.load(submitted)
    except filter_schema.Refusal as exc:
        return refuse(exc.reason, exc.detail, document)
    document["agent_filter"] = {k: v for k, v in agent_doc.items() if k != "notes"}
    document["agent_notes"] = str(agent_doc.get("notes", ""))[:2000]

    try:
        default_doc = filter_schema.load(DEFAULT_FILTER)
        reference_doc = filter_schema.load(REFERENCE_FILTER)
    except filter_schema.Refusal as exc:
        return refuse("verifier-anchor-invalid", f"{exc.reason}: {exc.detail}", document)

    # --- selection, before any accelerator work ----------------------------
    register = filter_schema.load_register(REGISTER)
    document["pool_blocks"] = len(register)
    selections = {}
    for label, doc in (("control", default_doc), ("reference", reference_doc), ("agent", agent_doc)):
        admitted, report = filter_schema.apply(doc, register)
        selections[label] = admitted
        document.setdefault("filter_reports", {})[label] = report
        print(f"[select:{label}] admitted {report['admitted']}/{report['pool_blocks']} "
              f"(rules resolved {report['rules_resolved']}/{report['rules_total']}, "
              f"unresolved {report['rules_unresolved']})", flush=True)

    spec = harness.load_spec()
    need_blocks = int(spec["budget"]["blocks"])
    document["blocks_required"] = need_blocks

    for label in ("control", "reference"):
        if len(selections[label]) < need_blocks:
            return refuse("verifier-anchor-underfilled",
                          f"the {label} anchor admitted {len(selections[label])} blocks, "
                          f"the budget consumes {need_blocks}", document)
    if len(selections["agent"]) < need_blocks:
        return refuse("pool-underfilled",
                      f"the submitted chain admitted {len(selections['agent'])} blocks, "
                      f"the budget consumes {need_blocks}. A chain must leave at least "
                      f"the budget standing; this one filtered the pool below it.",
                      document)

    import torch
    if not torch.cuda.is_available():
        return refuse("accelerator-absent", "no CUDA device is visible to the verifier", document)
    document["device"] = torch.cuda.get_device_name(0)

    harness.configure_backends()
    document["substrate"] = {"architecture": spec["architecture"], "budget": spec["budget"],
                             "evaluation": spec["evaluation"], "init": spec["init"],
                             "optimizer": spec["optimizer"], "schedule": spec["schedule"]}

    try:
        pool = harness.BlockPool(POOL_BLOCKS, spec, "cuda")
        holdout_tokens = harness.to_device_tokens(harness.load_shard(HOLDOUT_SLICE), "cuda")
        runner = harness.Runner(pool, "cuda", spec)
    except Exception as exc:
        traceback.print_exc()
        return refuse("verifier-substrate-unavailable", f"{type(exc).__name__}: {exc}", document)
    document["parameters"] = sum(p.numel() for p in runner.model.parameters())
    print(f"[setup] {document['parameters'] / 1e6:.1f}M parameters, "
          f"{time.time() - started:.1f}s", flush=True)

    # --- the three runs. Anchors first, so a crash in the submission's run
    # --- still leaves both measured endpoints in the score document. -------
    runs = {}
    for label in ("control", "reference", "agent"):
        used = selections[label][:need_blocks]
        print(f"\n[train:{label}] {need_blocks} blocks, {need_blocks} optimizer steps", flush=True)
        run_started = time.time()
        try:
            report = runner.run(used, holdout_tokens, progress=768)
        except Exception as exc:
            traceback.print_exc()
            if label == "agent":
                return refuse("agent-run-failed",
                              f"the submitted chain raised {type(exc).__name__}: {exc}",
                              {**document, "runs": runs})
            return refuse("anchor-run-failed",
                          f"the {label} anchor raised {type(exc).__name__}: {exc}",
                          {**document, "runs": runs})
        report["wall_seconds"] = round(time.time() - run_started, 1)
        report.pop("final_micro_loss", None)
        runs[label] = report
        print(f"[train:{label}] val_loss={report['val_loss']:.6f} "
              f"({report['wall_seconds']}s)", flush=True)
    document["runs"] = runs
    document["grading_seconds"] = round(time.time() - started, 1)

    control_loss = runs["control"]["val_loss"]
    reference_loss = runs["reference"]["val_loss"]
    agent_loss = runs["agent"]["val_loss"]

    for label in ("control", "reference"):
        if not math.isfinite(runs[label]["val_loss"]):
            return refuse("anchor-diverged",
                          f"the {label} anchor produced no finite loss, so the reward scale "
                          f"has no {label} endpoint on this run; refusing rather than "
                          f"substituting a stored value", document)

    span = control_loss - reference_loss
    document["control_loss"] = control_loss
    document["reference_loss"] = reference_loss
    document["agent_loss"] = agent_loss if math.isfinite(agent_loss) else None
    document["span"] = span

    if span <= 0.0:
        return refuse("calibration-span-nonpositive",
                      f"the control anchor measured {control_loss!r} and the reference anchor "
                      f"measured {reference_loss!r}, leaving no positive span to scale between. "
                      f"This is a property of this run's measurements, not of the submission.",
                      document)

    if not math.isfinite(agent_loss):
        return refuse("agent-diverged",
                      "the submitted chain produced no finite validation loss", document)

    raw = (control_loss - agent_loss) / span
    reward = min(max(raw, 0.0), 1.0)
    document["raw_score"] = raw
    document["gap_closed_percent"] = round(100.0 * raw, 3)

    if raw <= 0.0:
        reason = "no-improvement-over-uncurated-control"
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
