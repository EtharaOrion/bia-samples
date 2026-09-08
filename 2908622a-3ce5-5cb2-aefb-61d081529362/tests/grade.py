#!/usr/bin/env python3
"""The grader for OER-10. Trains three models and divides two measurements it made itself.

THE REWARD SCALE
    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

Both endpoints of that scale are MEASURED HERE, on this run, by real training runs of
this verifier's own harness:

    default_loss   -- the shipped environment/default_mixture.json, retrained from
                      scratch. The floor. A submission that ties it scores 0.0.
    reference_loss -- tests/private/reference_mixture.json, a stronger data mixture that
                      exists only inside the verifier image. The bar. A submission that
                      reaches it scores 1.0, and beating it also scores 1.0.

Neither number is stored, cached, defaulted or read from the bundle. There is no
literal anywhere in this file or in this bundle for either of them, and if either
anchor fails to produce a finite loss the run REFUSES with a named reason rather than
substituting anything. The cost of that discipline is that grading trains three models
instead of one; the model was sized so that fits the envelope.

WHAT IS NOT GRADED
    Nothing the submission reports about itself. The submission is a declaration, and
    the only thing read out of it is the declared values. It carries no loss, no metric
    and no claim that this file looks at. `notes` is free text and is copied into the
    score document for a human reader without ever being parsed.

WHY THE THREE RUNS ARE COMPARABLE
    harness.Runner builds ONE network under the frozen seed and one compiled graph
    serves all three runs. The anchors and the submission differ by their data mixture
    and by nothing else.
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
import mixture_schema as schema  # noqa: E402  the verifier's OWN copy

SUBMISSION_PATHS = (
    Path(os.environ.get("OER10_SUBMISSION", "/workspace/submission/mixture.json")),
    Path("/workspace/mixture.json"),
)
DEFAULT_DOC = HERE / "default_mixture.json"
REFERENCE_DOC = HERE / "private" / "reference_mixture.json"
HOLDOUT_SLICE = HERE / "holdout" / "val_slice.bin"
CORPUS_DIR = HERE / "corpus"

LOGDIR = Path(os.environ.get("OER10_LOGDIR", "/logs/verifier"))


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
        "slot": "OER-10",
        "graded_quantity": "data mixture",
        "metric": "validation cross-entropy in nats on the verifier's held-out FineWeb slice",
        "metric_direction": "lower is better",
        "reward_formula": "clamp((default_loss - agent_loss) / (default_loss - reference_loss), 0, 1)",
        "anchor_provenance": "both endpoints trained from scratch by this verifier on this run",
    }

    # --- hygiene gate. Refuses, never scales. ------------------------------
    submitted = locate_submission()
    if submitted is None:
        return refuse("submission-absent",
                      "no data mixture at " + " or ".join(str(p) for p in SUBMISSION_PATHS), document)
    document["submission_path"] = str(submitted)
    try:
        agent_doc = schema.load(submitted)
    except schema.Refusal as exc:
        return refuse(exc.reason, exc.detail, document)
    document["agent_submission"] = {k: v for k, v in agent_doc.items() if k != "notes"}
    document["agent_notes"] = str(agent_doc.get("notes", ""))[:2000]

    # --- the verifier's own inputs ----------------------------------------
    try:
        default_doc = schema.load(DEFAULT_DOC)
        reference_doc = schema.load(REFERENCE_DOC)
    except schema.Refusal as exc:
        return refuse("verifier-anchor-invalid", f"{exc.reason}: {exc.detail}", document)

    import torch
    if not torch.cuda.is_available():
        return refuse("accelerator-absent",
                      "no CUDA device is visible to the verifier", document)
    document["device"] = torch.cuda.get_device_name(0)

    harness.configure_backends()
    spec = harness.load_spec()
    document["substrate"] = {"architecture": spec["architecture"], "budget": spec["budget"],
                             "evaluation": spec["evaluation"], "init": spec["init"]}

    try:
        pool = harness.ShardPool(CORPUS_DIR, spec, "cuda")
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
    for label, doc in (("default", default_doc),
                       ("reference", reference_doc),
                       ("agent", agent_doc)):
        print(f"\n[train:{label}] {runner.steps_for(doc)} optimizer steps", flush=True)
        run_started = time.time()
        try:
            report = runner.run(doc, holdout_tokens, progress=768)
        except Exception as exc:
            traceback.print_exc()
            if label == "agent":
                return refuse("agent-run-failed",
                              f"the submitted data mixture raised {type(exc).__name__}: {exc}",
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

    default_loss = runs["default"]["val_loss"]
    reference_loss = runs["reference"]["val_loss"]
    agent_loss = runs["agent"]["val_loss"]

    # --- anchors must have computed. No substitution. ----------------------
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
                      f"scale between. This is a property of this run's measurements, not "
                      f"of the submission.", document)

    if not math.isfinite(agent_loss):
        return refuse("agent-diverged",
                      "the submitted data mixture produced no finite validation loss", document)

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
