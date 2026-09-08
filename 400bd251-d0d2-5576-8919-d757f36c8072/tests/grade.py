#!/usr/bin/env python3
"""The grader for OER-28. Checks equivalence, then times three plans against each other.

THE HARD GATE COMES FIRST
    Before a single microsecond is timed, the submitted plan is run forward and
    backward on the SAME frozen weights over the same fixed batches as the
    reference plan, and must agree with it on the loss to within the declared
    absolute tolerance and on the global gradient norm to within the declared
    relative one. A plan that does not agree is refused with
    `numerical-output-diverged` at exactly 0.0, however fast it was. The gate is
    never a soft penalty and never scales the reward: speed bought by computing a
    different function is not speed.

THE REWARD SCALE
    reward = clamp( (control_ms - agent_ms) / (control_ms - reference_ms), 0, 1 )

Both endpoints are MEASURED HERE, on this run:

    control_ms   -- the shipped environment/default_plan.json, timed on this run.
                    The floor. A submission that ties it scores 0.0.
    reference_ms -- tests/private/reference_plan.json, a plan that exists only
                    inside the verifier image. The bar. Reaching it scores 1.0,
                    and beating it also scores 1.0.

Neither number is stored, cached or defaulted. There is no literal anywhere in this
bundle for either of them.

WHY THE TIMING IS TRUSTWORTHY ON A SHARED MACHINE
    The three plans are timed ROUND-ROBIN and each plan's latency is the MINIMUM
    over its rounds. Within a round, a machine that is busy is busy for all three.
    The reward is a ratio of DIFFERENCES of latencies, so a slowdown that
    multiplies all three equally cancels out of it exactly; only a slowdown that
    lands differentially on one plan can move the reward, and the minimum-over-
    rounds estimator is what discards those rounds.

WHAT IS NOT GRADED
    Nothing the submission reports about itself. `notes` is free text and is copied
    into the score document without ever being parsed. No timing the submission
    claims is read.
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

import harness           # noqa: E402  the verifier's OWN copy
import kernel_plan       # noqa: E402  the verifier's OWN copy

SUBMISSION_PATHS = (
    Path(os.environ.get("OER28_SUBMISSION", "/workspace/submission/plan.json")),
    Path("/workspace/plan.json"),
)
DEFAULT_PLAN = HERE / "default_plan.json"
REFERENCE_PLAN = HERE / "private" / "reference_plan.json"
HOLDOUT_SLICE = HERE / "holdout" / "val_slice.bin"

LOGDIR = Path(os.environ.get("OER28_LOGDIR", "/logs/verifier"))


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
        "slot": "OER-28",
        "graded_quantity": "milliseconds per forward-and-backward step of the frozen computation",
        "metric": "step latency, lower is better",
        "metric_direction": "lower is better",
        "reward_formula": "clamp((control_ms - agent_ms) / (control_ms - reference_ms), 0, 1)",
        "anchor_provenance": "both endpoints timed by this verifier on this run",
        "gate": "numerical equivalence against the reference plan, checked before anything is timed",
    }

    submitted = locate_submission()
    if submitted is None:
        return refuse("submission-absent",
                      "no plan at " + " or ".join(str(p) for p in SUBMISSION_PATHS), document)
    document["submission_path"] = str(submitted)
    try:
        agent_doc = kernel_plan.load(submitted)
    except kernel_plan.Refusal as exc:
        return refuse(exc.reason, exc.detail, document)
    agent = kernel_plan.as_execution(agent_doc)
    document["agent_plan"] = agent
    document["agent_notes"] = str(agent_doc.get("notes", ""))[:2000]

    try:
        control = kernel_plan.as_execution(kernel_plan.load(DEFAULT_PLAN))
        reference = kernel_plan.as_execution(kernel_plan.load(REFERENCE_PLAN))
    except kernel_plan.Refusal as exc:
        return refuse("verifier-anchor-plan-invalid", f"{exc.reason}: {exc.detail}", document)
    document["control_plan"] = control
    document["reference_plan"] = reference

    import torch
    if not torch.cuda.is_available():
        return refuse("accelerator-absent", "no CUDA device is visible to the verifier", document)
    document["device"] = torch.cuda.get_device_name(0)

    harness.configure_backends()
    spec = harness.load_spec()
    document["substrate"] = {"architecture": spec["architecture"], "step": spec["step"],
                             "init": spec["init"], "equivalence": spec["equivalence"],
                             "timing": spec["timing"]}

    try:
        tokens = harness.to_device_tokens(harness.load_shard(HOLDOUT_SLICE), "cuda")
        bench = harness.Bench(spec, "cuda")
    except Exception as exc:
        traceback.print_exc()
        return refuse("verifier-substrate-unavailable", f"{type(exc).__name__}: {exc}", document)
    document["parameters"] = bench.parameters
    print(f"[setup] {bench.parameters / 1e6:.1f}M parameters, "
          f"{time.time() - started:.1f}s", flush=True)

    # --- THE HARD GATE, before anything is timed --------------------------
    checks = {}
    for label, plan in (("control", control), ("reference", reference), ("agent", agent)):
        try:
            checks[label] = bench.equivalence(plan, tokens)
        except Exception as exc:
            traceback.print_exc()
            if label == "agent":
                return refuse("agent-plan-failed",
                              f"the submitted plan raised {type(exc).__name__}: {exc}", document)
            return refuse("anchor-plan-failed",
                          f"the {label} anchor raised {type(exc).__name__}: {exc}", document)
        print(f"[equivalence:{label}] loss_delta="
              f"{checks[label]['worst_loss_abs_delta']:.6f} "
              f"grad_rel={checks[label]['worst_grad_relative_delta']:.6f} "
              f"-> {'OK' if checks[label]['equivalent'] else 'DIVERGED'}", flush=True)
    document["equivalence"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "batches"} for k, v in checks.items()}

    for label in ("control", "reference"):
        if not checks[label]["equivalent"]:
            return refuse("verifier-anchor-plan-invalid",
                          f"the {label} anchor is not equivalent to the reference plan on this "
                          f"run, so the scale it would define is not a scale of the same "
                          f"computation", document)

    if not checks["agent"]["equivalent"]:
        detail = (f"the submitted plan disagreed with the reference plan by "
                  f"{checks['agent']['worst_loss_abs_delta']:.6f} nats of loss "
                  f"(tolerance {checks['agent']['loss_tolerance']}) and "
                  f"{checks['agent']['worst_grad_relative_delta']:.6f} relative gradient norm "
                  f"(tolerance {checks['agent']['grad_relative_tolerance']}) on frozen weights. "
                  f"A plan that computes a different function is not a faster plan.")
        if not checks["agent"]["finite"]:
            detail = "the submitted plan produced a non-finite loss or gradient norm"
        return refuse("numerical-output-diverged", detail, document)

    # --- interleaved timing ------------------------------------------------
    try:
        latencies = harness.interleaved_latencies(
            bench, {"control": control, "reference": reference, "agent": agent}, tokens)
    except Exception as exc:
        traceback.print_exc()
        return refuse("timing-failed", f"{type(exc).__name__}: {exc}", document)
    document["latencies"] = latencies
    document["grading_seconds"] = round(time.time() - started, 1)

    control_ms = latencies["control"]["median_ms"]
    reference_ms = latencies["reference"]["median_ms"]
    agent_ms = latencies["agent"]["median_ms"]
    for label, value in (("control", control_ms), ("reference", reference_ms),
                         ("agent", agent_ms)):
        if not math.isfinite(value) or value <= 0.0:
            return refuse("timing-failed",
                          f"the {label} plan produced no usable latency ({value!r})", document)

    span = control_ms - reference_ms
    document["control_ms"] = control_ms
    document["reference_ms"] = reference_ms
    document["agent_ms"] = agent_ms
    document["span_ms"] = span
    document["agent_speedup_over_control"] = control_ms / agent_ms

    if span <= 0.0:
        return refuse("calibration-span-nonpositive",
                      f"the control timed {control_ms:.4f}ms and the reference {reference_ms:.4f}ms, "
                      f"leaving no positive span to scale between. This is a property of this "
                      f"run's measurements, not of the submission.", document)

    raw = (control_ms - agent_ms) / span
    reward = min(max(raw, 0.0), 1.0)
    document["raw_score"] = raw
    document["gap_closed_percent"] = round(100.0 * raw, 3)

    if raw <= 0.0:
        reason = "no-speedup-over-the-shipped-plan"
    elif raw >= 1.0:
        reason = "reference-reached"
    else:
        reason = "partial-speedup"
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
