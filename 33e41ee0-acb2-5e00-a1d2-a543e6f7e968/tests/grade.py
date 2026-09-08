#!/usr/bin/env python3
"""The grader for OER-04. Trains four models and divides two measurements it made itself.

THE REWARD SCALE
    reward = clamp( (control_loss - agent_loss) / (control_loss - reference_loss), 0, 1 )

Both endpoints of that scale are MEASURED HERE, on this run, by real training runs
of this verifier's own harness:

    control_loss   -- the shipped environment/default_shape.json, retrained from
                      scratch. The floor. A submission that ties it scores 0.0.
    reference_loss -- tests/private/reference_shape.json, a shape that exists only
                      inside the verifier image. The bar. A submission that reaches
                      it scores 1.0, and beating it also scores 1.0.

Neither number is stored, cached, defaulted or read from the bundle. There is no
literal anywhere in this file or in this bundle for either of them, and if either
anchor fails to produce a finite loss the run REFUSES with a named reason rather
than substituting anything.

THE FOURTH RUN, AND THE AMBIGUOUS INTERMEDIATE STATE
    The control shape is trained TWICE. The spread between those two runs is this
    run's REPEATABILITY FLOOR: two identical shapes, identical seeds, identical
    data and identical kernels, differing only in the order the accelerator
    happened to reduce in. It is measured on every grading run and is never a
    stored constant.

    When a submission's advantage over the control is smaller than that floor, the
    reward is still exactly what the measurement says -- nothing is rounded, scaled
    or suppressed -- but the run is reported with the reason
    `separation-unproven-at-the-repeatability-floor`. That is this slot's ambiguous
    intermediate state: a result that is neither a demonstrated improvement nor a
    demonstrated failure, recorded as such rather than silently resolved either way.

THE PARAMETER BAND IS COUNTED, NOT COMPUTED
    Whether a shape is inside the band is decided by counting the parameters of the
    INSTANTIATED model. No formula in this file and no number in a submission is
    trusted for it.

WHAT IS NOT GRADED
    Nothing the submission reports about itself. `notes` is free text and is copied
    into the score document for a human reader without ever being parsed.
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
import shape_schema       # noqa: E402  the verifier's OWN copy

SUBMISSION_PATHS = (
    Path(os.environ.get("OER04_SUBMISSION", "/workspace/submission/shape.json")),
    Path("/workspace/shape.json"),
)
DEFAULT_SHAPE = HERE / "default_shape.json"
REFERENCE_SHAPE = HERE / "private" / "reference_shape.json"
TRAIN_SLICE = HERE / "corpus" / "train_slice.bin"
HOLDOUT_SLICE = HERE / "holdout" / "val_slice.bin"

LOGDIR = Path(os.environ.get("OER04_LOGDIR", "/logs/verifier"))


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
        "slot": "OER-04",
        "graded_quantity": "architecture shape at a fixed parameter budget",
        "metric": "validation cross-entropy in nats on the verifier's held-out FineWeb slice",
        "metric_direction": "lower is better",
        "reward_formula": "clamp((control_loss - agent_loss) / (control_loss - reference_loss), 0, 1)",
        "anchor_provenance": "both endpoints trained from scratch by this verifier on this run",
        "repeatability_provenance": "measured on this run by training the control shape twice",
    }

    spec = harness.load_spec()
    bounds = spec["shape_bounds"]
    band = spec["parameter_band"]
    document["shape_bounds"] = bounds
    document["parameter_band"] = {"min": band["min"], "max": band["max"]}

    submitted = locate_submission()
    if submitted is None:
        return refuse("submission-absent",
                      "no shape at " + " or ".join(str(p) for p in SUBMISSION_PATHS), document)
    document["submission_path"] = str(submitted)
    try:
        agent_shape = shape_schema.load(submitted, bounds)
    except shape_schema.Refusal as exc:
        return refuse(exc.reason, exc.detail, document)
    document["agent_shape"] = {k: v for k, v in agent_shape.items() if k != "notes"}
    document["agent_notes"] = str(agent_shape.get("notes", ""))[:2000]

    try:
        default_shape = shape_schema.load(DEFAULT_SHAPE, bounds)
        reference_shape = shape_schema.load(REFERENCE_SHAPE, bounds)
    except shape_schema.Refusal as exc:
        return refuse("verifier-anchor-invalid", f"{exc.reason}: {exc.detail}", document)

    import torch
    if not torch.cuda.is_available():
        return refuse("accelerator-absent", "no CUDA device is visible to the verifier", document)
    document["device"] = torch.cuda.get_device_name(0)

    harness.configure_backends()
    document["substrate"] = {"computation": spec["computation"], "budget": spec["budget"],
                             "evaluation": spec["evaluation"], "init": spec["init"],
                             "optimizer": spec["optimizer"], "schedule": spec["schedule"]}

    # --- the band is decided by counting an instantiated model, on CPU, before
    # --- any training is paid for. ----------------------------------------
    counts = {}
    for label, shape in (("control", default_shape), ("reference", reference_shape),
                         ("agent", agent_shape)):
        try:
            probe, count = harness.build_model(shape, spec, device="cpu")
        except Exception as exc:
            traceback.print_exc()
            if label == "agent":
                return refuse("shape-not-instantiable",
                              f"the submitted shape could not be built: {type(exc).__name__}: {exc}",
                              document)
            return refuse("verifier-anchor-invalid",
                          f"the {label} anchor shape could not be built: {exc}", document)
        del probe
        counts[label] = count
        print(f"[shape:{label}] {shape['num_layers']}L x {shape['model_dim']}d "
              f"head_dim={shape['head_dim']} mlp={shape['mlp_ratio']} -> "
              f"{count} parameters", flush=True)
    document["parameter_counts"] = counts

    for label in ("control", "reference"):
        try:
            shape_schema.parameter_band_check(counts[label], band)
        except shape_schema.Refusal as exc:
            return refuse("verifier-anchor-invalid",
                          f"the {label} anchor is outside its own band: {exc.detail}", document)
    try:
        shape_schema.parameter_band_check(counts["agent"], band)
    except shape_schema.Refusal as exc:
        return refuse(exc.reason, exc.detail, document)

    try:
        train_tokens = harness.to_device_tokens(harness.load_shard(TRAIN_SLICE), "cuda")
        holdout_tokens = harness.to_device_tokens(harness.load_shard(HOLDOUT_SLICE), "cuda")
    except Exception as exc:
        traceback.print_exc()
        return refuse("verifier-substrate-unavailable", f"{type(exc).__name__}: {exc}", document)

    # --- four runs. Anchors first, so a crash in the submission's run still
    # --- leaves both measured endpoints in the score document. -------------
    plan = (("control", default_shape), ("control_repeat", default_shape),
            ("reference", reference_shape), ("agent", agent_shape))
    runs = {}
    for label, shape in plan:
        print(f"\n[train:{label}] {shape['num_layers']}L x {shape['model_dim']}d, "
              f"{spec['budget']['micro_steps']} optimizer steps", flush=True)
        run_started = time.time()
        runner = None
        try:
            runner = harness.ShapeRunner(shape, spec, "cuda")
            report = runner.run(train_tokens, holdout_tokens, progress=768)
        except Exception as exc:
            traceback.print_exc()
            if label == "agent":
                return refuse("agent-run-failed",
                              f"the submitted shape raised {type(exc).__name__}: {exc}",
                              {**document, "runs": runs})
            return refuse("anchor-run-failed",
                          f"the {label} run raised {type(exc).__name__}: {exc}",
                          {**document, "runs": runs})
        finally:
            if runner is not None:
                runner.release()
        report["wall_seconds"] = round(time.time() - run_started, 1)
        report.pop("final_micro_loss", None)
        runs[label] = report
        print(f"[train:{label}] val_loss={report['val_loss']:.6f} "
              f"({report['wall_seconds']}s)", flush=True)
    document["runs"] = runs
    document["grading_seconds"] = round(time.time() - started, 1)

    control_loss = runs["control"]["val_loss"]
    repeat_loss = runs["control_repeat"]["val_loss"]
    reference_loss = runs["reference"]["val_loss"]
    agent_loss = runs["agent"]["val_loss"]

    for label in ("control", "control_repeat", "reference"):
        if not math.isfinite(runs[label]["val_loss"]):
            return refuse("anchor-diverged",
                          f"the {label} run produced no finite loss, so the reward scale "
                          f"has no {label} endpoint on this run; refusing rather than "
                          f"substituting a stored value", document)

    repeatability_floor = abs(control_loss - repeat_loss)
    span = control_loss - reference_loss
    document["control_loss"] = control_loss
    document["control_repeat_loss"] = repeat_loss
    document["repeatability_floor"] = repeatability_floor
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
                      "the submitted shape produced no finite validation loss", document)

    raw = (control_loss - agent_loss) / span
    reward = min(max(raw, 0.0), 1.0)
    advantage = control_loss - agent_loss
    document["raw_score"] = raw
    document["gap_closed_percent"] = round(100.0 * raw, 3)
    document["advantage_over_control"] = advantage
    document["separation_proven"] = bool(abs(advantage) > repeatability_floor)

    if abs(advantage) <= repeatability_floor:
        reason = "separation-unproven-at-the-repeatability-floor"
        document["ambiguous_intermediate_state"] = (
            f"the submitted shape measured {advantage:+.6f} nats against the control, and "
            f"the control measured {repeatability_floor:.6f} nats against itself on this "
            f"same run. The submission is therefore neither a demonstrated improvement nor "
            f"a demonstrated failure at this budget. The reward below is exactly what the "
            f"measurement says; nothing has been rounded or suppressed."
        )
    elif raw <= 0.0:
        reason = "no-improvement-over-control"
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
