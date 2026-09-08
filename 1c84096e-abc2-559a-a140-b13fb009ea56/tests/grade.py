#!/usr/bin/env python3
"""The grader. Replays three evaluation policies over one training run it performed
itself, and divides two measurements it made itself.

THE REWARD SCALE
    reward = clamp( (default_loss - agent_loss) / (default_loss - reference_loss), 0, 1 )

where each `*_loss` is the HELD-OUT cross-entropy at the checkpoint that policy
selected. Both endpoints of that scale are MEASURED HERE, on this run:

    default_loss   -- the shipped tests/default_policy.json, replayed. It spends the
                      whole evaluation budget on the final checkpoint, which is what
                      a training script does when nobody chose a policy. The floor.
    reference_loss -- tests/private/reference_policy.json, a stronger policy that
                      exists only inside the verifier image. The bar. A submission
                      that reaches it scores 1.0, and beating it also scores 1.0.

Neither number is stored, cached, defaulted or read from the bundle. There is no
literal anywhere in this file or in this bundle for either of them. If either anchor
fails to produce a finite loss the run REFUSES with a named reason rather than
substituting anything.

WHY ONE TRAINING RUN SERVES ALL THREE
    The training run is FROZEN -- frozen/train_recipe.json, frozen/task_spec.json,
    the shipped token stream, one initialisation seed. A policy does not change what
    is trained, only where it looks. So the harness trains once, records the
    per-batch probe losses and the held-out loss at every candidate step, and every
    policy is then answered out of those same recorded numbers. Two policies differ
    by their choices and by nothing else, and no policy can make grading cost more
    than any other.

WHAT IS NOT GRADED
    Nothing the submission reports about itself. The policy declares where to look
    and how to choose; it carries no loss, no metric and no claim that this file
    reads. `notes` is free text copied into the score document without being parsed.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harness            # noqa: E402  the verifier's OWN copy
import policy_schema      # noqa: E402  the verifier's OWN copy

SUBMISSION_PATHS = (
    Path(os.environ.get("OER12_SUBMISSION", "/workspace/submission/policy.json")),
    Path("/workspace/policy.json"),
)
DEFAULT_POLICY = HERE / "default_policy.json"
REFERENCE_POLICY = HERE / "private" / "reference_policy.json"
TRAIN_STREAM = HERE / "corpus" / "train_stream.bin"
PROBE_SLICE = HERE / "probe" / "probe_slice.bin"
HOLDOUT_SLICE = HERE / "holdout" / "val_slice.bin"

LOGDIR = Path(os.environ.get("OER12_LOGDIR", "/logs/verifier"))


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
        "slot": os.environ.get("OER12_SLOT", "OER-12"),
        "metric": "held-out cross-entropy in nats at the checkpoint the policy selected",
        "metric_direction": "lower is better",
        "reward_formula": "clamp((default_loss - agent_loss) / (default_loss - reference_loss), 0, 1)",
        "anchor_provenance": "both endpoints replayed here, on this run, over curves this verifier measured",
    }

    # --- hygiene gate. Refuses, never scales. ------------------------------
    submitted = locate_submission()
    if submitted is None:
        return refuse("submission-absent",
                      "no policy at " + " or ".join(str(p) for p in SUBMISSION_PATHS), document)
    document["submission_path"] = str(submitted)
    try:
        agent_policy = policy_schema.load(submitted)
    except policy_schema.Refusal as exc:
        return refuse(exc.reason, exc.detail, document)
    document["agent_policy"] = agent_policy
    document["agent_notes"] = str(agent_policy.get("notes", ""))[:2000]

    try:
        default_policy = policy_schema.load(DEFAULT_POLICY)
        reference_policy = policy_schema.load(REFERENCE_POLICY)
    except policy_schema.Refusal as exc:
        return refuse("verifier-anchor-policy-invalid", f"{exc.reason}: {exc.detail}", document)

    import torch
    if not torch.cuda.is_available():
        return refuse("accelerator-absent", "no CUDA device is visible to the verifier", document)
    document["device"] = torch.cuda.get_device_name(0)

    harness.configure_backends()
    spec = harness.load_spec()
    recipe = harness.load_train_recipe()
    grid = harness.candidate_steps(spec)
    ev = spec["evaluation"]

    # The schema and the substrate must agree on the grid, or a policy validated on
    # the agent surface could name a step this run never records.
    if tuple(grid) != policy_schema.candidate_steps():
        return refuse("verifier-substrate-unavailable",
                      "frozen/task_spec.json and policy_schema.py disagree on the candidate grid",
                      document)
    document["substrate"] = {"architecture": spec["architecture"], "budget": spec["budget"],
                             "evaluation": ev, "init": spec["init"],
                             "corpus": spec["corpus"]}

    # --- THE DRAW -----------------------------------------------------------
    # The degradation onset is chosen HERE, on this run, from the choice set the
    # substrate declares, with a fresh permutation seed from the system entropy
    # source. It is not seeded from anything on the agent surface and it is not
    # reproducible from the bundle: that is the property that makes a policy a
    # procedure rather than a lookup. The same drawn stream trains the one run that
    # every one of the three policies is then replayed over, so the draw cannot
    # advantage or disadvantage any of them relative to the others.
    try:
        clean = harness.load_shard(TRAIN_STREAM)
        draw = harness.draw_degradation(spec, random.SystemRandom())
        tokens_per_step = int(spec["budget"]["micro_batch"]) * int(spec["architecture"]["seq_len"])
        dirty, corrupted = harness.degrade(
            clean, draw["ramp_start"], draw["ramp_end"], draw["block"],
            tokens_per_step, draw["seed"])
        train_tokens = harness.to_device_tokens(dirty, "cuda")
        probe_tokens = harness.to_device_tokens(harness.load_shard(PROBE_SLICE), "cuda")
        holdout_tokens = harness.to_device_tokens(harness.load_shard(HOLDOUT_SLICE), "cuda")
        runner = harness.Runner("cuda", spec)
    except Exception as exc:
        traceback.print_exc()
        return refuse("verifier-substrate-unavailable", f"{type(exc).__name__}: {exc}", document)
    document["degradation_draw"] = {**draw, "blocks_permuted": corrupted}
    print(f"[draw] ramp_start={draw['ramp_start']} ramp_end={draw['ramp_end']} "
          f"seed={draw['seed']} -> {corrupted} blocks permuted", flush=True)
    document["parameters"] = sum(p.numel() for p in runner.model.parameters())
    print(f"[setup] {document['parameters'] / 1e6:.1f}M parameters, "
          f"{time.time() - started:.1f}s", flush=True)

    # --- the ONE training run, instrumented ---------------------------------
    print(f"\n[train] one frozen run of {spec['budget']['micro_steps']} steps, "
          f"instrumented at {len(grid)} candidate steps", flush=True)
    try:
        swept = runner.sweep(
            recipe, train_tokens,
            {"probe": (probe_tokens, int(ev["probe_batches"])),
             "holdout": (holdout_tokens, int(ev["holdout_batches"]))},
            grid, progress=None)
    except Exception as exc:
        traceback.print_exc()
        return refuse("training-run-failed",
                      f"the frozen training run raised {type(exc).__name__}: {exc}", document)
    curves = swept["curves"]
    document["training_report"] = swept["report"]

    # --- replay the three policies over those curves ------------------------
    tpb = int(ev["tokens_per_batch"])
    outcomes = {}
    for label, policy in (("default", default_policy),
                          ("reference", reference_policy),
                          ("agent", agent_policy)):
        try:
            applied = harness.apply_policy(policy, curves, tpb, stream="probe")
        except Exception as exc:
            traceback.print_exc()
            if label == "agent":
                return refuse("agent-policy-failed",
                              f"the submitted policy raised {type(exc).__name__}: {exc}",
                              {**document, "outcomes": outcomes})
            return refuse("anchor-policy-failed",
                          f"the {label} anchor raised {type(exc).__name__}: {exc}",
                          {**document, "outcomes": outcomes})
        step = applied["selected_step"]
        applied["holdout_loss"] = float(
            sum(curves[step]["holdout"]) / len(curves[step]["holdout"]))
        outcomes[label] = applied
        print(f"[policy:{label}] selected step {step}  "
              f"tokens_spent {applied['tokens_spent']}  "
              f"held-out {applied['holdout_loss']:.6f}", flush=True)
    document["outcomes"] = outcomes
    document["grading_seconds"] = round(time.time() - started, 1)

    default_loss = outcomes["default"]["holdout_loss"]
    reference_loss = outcomes["reference"]["holdout_loss"]
    agent_loss = outcomes["agent"]["holdout_loss"]

    for label in ("default", "reference"):
        if not math.isfinite(outcomes[label]["holdout_loss"]):
            return refuse("anchor-diverged",
                          f"the {label} anchor produced no finite held-out loss, so the "
                          f"reward scale has no {label} endpoint on this run; refusing "
                          f"rather than substituting a stored value", document)

    span = default_loss - reference_loss
    document["default_loss"] = default_loss
    document["reference_loss"] = reference_loss
    document["agent_loss"] = agent_loss if math.isfinite(agent_loss) else None
    document["span"] = span

    if span <= 0.0:
        return refuse("calibration-span-nonpositive",
                      f"the default anchor selected a checkpoint measuring {default_loss!r} "
                      f"and the reference anchor one measuring {reference_loss!r}, leaving no "
                      f"positive span to scale between. This is a property of this run's "
                      f"measurements, not of the submission.", document)

    if not math.isfinite(agent_loss):
        return refuse("agent-policy-diverged",
                      "the submitted policy selected a checkpoint with no finite held-out loss",
                      document)

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
