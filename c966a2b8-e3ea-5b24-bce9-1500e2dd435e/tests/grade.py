"""Verifier entry point for the S10 reproduction-under-ablation slot.

What this program does, in order, and why the order is part of the grade.

1. Recompute the substrate hash manifest from disk and compare it with the
   manifest pinned in the image.
2. Resolve the two anchors. They are measured, never authored. The target
   anchor is the frozen reference recipe on the un-ablated model, which is the
   published result reproduced at the scaled operating point. The baseline
   anchor is the identical recipe on the ablated model, which is what a naive
   port of the published recipe produces. Both are cached by a key over the
   substrate manifest, the frozen config, the corpus and the device, so a
   session pays for them once rather than fifty times.
3. Run the agent arm: the ablated model driven by the submitted update rule.
4. Recompute the agent arm's validation loss independently from its saved
   checkpoint, in a fresh object graph.
5. Recompute the substrate hash manifest again.
6. Compute raw and the provisional score from the three measured numbers.
7. Run every checker over the assembled evidence. Every checker is hard pass:
   one failure gates the final score to exactly zero and the emitted reason is
   that checker's own machine-readable reason.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import checkers as CHECKERS  # noqa: E402

BUNDLE = os.environ.get("BIA_BUNDLE", os.path.dirname(HERE))
SUBSTRATE_DIR = os.environ.get("BIA_SUBSTRATE_DIR", os.path.join(BUNDLE, "environment", "substrate"))
SUBMISSION_DIR = os.environ.get("BIA_SUBMISSION_DIR", "/workspace/submission")
WORKDIR = os.environ.get("BIA_WORKDIR", "/workspace/artifacts")
LOGDIR = os.environ.get("BIA_LOGDIR", "/logs/verifier")
ANCHOR_DIR = os.environ.get("BIA_ANCHOR_DIR", "/anchors")
CORPUS = os.environ.get("BIA_CORPUS", "/opt/bia/data/enwik8")
SMOKE = os.environ.get("BIA_SMOKE", "0") == "1"
SCORE_PATH = os.environ.get("SCORE_PATH", os.path.join(LOGDIR, "score.json"))
OUTCOMES_PATH = os.environ.get("BIA_OUTCOMES", os.path.join(LOGDIR, "outcomes.json"))
RUN_RECORD = os.environ.get("BIA_RUN_RECORD", os.path.join(LOGDIR, "run_record.jsonl"))
EVIDENCE_PATH = os.environ.get("BIA_EVIDENCE", os.path.join(LOGDIR, "evidence.json"))
ARM_WORKER = os.path.join(HERE, "arm_worker.py")
ARM_TIMEOUT = float(os.environ.get("BIA_ARM_TIMEOUT", "3600"))

# The anchor cache survives across the fifty attempts of a session and lives in the agent
# image, so it is a persistent trust store the graded party can write to. It is now
# authenticated: every entry carries an HMAC taken with a key that lives in the verifier tree,
# which Harbor mounts only for the post-agent verifier and which no attempt can read. An entry
# whose tag does not verify is not a cache miss to be worked around, it is a graded failure,
# because a persistent store somebody forged once and harvested fifty times is worse than no
# cache at all.
ANCHOR_KEY_PATH = os.path.join(HERE, "checkers_anchor_key.bin")


def anchor_key():
    with open(ANCHOR_KEY_PATH, "rb") as f:
        return f.read()


def anchor_tag(payload_bytes):
    return hmac.new(anchor_key(), payload_bytes, hashlib.sha256).hexdigest()


def anchor_payload_bytes(entry):
    body = {k: v for k, v in entry.items() if k != "hmac_sha256"}
    return json.dumps(body, sort_keys=True, default=str).encode()

sys.path.insert(0, os.path.dirname(SUBSTRATE_DIR))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def substrate_manifest(directory):
    out = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".py"):
            continue
        out[name] = sha256_file(os.path.join(directory, name))
    return out


def pinned_manifest(directory):
    path = os.path.join(directory, "SUBSTRATE_MANIFEST.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))["files"]


def emit(score, reason, detail, outcomes, evidence):
    for p in (SCORE_PATH, OUTCOMES_PATH, EVIDENCE_PATH):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    payload = {"score": float(score), "reason": reason, "detail": detail or {}}
    json.dump(payload, open(SCORE_PATH, "w"), indent=1, sort_keys=True)
    json.dump(outcomes or {}, open(OUTCOMES_PATH, "w"), indent=1, sort_keys=True)
    json.dump(evidence or {}, open(EVIDENCE_PATH, "w"), indent=1, sort_keys=True, default=str)
    print(json.dumps(payload, sort_keys=True))
    return payload


def run_agent_arm_out_of_process(submission_path, cfg_smoke, device, workdir, run_record):
    """Run the agent arm in an interpreter this verifier does not share with the submission.

    The verifier never imports update_rule.py. Before this, load_submission exec_module'd it
    into the verifier process, and it did so BEFORE the anchor cache was read and BEFORE the
    checker registry was called, so both were reachable by the graded party. They are now on
    the far side of a process boundary that no submitted byte crosses.
    """
    tag = secrets.token_hex(8)
    result_path = os.path.join(LOGDIR, "arm_result_%s.json" % tag)
    worker_record = os.path.join(LOGDIR, "arm_record_%s.jsonl" % tag)
    req = {
        "submission_path": submission_path,
        "substrate_parent": os.path.dirname(SUBSTRATE_DIR),
        "smoke": "1" if cfg_smoke else "0",
        "corpus": CORPUS,
        "workdir": workdir,
        "device": device,
        "run_record": worker_record,
        "result_path": result_path,
    }
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", ARM_WORKER],
            input=json.dumps(req),
            capture_output=True,
            text=True,
            env=env,
            timeout=ARM_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None, "agent_arm_timed_out", None, None
    if not os.path.exists(result_path):
        return None, "agent_arm_produced_no_result", None, None
    with open(result_path) as f:
        res = json.load(f)
    if not res.get("ok"):
        return None, str(res.get("error", "agent_arm_failed")), res.get("worker_pid")
    return res["arm"], None, res.get("worker_pid"), worker_record


def weight_displacement_from_checkpoint(ckpt_path, cfg, device):
    """L2 distance between the checkpoint the worker returned and the frozen initialization.

    This is the verifier's OWN measurement of whether the submitted rule moved anything. The
    per-step delta_l2 events the substrate emits are produced beside the submission and are
    therefore claims; this number is produced here, from bytes, and cannot be authored by the
    graded party without actually moving the weights.
    """
    import torch

    from substrate.model import GPT

    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    trained = GPT(blob["cfg"], blob["norm_mode"]).to(device)
    trained.load_state_dict(blob["state_dict"])
    torch.manual_seed(int(blob["cfg"]["seed"]))
    fresh = GPT(blob["cfg"], blob["norm_mode"]).to(device)
    total = 0.0
    for (na, a), (nb, b) in zip(sorted(trained.state_dict().items()), sorted(fresh.state_dict().items())):
        if na != nb or not hasattr(a, "float"):
            continue
        total += float(((a.detach().float() - b.detach().float()) ** 2).sum())
    return total ** 0.5


def anchor_cache_key(manifest, cfg_digest, corpus_digest, device_name, recipe_digest):
    h = hashlib.sha256()
    h.update(json.dumps(manifest, sort_keys=True).encode())
    h.update(cfg_digest.encode())
    h.update(corpus_digest.encode())
    h.update(device_name.encode())
    h.update(recipe_digest.encode())
    return h.hexdigest()


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(WORKDIR, exist_ok=True)

    import torch  # imported here so a missing substrate is reported before torch cost

    from substrate import config as C
    from substrate import reference_recipe
    from substrate.data import FrozenByteCorpus
    from substrate.runlog import RunLog, read_records
    from substrate.trainer import independent_eval_from_checkpoint, run_arm

    log = RunLog(RUN_RECORD)

    if SMOKE:
        torch.set_num_threads(int(os.environ.get("BIA_SMOKE_THREADS", "2")))
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"

    cfg = C.get_config(SMOKE)
    cfg_digest = C.config_digest(cfg)

    pinned = pinned_manifest(SUBSTRATE_DIR)
    start_manifest = substrate_manifest(SUBSTRATE_DIR)
    log.emit("verifier_start", profile=cfg["profile"], device=device, device_name=device_name,
             config_digest=cfg_digest, substrate_files=len(start_manifest))

    submission_path = os.path.join(SUBMISSION_DIR, "update_rule.py")
    if not os.path.exists(submission_path):
        return emit(0.0, "submission_update_rule_absent", {"submission_path": submission_path}, {}, {})
    submission_sha = sha256_file(submission_path)
    log.emit("submission_present", submission_path=submission_path, submission_sha256=submission_sha)

    try:
        corpus_probe = FrozenByteCorpus(CORPUS, cfg)
    except Exception as exc:  # noqa: BLE001
        return emit(0.0, f"corpus_unavailable:{exc}", {"corpus": CORPUS}, {}, {})
    corpus_digest = corpus_probe.corpus_digest()
    del corpus_probe

    recipe_digest = hashlib.sha256(
        json.dumps(reference_recipe.PUBLISHED, sort_keys=True, default=str).encode()
    ).hexdigest()
    key = anchor_cache_key(start_manifest, cfg_digest, corpus_digest, device_name, recipe_digest)
    os.makedirs(ANCHOR_DIR, exist_ok=True)
    cache_path = os.path.join(ANCHOR_DIR, f"anchors_{key[:16]}.json")

    arms = {}
    anchor_source = "measured"
    anchor_cache_state = "absent"
    if os.path.exists(cache_path):
        try:
            cached = json.load(open(cache_path))
            presented = str(cached.get("hmac_sha256", ""))
            expected = anchor_tag(anchor_payload_bytes(cached))
            authentic = hmac.compare_digest(presented, expected) and cached.get("key") == key
        except Exception:  # noqa: BLE001
            cached, authentic = None, False
        if not authentic:
            # A persistent store the graded party can write to is a persistent trust store.
            # An entry that does not verify is refused outright rather than recomputed, so a
            # forged cache is a graded zero instead of a silent fall-through that would hide it.
            anchor_cache_state = "unauthenticated"
            log.emit("anchor_cache_rejected", cache_path=cache_path)
        else:
            anchor_cache_state = "authenticated"
            anchor_source = "cache"
            arms["anchor_target"] = cached["anchor_target"]
            arms["anchor_baseline"] = cached["anchor_baseline"]
            log.emit("anchor_target_ready", source="cache", cache_path=cache_path,
                     streaming_val_loss=arms["anchor_target"]["streaming_val_loss"],
                     measured_at=cached.get("measured_at"))
            log.emit("anchor_baseline_ready", source="cache", cache_path=cache_path,
                     streaming_val_loss=arms["anchor_baseline"]["streaming_val_loss"],
                     measured_at=cached.get("measured_at"))
    if anchor_cache_state != "authenticated":
        arms["anchor_target"] = run_arm(
            "anchor_target", "rmsnorm", reference_recipe.factory, cfg, CORPUS, log, WORKDIR, device)
        log.emit("anchor_target_ready", source="measured",
                 streaming_val_loss=arms["anchor_target"]["streaming_val_loss"])
        arms["anchor_baseline"] = run_arm(
            "anchor_baseline", "none", reference_recipe.factory, cfg, CORPUS, log, WORKDIR, device)
        log.emit("anchor_baseline_ready", source="measured",
                 streaming_val_loss=arms["anchor_baseline"]["streaming_val_loss"])
        entry = {"anchor_target": arms["anchor_target"], "anchor_baseline": arms["anchor_baseline"],
                 "measured_at": time.time(), "key": key, "device_name": device_name,
                 "config_digest": cfg_digest, "corpus_digest": corpus_digest}
        entry["hmac_sha256"] = anchor_tag(anchor_payload_bytes(entry))
        json.dump(entry, open(cache_path, "w"), indent=1, sort_keys=True, default=str)

    log.emit("agent_arm_start", submission_sha256=submission_sha)
    agent_arm, arm_err, worker_pid, worker_record = run_agent_arm_out_of_process(
        submission_path, SMOKE, device, WORKDIR, RUN_RECORD)
    if agent_arm is None:
        return emit(0.0, arm_err, {"submission_path": submission_path,
                                   "agent_arm_worker_pid": worker_pid}, {}, {})
    arms["agent"] = agent_arm
    # The worker writes its arm records into its own file so it cannot truncate the verifier's.
    # They are re-emitted here, in the verifier's own sequence, and every one of them is a
    # claim the worker made rather than something this process observed. The verifier's own
    # observations are the displacement below, the independent evaluation, and the anchors.
    for rec in (read_records(worker_record)[0] or []):
        payload = {k: v for k, v in rec.items() if k not in ("seq", "ts", "event", "phase")}
        log.emit(rec.get("event", "worker_event"), phase=rec.get("phase", "harness"),
                 reported_by="arm_worker", **payload)
    log.emit("agent_arm_complete", streaming_val_loss=arms["agent"]["streaming_val_loss"],
             aborted=arms["agent"]["aborted"], worker_pid=worker_pid,
             runner_pid=os.getpid())

    independent = None
    try:
        independent = independent_eval_from_checkpoint(arms["agent"]["checkpoint"], CORPUS, cfg, device)
    except Exception as exc:  # noqa: BLE001
        log.emit("independent_eval_failed", error=f"{type(exc).__name__}:{exc}")
    log.emit("independent_eval_complete", independent_val_loss=independent)

    displacement = None
    try:
        displacement = weight_displacement_from_checkpoint(arms["agent"]["checkpoint"], cfg, device)
    except Exception as exc:  # noqa: BLE001
        log.emit("weight_displacement_failed", error=f"{type(exc).__name__}:{exc}")
    log.emit("weight_displacement_measured", agent_weight_displacement_l2=displacement)

    end_manifest = substrate_manifest(SUBSTRATE_DIR)
    log.emit("scoring_begin")

    ceiling = math.log(cfg["vocab_size"])
    target_metric = arms["anchor_target"]["streaming_val_loss"]
    baseline_metric = arms["anchor_baseline"]["streaming_val_loss"]
    baseline_diverged = baseline_metric is None or not math.isfinite(float(baseline_metric))
    if baseline_diverged:
        baseline_metric = ceiling
    agent_metric = arms["agent"]["streaming_val_loss"]
    agent_nonfinite = agent_metric is None or not math.isfinite(float(agent_metric))
    if agent_nonfinite:
        agent_metric = ceiling

    if target_metric is None or not math.isfinite(float(target_metric)):
        raw = 0.0
    else:
        denom = float(baseline_metric) - float(target_metric)
        raw = 0.0 if denom == 0.0 else (float(baseline_metric) - float(agent_metric)) / denom
    provisional = min(max(float(raw), 0.0), 1.0)

    evidence = {
        "arms": arms,
        "substrate_manifest_pinned": pinned if pinned is not None else start_manifest,
        "substrate_manifest_start": start_manifest,
        "substrate_manifest_end": end_manifest,
        "run_records": (read_records(RUN_RECORD)[0] or []),
        "independent_val_loss": independent,
        "anchors": {
            "target_metric": target_metric,
            "baseline_metric": baseline_metric,
            "baseline_diverged": baseline_diverged,
            "untrained_ceiling": ceiling,
        },
        "agent_metric": agent_metric,
        "agent_nonfinite": agent_nonfinite,
        "raw": raw,
        "score": provisional,
        "params": {"min_anchor_gap": C.MIN_ANCHOR_GAP, "divergence_tol": C.DIVERGENCE_TOL},
        "submission_path": submission_path,
        "submission_sha256": submission_sha,
        "profile": cfg["profile"],
        "device_name": device_name,
        "pinned_manifest_present": pinned is not None,
        "anchor_source": anchor_source,
        "anchor_cache_state": anchor_cache_state,
        "agent_arm_isolation": {
            "mode": "subprocess",
            "worker": os.path.basename(ARM_WORKER),
            "runner_pid": os.getpid(),
            "worker_pid": worker_pid,
        },
        "agent_weight_displacement_l2": displacement,
    }

    outcomes, first_failure = CHECKERS.run_all(evidence)

    if pinned is None:
        first_failure = first_failure or ("substrate_bytes_unmodified", "substrate_manifest_absent")

    if first_failure is not None:
        name, reason = first_failure
        final, final_reason = 0.0, reason
    elif agent_nonfinite:
        final, final_reason = 0.0, "agent_arm_diverged"
    elif arms["agent"]["aborted"]:
        final, final_reason = 0.0, arms["agent"]["aborted"]
    else:
        final, final_reason = provisional, "scored"

    detail = {
        "target_metric": target_metric,
        "baseline_metric": baseline_metric,
        "agent_metric": agent_metric,
        "baseline_diverged": baseline_diverged,
        "raw": raw,
        "provisional_score": provisional,
        "independent_val_loss": independent,
        "profile": cfg["profile"],
        "device_name": device_name,
        "submission_sha256": submission_sha,
        "checkers_failed": [k for k, v in outcomes.items() if not v["passed"]],
        "arm_wallclock_sec": {k: v.get("wallclock_sec") for k, v in arms.items()},
    }
    log.emit("score_emitted", score=final, reason=final_reason)
    evidence["run_records"] = read_records(RUN_RECORD)[0] or []
    emit(final, final_reason, detail, outcomes, evidence)
    return 0


if __name__ == "__main__":
    # The verdict travels in score.json, never in the exit code, so a scored
    # zero and a clean run both exit zero and only a crash exits nonzero.
    main()
    sys.exit(0)
