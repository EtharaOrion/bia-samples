#!/usr/bin/env python3
"""Harbor verifier entry point.

Ported from a prior voided attempt, re-keyed from optimizer steps to training
tokens.

Runs the submission on seeds drawn after submission, evaluates the checkpoints
the provided loader produced with verifier-owned code, and writes a bounded
continuous reward.

Fails closed in every branch: any path out of this program writes a reward file
with a machine-readable reason. An unwritten reward file is an error, not a zero,
so there is no route by which a crash becomes a silent pass.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import traceback

from crossing import find_crossing
from reward import Anchors, AnchorsUnmeasured, Reward, compute

# The runtime mounts /logs/verifier/reward.txt and reads exactly one float from it, so
# the zero reason and the per-checker outcomes go to the sibling record instead.
REWARD_PATH = pathlib.Path(os.environ.get("BIA_REWARD_PATH", "/logs/verifier/reward.txt"))
RECORD_PATH = pathlib.Path(os.environ.get("BIA_REWARD_RECORD", "/logs/verifier/record.json"))
SUBMISSION = pathlib.Path(os.environ.get("BIA_SUBMISSION", "/app/train.py"))
ANCHORS_PATH = pathlib.Path(os.environ.get("BIA_ANCHORS", "/tests/anchors.json"))
RUBRIC_VERDICTS = pathlib.Path(os.environ.get("BIA_RUBRIC_VERDICTS", "/logs/verifier/rubric_verdicts.json"))
SEED_ENV = "BIA_SEED"
CHECKPOINT_ENV = "BIA_CHECKPOINT_DIR"
RUN_TIMEOUT_S = int(os.environ.get("BIA_RUN_TIMEOUT_S", "5400"))


def write_reward(value: float) -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text("%.6f\n" % min(max(float(value), 0.0), 1.0))


def emit(reward: Reward) -> int:
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = reward.as_payload()
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_reward(payload["reward"])
    print(json.dumps({"reward": payload["reward"], "reason": payload["reason"]}))
    return 0


def load_anchors() -> Anchors:
    """Read anchors. A null anchor is unmeasured, never a number.

    Nulls are coerced to a shape validate() rejects only when measured is true,
    so an unmeasured file reaches the anchors-unmeasured branch in reward.compute
    rather than raising here. That ordering matters: an unmeasured bundle must
    produce a scored zero with a machine-readable reason, not a crash.
    """
    cfg = json.loads(ANCHORS_PATH.read_text())
    measured = bool(cfg.get("measured", False))
    if not measured:
        return Anchors(
            baseline_tokens=1, oracle_tokens=0,
            target_loss=float("inf"), margin=1e-9, min_seeds=int(cfg.get("min_seeds", 1)),
            pass_threshold=float(cfg.get("pass_threshold", 1.0)),
            measured=False, provenance="",
        )
    anchors = Anchors(
        baseline_tokens=int(cfg["baseline_tokens"]),
        oracle_tokens=int(cfg["oracle_tokens"]),
        target_loss=float(cfg["target_loss"]),
        margin=float(cfg["margin"]),
        min_seeds=int(cfg["min_seeds"]),
        pass_threshold=float(cfg["pass_threshold"]),
        measured=True,
        provenance=str(cfg.get("provenance", "")),
    )
    anchors.validate()
    return anchors


def draw_seeds(bundle_hash: str, submission_digest: str, n: int) -> list[int]:
    """Derived after submission, so the seed set cannot be targeted in advance.

    This is stronger than a published fixed seed set. With fixed public seeds an
    agent can tune hyperparameters to those specific draws, which is the seed
    lottery run in reverse. Deriving from the submission digest keeps the draw
    deterministic and reproducible for any auditor holding the bundle and the
    submission, while leaving it unknowable to the agent at authoring time.
    """
    out: list[int] = []
    i = 0
    while len(out) < n:
        h = hashlib.sha256(f"{bundle_hash}:{submission_digest}:{i}".encode()).digest()
        v = int.from_bytes(h[:4], "big")
        if v not in out:
            out.append(v)
        i += 1
    return out


def run_one_seed(script: pathlib.Path, seed: int, workdir: pathlib.Path) -> tuple[bool, str]:
    ckpt_dir = workdir / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env[SEED_ENV] = str(seed)
    env[CHECKPOINT_ENV] = str(ckpt_dir)
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=RUN_TIMEOUT_S,
            cwd=str(workdir), env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "run-timeout"
    if proc.returncode != 0:
        return False, f"run-failed-exit-{proc.returncode}"
    return True, ""


def read_rubric_verdicts() -> dict[str, bool] | None:
    if not RUBRIC_VERDICTS.is_file():
        return None
    try:
        data = json.loads(RUBRIC_VERDICTS.read_text())
    except Exception:  # noqa: BLE001
        return {"rubric_verdicts_readable": False}
    if not isinstance(data, dict):
        return {"rubric_verdicts_readable": False}
    return {str(k): bool(v) for k, v in data.items()}


def main() -> int:
    validity: dict[str, bool] = {}
    try:
        anchors = load_anchors()
    except Exception as exc:  # noqa: BLE001
        return emit(Reward(0.0, f"anchors-unreadable: {type(exc).__name__}", passed=False))

    # The anchor guard fires before anything else. If the anchors are not
    # measured on the frozen bundle, this task cannot be scored at all, and
    # reporting some downstream problem instead would misdescribe the blocker.
    # Placing it here means there is no path through this program to a nonzero
    # reward while anchors are unmeasured.
    try:
        anchors.require_measured()
    except AnchorsUnmeasured:
        return emit(Reward(0.0, "anchors-unmeasured", passed=False,
                           checkers={"anchors_measured": False}))

    validity["submission_present"] = SUBMISSION.is_file()
    if not validity["submission_present"]:
        return emit(Reward(0.0, "submission-missing", passed=False, checkers=validity))

    submission_digest = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    bundle_hash = os.environ.get("BIA_BUNDLE_HASH", "unbound")
    seeds = draw_seeds(bundle_hash, submission_digest, anchors.min_seeds)

    import evaluate

    per_seed: dict[int, dict[int, float]] = {}
    with tempfile.TemporaryDirectory() as td:
        workdir = pathlib.Path(td)
        shutil.copy2(SUBMISSION, workdir / SUBMISSION.name)
        script = workdir / SUBMISSION.name
        for seed in seeds:
            ok, why = run_one_seed(script, seed, workdir)
            if not ok:
                validity["run_completes"] = False
                return emit(Reward(0.0, why, passed=False, checkers=validity))
            measured = evaluate.evaluate_run(workdir / f"seed_{seed}")
            if not measured:
                validity["checkpoints_present"] = False
                return emit(Reward(0.0, "no-evaluable-checkpoints", passed=False, checkers=validity))
            per_seed[seed] = measured

    validity["run_completes"] = True
    validity["checkpoints_present"] = True

    crossing = find_crossing(per_seed, anchors.target_loss, anchors.margin, anchors.min_seeds)
    return emit(compute(crossing, anchors, validity, read_rubric_verdicts()))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECORD_PATH.write_text(json.dumps({
            "reward": 0.0,
            "pass": 0,
            "reason": "grader-internal-error",
            "checkers": {},
            "measurement": {"traceback": traceback.format_exc()[-2000:]},
        }, indent=2) + "\n")
        write_reward(0.0)
        raise SystemExit(0)
