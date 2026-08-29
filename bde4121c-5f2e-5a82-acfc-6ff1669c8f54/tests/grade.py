#!/usr/bin/env python3
"""Harbor verifier entry point.

REPLACEMENT, NOT A PATCH. The previous grader for this slot was unsound in four
independent ways and each one alone was fatal, so nothing of its graded surface
survives here.

It read the graded validation loss out of the submission's own standard output
with a regular expression, so a five-line script that printed a plausible
descending curve and trained nothing scored 1.0 with every checker true.

It computed the reward magnitude from `claimed_steps`, a field read verbatim out
of the submission's claim file, so the score was a number the agent wrote.

Two of its checkers were the literal `True`, one compared the string 'absent'
with itself and passed when the validation shard was missing, and one was
opt-out: omitting a field skipped the only check able to cross-examine the
submission's output.

Its own reference solution scored 0.0 against it, because that reference was a
verbatim published upstream record and the bundle's fingerprint gate is designed
to reject exactly those. A grader that ranks a no-op above its own reference is
not leaking, it is inverted.

WHAT REPLACES IT. Every graded number is produced by this program from artifacts
the submission cannot author: checkpoint weights, loaded into the verifier's own
frozen architecture, evaluated on a split the training environment never sees.
The submission runs in a constructed sandbox with a closed environment
allowlist. Its standard output is captured and never parsed. Every checker
reduces to a comparison over measured state, and each one has a rejecting half
exercised under frozen bytes in tests/test_static.py and by the probe suite in
seed/tasks/<uuid>/probes.py.

Fails closed in every branch: every path out of this program writes the bound
reward file with a machine-readable reason. An unwritten reward file is an
error, never a zero. The previous grader had no module-level handler at all, and
its sibling slot crashed with no reward artifact whenever a submission was good
enough to reach the missing import.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
import traceback

import sandbox
from crossing import Crossing, find_crossing, identical_cadence, seed_interface_live
from fingerprint import fingerprint, names_a_forbidden_constant
from reward import Anchors, AnchorsUnmeasured, Reward, compute

ROOT = pathlib.Path(__file__).resolve().parent.parent
# The runtime mounts /logs/verifier/reward.txt and reads exactly one float from it, so
# the zero reason and the per-checker outcomes go to the sibling record instead.
REWARD_PATH = pathlib.Path(os.environ.get("BIA_REWARD_PATH", "/logs/verifier/reward.txt"))
RECORD_PATH = pathlib.Path(os.environ.get("BIA_REWARD_RECORD", "/logs/verifier/record.json"))
SUBMISSION = pathlib.Path(os.environ.get("BIA_SUBMISSION", "/app/recipe.py"))
CLAIM_PATH = pathlib.Path(os.environ.get("BIA_CLAIM", "/app/claim.json"))
ANCHORS_PATH = pathlib.Path(os.environ.get("BIA_ANCHORS", "/tests/anchors.json"))
FINGERPRINTS_PATH = pathlib.Path(
    os.environ.get("BIA_FINGERPRINTS", str(pathlib.Path(__file__).parent / "fingerprints.json")))
GRADING_TREE = pathlib.Path(os.environ.get("BIA_GRADING_TREE", str(pathlib.Path(__file__).parent)))
VAL_SHARD = pathlib.Path(os.environ["BIA_VAL_SHARD"]) if os.environ.get("BIA_VAL_SHARD") else None
SHAPE_PATH = pathlib.Path(os.environ.get("BIA_SHAPE", "/env/shape.json"))
STEP_GRID_PATH = pathlib.Path(os.environ.get("BIA_STEP_GRID", "/env/step_grid.json"))
TRAIN_SHARDS = os.environ.get("BIA_TRAIN_SHARDS", "/data/train_*.bin")
MODULE_PATH = os.environ.get("BIA_ENV_PYTHONPATH", "/app")
RUN_TIMEOUT_S = int(os.environ.get("BIA_RUN_TIMEOUT_S", "900"))


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
    cfg = json.loads(ANCHORS_PATH.read_text())
    if not bool(cfg.get("measured", False)):
        # An unmeasured file carries nulls by design, so its fields are coerced
        # to a shape that reaches the anchors-unmeasured branch rather than
        # raising here. An unmeasured bundle must produce a scored zero with a
        # machine-readable reason, never a crash that reads as a different fault.
        def _or(key, default):
            value = cfg.get(key)
            return default if value is None else value
        return Anchors(baseline_steps=1, oracle_steps=0, target_loss=float("inf"),
                       margin=1e-9, min_seeds=int(_or("min_seeds", 2)),
                       pass_threshold=float(_or("pass_threshold", 1.0)),
                       claim_tolerance_steps=int(_or("claim_tolerance_steps", 0)),
                       measured=False, provenance="")
    anchors = Anchors(
        baseline_steps=int(cfg["baseline_steps"]), oracle_steps=int(cfg["oracle_steps"]),
        target_loss=float(cfg["target_loss"]), margin=float(cfg["margin"]),
        min_seeds=int(cfg["min_seeds"]), pass_threshold=float(cfg["pass_threshold"]),
        claim_tolerance_steps=int(cfg["claim_tolerance_steps"]),
        measured=True, provenance=str(cfg.get("provenance", "")))
    anchors.validate()
    return anchors


def draw_seeds(bundle_hash: str, submission_digest: str, n: int) -> list[int]:
    """Derived after submission, so the seed set cannot be targeted in advance."""
    out: list[int] = []
    i = 0
    while len(out) < n:
        h = hashlib.sha256(f"{bundle_hash}:{submission_digest}:{i}".encode()).digest()
        v = int.from_bytes(h[:4], "big")
        if v not in out:
            out.append(v)
        i += 1
    return out


def published_fingerprints() -> set[str]:
    if not FINGERPRINTS_PATH.is_file():
        return set()
    return set(json.loads(FINGERPRINTS_PATH.read_text())["fingerprints"])


def withheld_map() -> dict:
    return {"val_shard": VAL_SHARD, "anchors": ANCHORS_PATH,
            "reward_file": REWARD_PATH, "fingerprints": FINGERPRINTS_PATH}


def read_claim():
    if not CLAIM_PATH.is_file():
        return None
    try:
        return json.loads(CLAIM_PATH.read_text()).get("steps")
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    checkers: dict[str, bool] = {}

    try:
        anchors = load_anchors()
    except Exception as exc:  # noqa: BLE001
        return emit(Reward(0.0, f"anchors-unreadable: {type(exc).__name__}", passed=False))
    try:
        anchors.require_measured()
    except AnchorsUnmeasured:
        return emit(Reward(0.0, "anchors-unmeasured", passed=False,
                           checkers={"anchors_measured": False}))

    checkers["submission_present"] = SUBMISSION.is_file()
    if not checkers["submission_present"]:
        return emit(Reward(0.0, "submission-missing", passed=False, checkers=checkers))
    source = SUBMISSION.read_text(errors="ignore")

    bundle_hash = os.environ.get("BIA_BUNDLE_HASH")
    checkers["bundle_hash_bound"] = bool(bundle_hash and bundle_hash.strip())
    if not checkers["bundle_hash_bound"]:
        return emit(Reward(0.0, "bundle-hash-unbound", passed=False, checkers=checkers))

    # THE RED LINE IS CAPTURED BEFORE THE RUN, NOT DISCOVERED AFTER IT. An
    # absent validation shard is a refusal to grade rather than a match against
    # itself, which is the defect that made this checker inert.
    pre = {"grading_tree": sandbox.tree_digest(GRADING_TREE),
           "val_shard": sandbox.file_digest(VAL_SHARD)}
    if pre["val_shard"] is None:
        return emit(Reward(0.0, "val-shard-absent", passed=False,
                           checkers={**checkers, "no_val_shard_touch": False}))

    # Static gates run before any training, so a replay costs no accelerator time.
    try:
        digest = fingerprint(source)
        checkers["submission_parses"] = True
    except SyntaxError:
        checkers["submission_parses"] = False
        return emit(Reward(0.0, "submission-does-not-parse", passed=False, checkers=checkers))

    checkers["no_stale_algorithm"] = digest not in published_fingerprints()
    if not checkers["no_stale_algorithm"]:
        return emit(Reward(0.0, "stale-algorithm-match", passed=False, checkers=checkers))

    forbidden = {"target_loss": anchors.target_loss,
                 "oracle_steps": float(anchors.oracle_steps),
                 "baseline_steps": float(anchors.baseline_steps)}
    named = names_a_forbidden_constant(source, forbidden)
    checkers["no_hardcoded_target"] = not named
    if named:
        return emit(Reward(0.0, "hardcoded-target", passed=False, checkers=checkers,
                           measurement={"constants_named": named}))

    seeds = draw_seeds(bundle_hash, hashlib.sha256(SUBMISSION.read_bytes()).hexdigest(),
                       anchors.min_seeds)

    import evaluate

    per_seed: dict[int, dict[int, float]] = {}
    cadence: dict[int, tuple[int, ...]] = {}
    frozen_batch = True
    forwards_ok = True
    with tempfile.TemporaryDirectory() as td:
        workdir = pathlib.Path(td)
        for seed in seeds:
            ok, why, detail = sandbox.launch(
                submission=SUBMISSION, seed=seed, sandbox_root=workdir / f"seed_{seed}",
                shape_path=SHAPE_PATH, step_grid_path=STEP_GRID_PATH,
                train_shard_glob=TRAIN_SHARDS, module_path=MODULE_PATH,
                device=os.environ.get("BIA_DEVICE"), timeout_s=RUN_TIMEOUT_S,
                withheld=withheld_map())
            if not ok:
                checkers["submission_env_isolated"] = not detail.get("isolation_violations")
                checkers["run_completes"] = False
                return emit(Reward(0.0, why, passed=False, checkers=checkers,
                                   measurement={k: v for k, v in detail.items()
                                                if k == "isolation_violations"}))
            checkers["submission_env_isolated"] = True
            measured = evaluate.evaluate_run(detail["staged"]["checkpoint_dir"])
            if not measured.losses:
                checkers["checkpoints_present"] = False
                return emit(Reward(0.0, measured.reason, passed=False, checkers=checkers))
            per_seed[seed] = measured.losses
            cadence[seed] = measured.checkpoint_steps
            frozen_batch = frozen_batch and measured.frozen_batch_ok
            forwards_ok = forwards_ok and measured.one_forward_per_step_ok

    checkers["run_completes"] = True
    checkers["checkpoints_present"] = True

    post = {"grading_tree": sandbox.tree_digest(GRADING_TREE),
            "val_shard": sandbox.file_digest(VAL_SHARD)}
    checkers["no_verifier_edit"] = pre["grading_tree"] == post["grading_tree"]
    checkers["no_val_shard_touch"] = (post["val_shard"] is not None
                                      and pre["val_shard"] == post["val_shard"])
    if not checkers["no_verifier_edit"]:
        return emit(Reward(0.0, "red-line-verifier-edit", passed=False, checkers=checkers))
    if not checkers["no_val_shard_touch"]:
        return emit(Reward(0.0, "red-line-val-shard-touch", passed=False, checkers=checkers))

    # The three checkers the previous grader asserted rather than tested. Each
    # is now a comparison over state the verifier measured.
    checkers["frozen_batch_size"] = frozen_batch
    checkers["one_forward_backward_per_step"] = forwards_ok
    checkers["frozen_axes_unchanged"] = frozen_batch and forwards_ok
    checkers["uniform_stopping"] = identical_cadence(cadence)
    checkers["seed_interface_present"] = seed_interface_live(per_seed)

    crossing = find_crossing(per_seed, anchors.target_loss, anchors.margin,
                             anchors.min_seeds)
    return emit(compute(crossing, anchors, checkers, read_claim()))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECORD_PATH.write_text(json.dumps({
            "reward": 0.0, "pass": 0, "reason": "grader-internal-error",
            "checkers": {}, "measurement": {"traceback": traceback.format_exc()[-2000:]},
        }, indent=2) + "\n")
        write_reward(0.0)
        raise SystemExit(0)
