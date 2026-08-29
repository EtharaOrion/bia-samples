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

THE ORDERING RULE THIS PROGRAM NOW APPLIES UNIFORMLY. Every artifact the grade
reads is captured before the submission runs, or is produced by the verifier
after it. The anchors always followed that rule and were never forgeable. The
rubric verdicts did not: they were re-read after the run from a path handed to
the submission, so a submission could write its own verdicts and did. Both are
now read once, before any child process exists, into frozen values, and the
verdict file is additionally digest-checked across the run so a mutation is a
scored refusal rather than a silent substitution.

THE ISOLATION RULE. The submission runs in a separate process whose entire
environment and entire working directory are constructed by this program, in
tests/sandbox.py. It is handed the six variables the task grants it and nothing
else, and it is staged into a directory that holds only the training shards its
declared glob matched. The withheld validation split is off disk for the
lifetime of the child. Nothing this program grades is read from the child's
stdout, because stdout is a surface the child controls.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
import traceback

import sandbox
from crossing import find_crossing
from reward import Anchors, AnchorsUnmeasured, Reward, compute

# The runtime mounts /logs/verifier/reward.txt and reads exactly one float from it, so
# the zero reason and the per-checker outcomes go to the sibling record instead.
REWARD_PATH = pathlib.Path(os.environ.get("BIA_REWARD_PATH", "/logs/verifier/reward.txt"))
RECORD_PATH = pathlib.Path(os.environ.get("BIA_REWARD_RECORD", "/logs/verifier/record.json"))
SUBMISSION = pathlib.Path(os.environ.get("BIA_SUBMISSION", "/app/train.py"))
ANCHORS_PATH = pathlib.Path(os.environ.get("BIA_ANCHORS", "/tests/anchors.json"))
RUBRIC_VERDICTS = pathlib.Path(os.environ.get("BIA_RUBRIC_VERDICTS", "/logs/verifier/rubric_verdicts.json"))
VAL_SHARD = pathlib.Path(os.environ["BIA_VAL_SHARD"]) if os.environ.get("BIA_VAL_SHARD") else None
SHAPE_PATH = pathlib.Path(os.environ.get("BIA_SHAPE", "/env/shape.json"))
TOKEN_GRID_PATH = pathlib.Path(os.environ.get("BIA_TOKEN_GRID", "/env/token_grid.json"))
TRAIN_SHARDS = os.environ.get("BIA_TRAIN_SHARDS", "/data/train_*.bin")
# Where the submission's own modules live. The provided loader and the frozen
# architecture ship here; the verifier tree deliberately does not, so a
# submission cannot import the grading code that scores it.
MODULE_PATH = os.environ.get("BIA_ENV_PYTHONPATH", "/app")
RUN_TIMEOUT_S = int(os.environ.get("BIA_RUN_TIMEOUT_S", "900"))

# Everything the grade is computed from that the submission must not reach.
def withheld_map() -> dict:
    return {
        "val_shard": VAL_SHARD,
        "anchors": ANCHORS_PATH,
        "rubric_verdicts": RUBRIC_VERDICTS,
        "reward_file": REWARD_PATH,
    }


def digest_or_absent(path: pathlib.Path) -> str:
    """Content digest of an artifact, or the literal marker for its absence.

    Absence is a distinct value rather than a neutral one, so a file appearing
    or disappearing across the run is as visible as a file being edited.
    """
    p = pathlib.Path(path)
    if not p.is_file():
        return "absent"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def unmodified(before: str, after: str) -> bool:
    """Both halves are exercised in tests/test_static.py under frozen bytes."""
    return before == after


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


def run_one_seed(script: pathlib.Path, seed: int, workdir: pathlib.Path) -> tuple[bool, str, dict]:
    """One submission run, in its own process and its own constructed world.

    The parent owns the seed, the sandbox, the training shards the child can
    see and the withholding of the validation split. The child returns nothing
    but files in the checkpoint directory the parent named.
    """
    return sandbox.launch(
        submission=script,
        seed=seed,
        sandbox_root=workdir / f"seed_{seed}",
        shape_path=SHAPE_PATH,
        token_grid_path=TOKEN_GRID_PATH,
        train_shard_glob=TRAIN_SHARDS,
        module_path=MODULE_PATH,
        device=os.environ.get("BIA_DEVICE"),
        timeout_s=RUN_TIMEOUT_S,
        withheld=withheld_map(),
    )


def rubric_surface_declared() -> bool:
    """True when this bundle ships natural-language rubrics that need a judge."""
    p = pathlib.Path(__file__).resolve().parent / "rubrics.jsonl"
    return p.is_file() and bool(p.read_text().strip())


def declared_rubric_ids() -> list[str]:
    p = pathlib.Path(__file__).resolve().parent / "rubrics.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(str(json.loads(line)["id"]))
        except Exception:  # noqa: BLE001
            return []
    return out


def read_rubric_verdicts() -> dict[str, bool] | None:
    """Verdicts for the DECLARED rubric ids only.

    Two filters, both load-bearing.

    Keys beginning with an underscore are metadata, not verdicts. Without this a
    verdicts file that documents its own method would have every metadata field
    graded as though it were a rubric, and any falsy one, an empty string or an
    empty object, would zero the reward under a rubric id that does not exist.
    That is a zero for a reason nobody can act on.

    The verdict set is then reconciled against tests/rubrics.jsonl. A declared
    rubric with no verdict fails closed rather than being skipped, because the
    client contract is that only a solution where every rubric passes counts as
    correct, and silence about a rubric is not a pass on it.
    """
    if not RUBRIC_VERDICTS.is_file():
        return None
    try:
        data = json.loads(RUBRIC_VERDICTS.read_text())
    except Exception:  # noqa: BLE001
        return {"rubric_verdicts_readable": False}
    if not isinstance(data, dict):
        return {"rubric_verdicts_readable": False}

    supplied = {str(k): bool(v) for k, v in data.items() if not str(k).startswith("_")}
    declared = declared_rubric_ids()
    if not declared:
        return supplied
    verdicts = {rid: supplied.get(rid, False) for rid in declared}
    for rid in declared:
        if rid not in supplied:
            verdicts[f"rubric_verdict_missing:{rid}"] = False
    return verdicts


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

    # A declared rubric surface with no judge is refused here, before the run
    # loop, for two reasons. It is a property of the task rather than of the
    # submission, so it is knowable without running anything, and running eight
    # training runs before discovering the result cannot be scored wastes the
    # scarcest resource in this project. The client contract is that only a
    # solution where every rubric passes counts as correct, so proceeding without
    # verdicts would score the automated dimensions while silently ignoring the
    # rest, which is a silent pass on the rubric dimension rather than a neutral
    # omission.
    # READ ONCE, BEFORE THE RUN. This is the same discipline load_anchors has
    # always followed, now applied to the verdicts. The value used to grade is
    # the one captured here; the file is never consulted again for its content,
    # only for its digest, so a submission that rewrites it mid-run rewrites
    # something nothing reads.
    verdicts_before_digest = digest_or_absent(RUBRIC_VERDICTS)
    verdicts = read_rubric_verdicts()
    if rubric_surface_declared() and verdicts is None:
        return emit(Reward(0.0, "rubric-verdicts-absent", passed=False,
                           checkers={"anchors_measured": True,
                                     "rubric_verdicts_present": False}))

    validity["submission_present"] = SUBMISSION.is_file()
    if not validity["submission_present"]:
        return emit(Reward(0.0, "submission-missing", passed=False, checkers=validity))

    submission_digest = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    # BIA_BUNDLE_HASH is fatal when unbound. It is one of the two inputs to the
    # seed derivation, so a default would make draw_seeds() return the same seeds
    # for every bundle carrying the same submission digest. That is exactly the
    # predictability the post-submission draw exists to remove, and it would fail
    # silently because a constant is still a valid string.
    bundle_hash = os.environ.get("BIA_BUNDLE_HASH")
    if not bundle_hash or not bundle_hash.strip():
        return emit(Reward(0.0, "bundle-hash-unbound", passed=False,
                           checkers={"bundle_hash_bound": False}))
    validity["bundle_hash_bound"] = True
    seeds = draw_seeds(bundle_hash, submission_digest, anchors.min_seeds)

    import evaluate

    per_seed: dict[int, dict[int, float]] = {}
    with tempfile.TemporaryDirectory() as td:
        workdir = pathlib.Path(td)
        for seed in seeds:
            ok, why, detail = run_one_seed(SUBMISSION, seed, workdir)
            if not ok:
                validity["submission_env_isolated"] = not detail.get("isolation_violations")
                validity["run_completes"] = ok
                return emit(Reward(0.0, why, passed=False, checkers=validity,
                                   measurement={k: v for k, v in detail.items()
                                                if k == "isolation_violations"}))
            validity["submission_env_isolated"] = True
            measured = evaluate.evaluate_run(detail["staged"]["checkpoint_dir"])
            if not measured:
                validity["checkpoints_present"] = False
                return emit(Reward(0.0, "no-evaluable-checkpoints", passed=False, checkers=validity))
            per_seed[seed] = measured

    validity["run_completes"] = True
    validity["checkpoints_present"] = True

    # The rejecting half of the read-once rule. The verdicts used to grade were
    # frozen before the first child existed, so a mutation cannot change the
    # score; recording it as a refusal rather than ignoring it is what turns a
    # silently-inert forgery attempt into a machine-readable zero.
    validity["rubric_verdicts_unmodified"] = unmodified(
        verdicts_before_digest, digest_or_absent(RUBRIC_VERDICTS))
    if not validity["rubric_verdicts_unmodified"]:
        return emit(Reward(0.0, "rubric-verdicts-mutated", passed=False, checkers=validity))

    crossing = find_crossing(per_seed, anchors.target_loss, anchors.margin, anchors.min_seeds)
    return emit(compute(crossing, anchors, validity, verdicts))


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
