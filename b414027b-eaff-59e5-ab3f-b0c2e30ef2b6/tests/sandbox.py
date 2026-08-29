"""Parent-owned isolation for the submission process.

The submission is the only untrusted code in this bundle, and every earlier
break in this slot came from the same root cause: the verifier launched it with
`env = dict(os.environ)` and therefore handed it the filesystem path to every
artifact the grade is computed from. Knowing a path is knowing the artifact, so
the fix is not a denylist of forbidden strings. It is a closed construction: the
parent builds the child's whole world and the child receives nothing else.

Three separate mechanisms, each with its own rejecting half.

ALLOWLIST. `child_env` never inherits the verifier's environment. It copies a
fixed, closed set of host variables that are needed to run a Python process on
a GPU host and carry no task state, then adds exactly the six task variables the
submission is contractually entitled to. Every withheld path, the validation
shard, the anchors, the rubric verdicts, the reward file and the bundle hash, is
absent by construction rather than removed by name, so a variable added to the
verifier later cannot leak by default.

STAGING. `stage_run` builds a per-seed sandbox directory holding a copy of the
submission, the shape file, the token grid, and only those data shards that
match the declared training glob. The child's working directory and every path
in its environment sit inside that sandbox. A withheld artifact that happens to
live beside a training shard therefore does not travel with it, which is the
leak that survives environment scrubbing alone.

WITHHOLDING. `withheld_during` removes the validation shard from the filesystem
for the lifetime of the child process and restores it, digest-checked, when the
child exits. Scrubbing the environment stops the submission being told where the
withheld split is; withholding stops it finding out by searching. The parent
holds the only copy in memory while a child is alive, so there is no file
anywhere for a scan to reach.

`isolation_violations` is the deterministic predicate over the first two. It is
a checker, not a comment: the grader refuses to score when it returns anything,
and its rejecting half is exercised by placing a withheld artifact inside the
staged sandbox and requiring a violation to be reported.
"""
from __future__ import annotations

import contextlib
import glob as globlib
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

# Host variables a Python training process needs that carry no task state. This
# tuple is the whole inheritance surface; anything not named here never reaches
# the child, including a variable this bundle has not heard of yet.
PASSTHROUGH = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "TMPDIR",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
)

# Task variables the submission is entitled to. Named here so the set is closed
# in both directions: the child gets these and only these.
GRANTED = (
    "BIA_SEED",
    "BIA_SHAPE",
    "BIA_TOKEN_GRID",
    "BIA_TRAIN_SHARDS",
    "BIA_CHECKPOINT_DIR",
    "BIA_DEVICE",
    "PYTHONPATH",
)


def _real(p) -> pathlib.Path:
    return pathlib.Path(os.path.realpath(str(p)))


def _contains(directory: pathlib.Path, target: pathlib.Path) -> bool:
    """True when target sits at or below directory, both resolved."""
    try:
        target.relative_to(directory)
    except ValueError:
        return False
    return True


def child_env(
    *,
    seed: int,
    shape_path: pathlib.Path,
    token_grid_path: pathlib.Path,
    train_shards: str,
    checkpoint_dir: pathlib.Path,
    device: str | None,
    module_path: str,
) -> dict[str, str]:
    """The child's complete environment, constructed rather than inherited."""
    env: dict[str, str] = {}
    for key in PASSTHROUGH:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["BIA_SEED"] = str(seed)
    env["BIA_SHAPE"] = str(shape_path)
    env["BIA_TOKEN_GRID"] = str(token_grid_path)
    env["BIA_TRAIN_SHARDS"] = train_shards
    env["BIA_CHECKPOINT_DIR"] = str(checkpoint_dir)
    env["PYTHONPATH"] = module_path
    if device:
        env["BIA_DEVICE"] = device
    return env


def stage_run(
    sandbox_root: pathlib.Path,
    *,
    submission: pathlib.Path,
    shape_path: pathlib.Path,
    token_grid_path: pathlib.Path,
    train_shard_glob: str,
) -> dict:
    """Build the child's whole visible surface under sandbox_root.

    Returns the staged paths. Raises ValueError when the declared training glob
    matches nothing, because a run with no training data must fail closed rather
    than quietly train on an empty stream.
    """
    sandbox_root.mkdir(parents=True, exist_ok=True)
    env_dir = sandbox_root / "env"
    data_dir = sandbox_root / "data"
    ckpt_dir = sandbox_root / "checkpoints"
    for d in (env_dir, data_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    staged_shape = env_dir / "shape.json"
    staged_grid = env_dir / "token_grid.json"
    shutil.copyfile(shape_path, staged_shape)
    shutil.copyfile(token_grid_path, staged_grid)

    matched = sorted(globlib.glob(str(train_shard_glob)))
    if not matched:
        raise ValueError(f"no training shard matches {train_shard_glob!r}")
    for src in matched:
        dest = data_dir / pathlib.Path(src).name
        if dest.exists():
            continue
        try:
            os.link(src, dest)  # no path to the original travels with a hard link
        except OSError:
            shutil.copyfile(src, dest)

    staged_submission = sandbox_root / submission.name
    shutil.copyfile(submission, staged_submission)

    return {
        "sandbox_root": sandbox_root,
        "submission": staged_submission,
        "shape_path": staged_shape,
        "token_grid_path": staged_grid,
        "train_shards": str(data_dir / pathlib.Path(train_shard_glob).name),
        "checkpoint_dir": ckpt_dir,
        "staged_shards": [pathlib.Path(p).name for p in matched],
        "matched_sources": matched,
    }


def isolation_violations(
    env: dict[str, str],
    sandbox_root: pathlib.Path,
    withheld: dict[str, pathlib.Path | None],
    staged_sources: list[str] | None = None,
) -> list[str]:
    """Every way a withheld artifact is reachable from what the child was given.

    Two relations, both structural.

    Containment: a withheld artifact sitting inside the staged sandbox is
    reachable by the child's working directory alone.

    Naming: a withheld artifact named by, or sitting in a directory named by,
    any value in the child's environment is reachable by reading that variable.
    Directory containment rather than string equality is the load-bearing half,
    because handing over `train_*.bin` in a directory that also holds `val.bin`
    hands over `val.bin`.
    """
    out: list[str] = []
    root = _real(sandbox_root)

    named_dirs: set[pathlib.Path] = set()
    named_files: set[pathlib.Path] = set()
    # A shard the stager copied into the sandbox was handed over as surely as
    # one named in a variable, so the sources it matched count as named. Without
    # this the check reports clean when a mount places the validation shard
    # under the training glob, because the ORIGINAL then sits outside the
    # sandbox while its copy sits inside.
    for src in staged_sources or []:
        named_files.add(_real(src))
    for key, value in env.items():
        # Only the task variables are examined. A passthrough such as HOME or
        # TMPDIR names a directory that contains most of the filesystem, so
        # treating it as naming an artifact reports every file as reachable and
        # says nothing about what the verifier handed over. This checker answers
        # exactly one question: did the verifier give the submission a path to a
        # withheld artifact. Whether the submission can find one by searching is
        # the separate question withheld_during answers.
        if not value or not key.startswith("BIA_"):
            continue
        # Only values that are actually paths name a directory. Without this a
        # scalar such as BIA_SEED=1234 resolves through Path('1234').parent to
        # the verifier's own working directory, which then reports every file
        # beside the grader as reachable. Measured on the sibling slot.
        if os.sep not in value:
            continue
        raw = pathlib.Path(value)
        if "*" in value or "?" in value:
            # A glob names its whole directory, not only the files it matches,
            # which is exactly how a validation shard mounted beside the
            # training shards becomes reachable without ever being named.
            head = value.split("*")[0].split("?")[0]
            named_dirs.add(_real(pathlib.Path(head) if head.endswith(os.sep)
                                 else pathlib.Path(head).parent))
        else:
            named_files.add(_real(raw))
            named_dirs.add(_real(raw.parent))

    for label, path in sorted(withheld.items()):
        if path is None:
            continue
        target = _real(path)
        if not target.exists():
            continue
        if _contains(root, target):
            out.append(f"{label}-inside-sandbox")
            continue
        if target in named_files:
            out.append(f"{label}-named-by-submission-environment")
            continue
        for d in named_dirs:
            if _contains(d, target):
                out.append(f"{label}-reachable-from-submission-environment")
                break
    return out


@contextlib.contextmanager
def withheld_during(path: pathlib.Path | None):
    """Hold the withheld split in parent memory, off disk, for the duration.

    Scrubbing the environment stops the submission being told where the
    validation split is. This stops it finding out by walking the filesystem,
    because while the child is alive no file anywhere holds those bytes.

    The restore is digest-checked and runs in a finally, so a failing child
    cannot leave the shard missing. When the shard is not writable by the
    verifier the manager yields False and the caller records the named gap
    rather than proceeding as though the shard had been withheld.
    """
    if path is None or not pathlib.Path(path).is_file():
        yield False
        return
    p = pathlib.Path(path)
    payload = p.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    mode = p.stat().st_mode
    try:
        p.unlink()
    except OSError:
        yield False
        return
    try:
        yield True
    finally:
        p.write_bytes(payload)
        os.chmod(p, mode & 0o7777)
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:  # pragma: no cover
            raise RuntimeError("validation shard did not restore to its original bytes")


def launch(
    *,
    submission: pathlib.Path,
    seed: int,
    sandbox_root: pathlib.Path,
    shape_path: pathlib.Path,
    token_grid_path: pathlib.Path,
    train_shard_glob: str,
    module_path: str,
    device: str | None,
    timeout_s: int,
    withheld: dict[str, pathlib.Path | None],
) -> tuple[bool, str, dict]:
    """Stage, verify isolation, then run the submission as a separate process.

    The channel is one way. The parent writes the sandbox and reads the
    checkpoint directory afterwards; nothing the child prints is read, because
    stdout is a surface the child controls and this bundle grades none of it.
    """
    try:
        staged = stage_run(
            sandbox_root,
            submission=submission,
            shape_path=shape_path,
            token_grid_path=token_grid_path,
            train_shard_glob=train_shard_glob,
        )
    except ValueError as exc:
        return False, f"staging-failed: {exc}", {}

    env = child_env(
        seed=seed,
        shape_path=staged["shape_path"],
        token_grid_path=staged["token_grid_path"],
        train_shards=staged["train_shards"],
        checkpoint_dir=staged["checkpoint_dir"],
        device=device,
        module_path=module_path,
    )

    violations = isolation_violations(env, staged["sandbox_root"], withheld,
                                      staged.get("matched_sources"))
    if violations:
        return False, "val-shard-reachable-from-submission", {
            "isolation_violations": violations, "staged": staged}

    with withheld_during(withheld.get("val_shard")) as held:
        try:
            proc = subprocess.run(
                [sys.executable, str(staged["submission"])],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(staged["sandbox_root"]),
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, "run-timeout", {"staged": staged, "withheld_enforced": held}
    if proc.returncode != 0:
        return False, f"run-failed-exit-{proc.returncode}", {
            "staged": staged, "withheld_enforced": held}
    return True, "", {"staged": staged, "withheld_enforced": held,
                      "isolation_violations": []}
