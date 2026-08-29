"""Parent-owned isolation for the submission process.

The previous verifier for this slot did none of this. It re-executed the
submitted recipe with the verifier's own environment and then read the graded
loss out of the child's standard output, so the submission both saw every path
the grade depended on and authored the number the grade was computed from. Both
halves of that are removed here.

THE CHANNEL IS ONE WAY AND IT IS NOT STDOUT. The parent writes the sandbox and
reads the checkpoint directory afterwards. Nothing the child prints is parsed by
anything in this bundle. A submission may log whatever it likes; no byte of it
reaches the reward.

THE ENVIRONMENT IS CONSTRUCTED, NOT INHERITED. A closed allowlist of host
variables that carry no task state, plus exactly the task variables the
submission is entitled to. The withheld validation split, the anchors, the
reward path and the bundle hash are absent by construction, so a variable added
to the verifier later cannot leak by default.

THE WORKING DIRECTORY IS BUILT. Only shards matching the declared training glob
are staged, so a withheld artifact that happens to be mounted beside the
training data does not travel with it.

THE WITHHELD SPLIT IS OFF DISK WHILE THE CHILD IS ALIVE. Scrubbing the
environment stops the submission being told where it is. Withholding stops it
finding out by searching.
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

PASSTHROUGH = (
    "PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "TMPDIR",
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUDA_HOME",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE",
)

GRANTED = (
    "BIA_SEED", "BIA_SHAPE", "BIA_STEP_GRID", "BIA_TRAIN_SHARDS",
    "BIA_CHECKPOINT_DIR", "BIA_DEVICE", "PYTHONPATH",
)


def _real(p) -> pathlib.Path:
    return pathlib.Path(os.path.realpath(str(p)))


def _contains(directory: pathlib.Path, target: pathlib.Path) -> bool:
    try:
        target.relative_to(directory)
    except ValueError:
        return False
    return True


def child_env(*, seed, shape_path, step_grid_path, train_shards, checkpoint_dir,
              device, module_path) -> dict[str, str]:
    env = {k: os.environ[k] for k in PASSTHROUGH if k in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["BIA_SEED"] = str(seed)
    env["BIA_SHAPE"] = str(shape_path)
    env["BIA_STEP_GRID"] = str(step_grid_path)
    env["BIA_TRAIN_SHARDS"] = str(train_shards)
    env["BIA_CHECKPOINT_DIR"] = str(checkpoint_dir)
    env["PYTHONPATH"] = str(module_path)
    if device:
        env["BIA_DEVICE"] = str(device)
    return env


def stage_run(sandbox_root, *, submission, shape_path, step_grid_path,
              train_shard_glob) -> dict:
    sandbox_root = pathlib.Path(sandbox_root)
    env_dir, data_dir, ckpt_dir = (sandbox_root / "env", sandbox_root / "data",
                                   sandbox_root / "checkpoints")
    for d in (env_dir, data_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    staged_shape = env_dir / "shape.json"
    staged_grid = env_dir / "step_grid.json"
    shutil.copyfile(shape_path, staged_shape)
    shutil.copyfile(step_grid_path, staged_grid)

    matched = sorted(globlib.glob(str(train_shard_glob)))
    if not matched:
        raise ValueError(f"no training shard matches {train_shard_glob!r}")
    for src in matched:
        dest = data_dir / pathlib.Path(src).name
        if dest.exists():
            continue
        try:
            os.link(src, dest)
        except OSError:
            shutil.copyfile(src, dest)

    staged_submission = sandbox_root / pathlib.Path(submission).name
    shutil.copyfile(submission, staged_submission)
    return {
        "sandbox_root": sandbox_root,
        "submission": staged_submission,
        "shape_path": staged_shape,
        "step_grid_path": staged_grid,
        "train_shards": str(data_dir / pathlib.Path(train_shard_glob).name),
        "checkpoint_dir": ckpt_dir,
        "matched_sources": matched,
    }


def isolation_violations(env, sandbox_root, withheld, staged_sources=None) -> list[str]:
    """Every way a withheld artifact is reachable from what the child was given."""
    out: list[str] = []
    root = _real(sandbox_root)
    named_files = {_real(s) for s in (staged_sources or [])}
    named_dirs: set[pathlib.Path] = set()

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
        # beside the grader as reachable.
        if os.sep not in value:
            continue
        if "*" in value or "?" in value:
            head = value.split("*")[0].split("?")[0]
            named_dirs.add(_real(pathlib.Path(head) if head.endswith(os.sep)
                                 else pathlib.Path(head).parent))
        else:
            named_files.add(_real(value))
            named_dirs.add(_real(pathlib.Path(value).parent))

    for label, path in sorted(withheld.items()):
        if path is None:
            continue
        target = _real(path)
        if not target.exists():
            continue
        if _contains(root, target):
            out.append(f"{label}-inside-sandbox")
        elif target in named_files:
            out.append(f"{label}-named-by-submission-environment")
        elif any(_contains(d, target) for d in named_dirs):
            out.append(f"{label}-reachable-from-submission-environment")
    return out


@contextlib.contextmanager
def withheld_during(path):
    """Hold the withheld split in parent memory, off disk, for the duration."""
    if path is None or not pathlib.Path(path).is_file():
        yield False
        return
    p = pathlib.Path(path)
    payload = p.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    mode = p.stat().st_mode & 0o7777
    try:
        p.unlink()
    except OSError:
        yield False
        return
    try:
        yield True
    finally:
        p.write_bytes(payload)
        os.chmod(p, mode)
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:  # pragma: no cover
            raise RuntimeError("validation shard did not restore to its original bytes")


def tree_digest(path: pathlib.Path) -> str:
    """Path-independent digest over a directory, for the red-line comparison."""
    rows = []
    for p in sorted(pathlib.Path(path).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            rows.append(p.relative_to(path).as_posix() + ":"
                        + hashlib.sha256(p.read_bytes()).hexdigest())
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def file_digest(path) -> str | None:
    """Digest of a file, or None when it is absent.

    None rather than a literal marker, because the previous verifier returned
    the string 'absent' here and then compared it with itself, so a missing
    validation shard passed the red line instead of refusing to grade. The
    caller must decide what absence means; it can no longer be mistaken for a
    match.
    """
    p = pathlib.Path(path) if path is not None else None
    if p is None or not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def launch(*, submission, seed, sandbox_root, shape_path, step_grid_path,
           train_shard_glob, module_path, device, timeout_s, withheld):
    try:
        staged = stage_run(sandbox_root, submission=submission, shape_path=shape_path,
                           step_grid_path=step_grid_path,
                           train_shard_glob=train_shard_glob)
    except ValueError as exc:
        return False, f"staging-failed: {exc}", {}

    env = child_env(seed=seed, shape_path=staged["shape_path"],
                    step_grid_path=staged["step_grid_path"],
                    train_shards=staged["train_shards"],
                    checkpoint_dir=staged["checkpoint_dir"], device=device,
                    module_path=module_path)

    violations = isolation_violations(env, staged["sandbox_root"], withheld,
                                      staged["matched_sources"])
    if violations:
        return False, "withheld-artifact-reachable-from-submission", {
            "isolation_violations": violations, "staged": staged}

    with withheld_during(withheld.get("val_shard")) as held:
        try:
            proc = subprocess.run([sys.executable, str(staged["submission"])],
                                  capture_output=True, text=True, timeout=timeout_s,
                                  cwd=str(staged["sandbox_root"]), env=env)
        except subprocess.TimeoutExpired:
            return False, "run-timeout", {"staged": staged, "withheld_enforced": held}
    if proc.returncode != 0:
        return False, f"run-failed-exit-{proc.returncode}", {
            "staged": staged, "withheld_enforced": held}
    return True, "", {"staged": staged, "withheld_enforced": held,
                      "isolation_violations": []}
