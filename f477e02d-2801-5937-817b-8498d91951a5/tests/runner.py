#!/usr/bin/env python3
"""Isolation, and the read of built environment state.

Two jobs, both of which have to sit outside the checkers so the checkers stay pure.

FIRST, the producer. The submission is a PROGRAM. The grading process never imports it,
never execs it and never evaluates it: it is launched as a NEW SESSION LEADER, under a small
environment allowlist, with its working directory in a fresh scratch tree, and the whole
process group is killed in a `finally` block whether the run succeeded, failed or timed out.
Only what it printed on standard output crosses back, and it crosses back as JSON data.

SECOND, the seam offset. It is established at image build time by
`environment/establish_state.py` and lives in `environment/state/seam_state.json` INSIDE THE
BUILT IMAGE. It is in no bundle byte. This module reads it back through the same handle the
agent uses. When no built state is present, because the verifier is being exercised outside
its image, the state is re-established over the frozen bytes in a scratch tree, and the
origin is recorded so the verdict says which route was taken instead of hiding it.

Nothing here reads a clock, opens a socket or draws from a random source.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# The whole environment a launched child is given. Nothing else crosses.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONHASHSEED")

DEFAULT_TIMEOUT_SECONDS = 120

# Where a built image puts the environment. Checked before the bundle-local route.
IMAGE_ENVIRONMENT_ROOT = Path("/task/environment")

STATE_RELATIVE = Path("state") / "seam_state.json"

# The environment modules that have to travel together for the state to be re-establishable.
ENVIRONMENT_MEMBERS = (
    "instance.json",
    "envelope.json",
    "establish_state.py",
    "harness.py",
    "fitness.py",
)


class RunFailed(Exception):
    """The producer did not yield a document. Never a score, always a reason."""


def _environment(extra=None):
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    # Fixed so dictionary iteration inside the child cannot vary between runs.
    env["PYTHONHASHSEED"] = "0"
    if extra:
        env.update(extra)
    return env


def _kill_group(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def _remove_tree(root):
    for entry in sorted(Path(root).rglob("*"), reverse=True):
        try:
            entry.unlink() if entry.is_file() else entry.rmdir()
        except OSError:
            pass
    try:
        Path(root).rmdir()
    except OSError:
        pass


def environment_root(bundle: Path):
    """Where the environment the verifier reads state from lives, and how it got there.

    The image route is preferred and is the graded one: the verifier image runs
    `establish_state.py` at build time, so `/task/environment/state/seam_state.json` exists
    before any submission does. The bundle route is next. Re-establishment is last and is
    reported as such.
    """
    if (IMAGE_ENVIRONMENT_ROOT / STATE_RELATIVE).is_file():
        return IMAGE_ENVIRONMENT_ROOT, "built-image-state", None
    local = Path(bundle) / "environment"
    if (local / STATE_RELATIVE).is_file():
        return local, "bundle-local-state", None
    scratch = Path(tempfile.mkdtemp(prefix="oer25-env-"))
    for member in ENVIRONMENT_MEMBERS:
        source = local / member
        if not source.is_file():
            _remove_tree(scratch)
            raise RunFailed("environment member absent: " + source.as_posix())
        shutil.copyfile(source, scratch / member)
    argv = [sys.executable, "-I", "-S", str(scratch / "establish_state.py"), "--root", str(scratch)]
    process = None
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(scratch),
            env=_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        _out, err = process.communicate(timeout=60)
        if process.returncode != 0:
            raise RunFailed(
                "re-establishing environment state exited "
                + str(process.returncode)
                + ": "
                + err.decode("utf-8", "replace").strip()[:400]
            )
    except subprocess.TimeoutExpired:
        raise RunFailed("re-establishing environment state exceeded 60 seconds")
    finally:
        _kill_group(process)
    return scratch, "re-established-over-frozen-bytes", scratch


def read_built_state(bundle: Path):
    """The DISCOVERY READ, through the same handle the agent uses.

    Returns the state document and the origin string. The origin travels into the score
    document so a reader can tell which route supplied the offset.
    """
    root, origin, scratch = environment_root(Path(bundle))
    try:
        document = json.loads((root / STATE_RELATIVE).read_text(encoding="utf-8"))
    finally:
        if scratch is not None and root != scratch:
            _remove_tree(scratch)
    if document.get("schema") != "oer25.state/v1":
        raise RunFailed(
            "built environment state carries schema "
            + repr(document.get("schema"))
            + " and this verifier reads oer25.state/v1"
        )
    return document, origin, root


def run_producer(bundle: Path, submission: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS):
    """Launch the submission program in isolation and return the document it printed."""
    bundle = Path(bundle).resolve()
    submission = Path(submission)
    if not submission.is_file():
        raise RunFailed("no submission program at " + submission.as_posix())

    root, _origin, scratch = environment_root(bundle)
    workdir = Path(tempfile.mkdtemp(prefix="oer25-producer-"))
    process = None
    try:
        argv = [sys.executable, "-I", "-S", str(submission.resolve())]
        process = subprocess.Popen(
            argv,
            cwd=str(workdir),
            env=_environment({"OER25_ENV_ROOT": str(root)}),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RunFailed("the producer exceeded " + str(timeout) + " seconds")
        if process.returncode != 0:
            raise RunFailed(
                "the producer exited "
                + str(process.returncode)
                + ": "
                + err.decode("utf-8", "replace").strip()[:400]
            )
        text = out.decode("utf-8", "replace").strip()
        if not text:
            raise RunFailed("the producer printed nothing on standard output")
        try:
            return json.loads(text)
        except ValueError as failure:
            raise RunFailed("the producer's standard output is not one JSON document: " + str(failure))
    finally:
        _kill_group(process)
        _remove_tree(workdir)
        if scratch is not None:
            _remove_tree(scratch)
