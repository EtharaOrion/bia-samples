#!/usr/bin/env python3
"""Run the submission, alone, in a fresh directory, and return raw artifacts.

This module returns bytes and a status. It computes no verdict, reads no number
the submission printed as a result, and never imports the submitted file. The
grading process (`grade.py`) imports the checkers, the harness and the reference
and never imports the submission, so no module the submission could rebind is
reachable from the process that scores.

Three properties are load-bearing and all three are here rather than in prose:

  * the submission is copied ALONE into a fresh temporary directory, so nothing
    in the bundle's `tests/` or `solution/` trees is on its path;
  * it is launched as a NEW SESSION LEADER under a small environment allowlist,
    so it inherits neither the verifier's environment nor its process group;
  * the whole process group is killed in a `finally` block, so a submission that
    forks or sleeps cannot outlive the run and cannot hold the verifier open.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# The whole environment the submission gets. Anything not listed is not passed.
ENVIRONMENT_ALLOWLIST = ("PATH", "HOME", "TMPDIR")

DEFAULT_TIMEOUT_SEC = 300


@dataclass
class RunResult:
    """What the submission process did. No verdict, no score, no interpretation."""

    workspace: Path
    artifacts: Path
    returncode: int
    timed_out: bool
    stdout: str
    stderr: str
    launched: bool


def _environment() -> dict:
    env = {name: os.environ[name] for name in ENVIRONMENT_ALLOWLIST if name in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    # Fixed, so the submission process is not handed a locale or a hash seed the
    # verifier's own process happened to carry.
    env["LANG"] = "C"
    env["LC_ALL"] = "C"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_submission(
    submission: Path,
    frozen_dir: Path,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> RunResult:
    workspace = Path(tempfile.mkdtemp(prefix="oer16-submission-"))
    artifacts = workspace / "out"
    artifacts.mkdir(parents=True, exist_ok=True)

    source = Path(submission)
    if not source.is_file():
        return RunResult(workspace, artifacts, 127, False, "", "submission absent", False)

    target = workspace / "submission.py"
    shutil.copy2(source, target)
    # The frozen half travels with the submission. It is not held out: the whole
    # point of this slot is that an honest submission CAN compute the figure
    # itself and still earns nothing for saying so.
    shutil.copytree(Path(frozen_dir), workspace / "frozen")

    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "submission.py", "frozen", "out"],
            cwd=str(workspace),
            env=_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        try:
            out, err = process.communicate(timeout=timeout_sec)
            return RunResult(workspace, artifacts, process.returncode, False, out, err, True)
        except subprocess.TimeoutExpired:
            return RunResult(workspace, artifacts, 124, True, "", "submission timed out", True)
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if process is not None:
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass


def cleanup(result: RunResult) -> None:
    shutil.rmtree(result.workspace, ignore_errors=True)
