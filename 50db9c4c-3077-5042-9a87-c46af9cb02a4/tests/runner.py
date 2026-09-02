#!/usr/bin/env python3
"""Launch submission stages out of process. The grading process never imports them.

The submission is copied alone into a fresh working directory, each stage is
launched as a new session leader under a small environment allowlist, its output
is captured, and the whole process group is killed in a `finally` block. Nothing
in this file interprets what the submission produced; that is the harness's job
and the checkers' job.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

# The whole environment a submission stage is given. Anything not named here is
# absent from the child, so a stage cannot read a secret this process holds.
ENV_ALLOWLIST = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LC_ALL": "C",
    "LANG": "C",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}

STAGE_TIMEOUT_SECONDS = 120
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "work")


def stage_submission(source: Path, dest: Path) -> Path:
    """Copy the submission alone into a fresh directory. No verifier byte travels."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, symlinks=False, ignore=IGNORED)
    (dest / "work").mkdir(parents=True, exist_ok=True)
    return dest


def launch(workdir: Path, script: str, args: list) -> dict:
    """Run one stage as a new session leader and reap its whole process group."""
    argv = [sys.executable, script] + [str(item) for item in args]
    proc = subprocess.Popen(
        argv,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(ENV_ALLOWLIST),
        start_new_session=True,
        text=True,
    )
    stdout, stderr, returncode = "", "", None
    try:
        stdout, stderr = proc.communicate(timeout=STAGE_TIMEOUT_SECONDS)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        returncode = 124
        stdout, stderr = "", "stage exceeded the bound stage timeout"
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    return {"argv": argv[1:], "returncode": returncode, "stdout": stdout, "stderr": stderr}
