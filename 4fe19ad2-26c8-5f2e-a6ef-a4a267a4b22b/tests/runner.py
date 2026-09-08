"""Launch one attempt of the submission. The grading process never imports it.

The submission is copied ALONE into a fresh temporary directory, so nothing in
the tests tree, the solution tree or the environment tree is importable from
where it runs. It is launched as a new session leader under a small environment
allowlist, its output is captured and never parsed for a graded number, and the
whole process group is killed in a `finally` block so a child that outlives its
parent cannot survive into the next attempt.

The only thing that crosses back from the submission is `proposal.json`, which
is a document over a closed key set. It is never executed, never imported, and
never trusted for a graded value.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time

# The whole environment a submission gets. Nothing else crosses, so a secret in
# the verifier's environment cannot be read by an attempt and then reported back.
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONHASHSEED")


class Attempt:
    """What one launch produced. `proposal` is a document or None."""

    def __init__(self, proposal, stdout: str, stderr: str, seconds: float, error: str = ""):
        self.proposal = proposal
        self.stdout = stdout
        self.stderr = stderr
        self.seconds = seconds
        self.error = error


def _environment() -> dict:
    env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def launch(submission: pathlib.Path, state: dict, timeout_seconds: float) -> Attempt:
    """One attempt, in its own directory, in its own process group."""
    workspace = pathlib.Path(tempfile.mkdtemp(prefix="bia_attempt_"))
    started = time.monotonic()
    process = None
    try:
        shutil.copy2(submission, workspace / "refine.py")
        (workspace / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True),
                                              encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "refine.py"],
            cwd=str(workspace),
            env=_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            out, err = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return Attempt(None, "", "", time.monotonic() - started, "attempt-timeout")
        seconds = time.monotonic() - started
        document = workspace / "proposal.json"
        if not document.is_file():
            return Attempt(None, out, err, seconds, "proposal-absent")
        try:
            payload = json.loads(document.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return Attempt(None, out, err, seconds, "proposal-unreadable")
        if not isinstance(payload, dict):
            return Attempt(None, out, err, seconds, "proposal-not-a-mapping")
        return Attempt(payload, out, err, seconds, "")
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        shutil.rmtree(workspace, ignore_errors=True)
