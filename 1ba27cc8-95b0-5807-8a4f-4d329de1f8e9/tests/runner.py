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

ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONHASHSEED")

class Attempt:

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
