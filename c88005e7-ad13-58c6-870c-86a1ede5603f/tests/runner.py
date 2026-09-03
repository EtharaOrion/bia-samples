"""Run the submission in isolation. The grading process never imports it.

The submission is a script. It is copied ALONE into a fresh temporary directory beside a
verifier-owned copy of the harness, launched as a new session leader under a small
environment allowlist, and its whole process group is killed in a `finally` block whatever
happens. Nothing it defines ever enters the grading interpreter, so it cannot monkeypatch
a checker, a percentile, or the harness it is measured by.

The harness the submission runs against is the VERIFIER'S copy, materialised here from
tests/fixtures/substrate.json. That is what makes the resulting telemetry harness-owned:
the submission never supplies the serving loop, the trace, the envelope or the clock.
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

SESSION_NAME = "session.json"
SUBMISSION_NAME = "search.py"
HARNESS_NAME = "serve_sim.py"

# Everything the submission is allowed to see. A submission that needs a secret to serve a
# frozen trace is not serving the frozen trace.
ENVIRONMENT_ALLOWLIST = ("PATH", "LC_ALL", "LANG", "PYTHONHASHSEED")


def materialise_substrate(substrate: dict, target: Path) -> Path:
    """Write the verifier's own frozen trace, envelope and objective into a run directory."""
    target.mkdir(parents=True, exist_ok=True)
    for name in ("trace", "hardware", "objective"):
        (target / (name + ".json")).write_text(
            json.dumps(substrate[name], sort_keys=True, separators=(",", ":"), ensure_ascii=True),
            encoding="utf-8",
        )
    return target


def run_submission(submission: Path, harness: Path, substrate_dir: Path, workspace: Path, seconds: int = 900) -> dict:
    """Execute the submission once and return the session document it produced."""
    scratch = Path(tempfile.mkdtemp(prefix="oer23-", dir=str(workspace)))
    outcome = {"present": False, "session": None, "stdout": "", "stderr": "", "reason": "session-absent"}
    if not Path(submission).is_file():
        outcome["reason"] = "submission-absent"
        return outcome
    shutil.copy2(str(submission), str(scratch / SUBMISSION_NAME))
    shutil.copy2(str(harness), str(scratch / HARNESS_NAME))
    session_path = scratch / SESSION_NAME

    environment = {key: os.environ[key] for key in ENVIRONMENT_ALLOWLIST if key in os.environ}
    environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    environment["HOME"] = str(scratch)
    environment["LC_ALL"] = "C"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OER23_ENVIRONMENT"] = str(substrate_dir)
    environment["OER23_SESSION_OUT"] = str(session_path)

    process = subprocess.Popen(
        [sys.executable, SUBMISSION_NAME],
        cwd=str(scratch),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=seconds)
        outcome["stdout"] = stdout.decode("utf-8", "replace")[-4000:]
        outcome["stderr"] = stderr.decode("utf-8", "replace")[-4000:]
    except subprocess.TimeoutExpired:
        outcome["reason"] = "submission-timed-out"
    finally:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass

    if session_path.is_file():
        try:
            outcome["session"] = json.loads(session_path.read_text(encoding="utf-8"))
            outcome["present"] = True
            outcome["reason"] = ""
        except ValueError:
            outcome["reason"] = "session-unreadable"
    return outcome
