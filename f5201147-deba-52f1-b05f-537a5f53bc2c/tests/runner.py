"""Run a submission in isolation and return what it produced. Nothing here imports it.

The grading process never imports the submission, because an import runs arbitrary bytes
inside the process that decides the score. This module instead:

  1. copies the submission ALONE into a fresh temporary directory, so it cannot read the
     checkers, the reference, the control table or the grounding source;
  2. launches it as a NEW SESSION LEADER, so the whole tree it spawns is one process group;
  3. hands it a SMALL ENVIRONMENT ALLOWLIST rather than the verifier's environment;
  4. captures stdout and stderr as bytes nobody grades; and
  5. kills the whole process group in a `finally` block, so a submission that forks and
     detaches cannot outlive the grading pass.

The submission's stdout is returned so a human can read it. It is never parsed and never
reaches a checker: `tests/harness.py` computes every graded number itself.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

#: The whole environment a submission is given. Everything else is stripped, so a submission
#: cannot read a token, a mirror URL or a path back into the verifier tree out of the
#: environment it inherits.
ENVIRONMENT_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONHASHSEED")

SUBMISSION_NAME = "submission.json"

#: Wall-clock ceiling for one submission run, in seconds. The bound per-attempt budget is
#: 0.12 hours, which is 432 seconds, and this is that number.
RUN_TIMEOUT_SECONDS = 432


@dataclass
class Run:
    """What one isolated submission run produced. No field here is a graded number."""

    ok: bool
    reason: str
    document: dict = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


def _environment() -> dict:
    row = {name: os.environ[name] for name in ENVIRONMENT_ALLOWLIST if name in os.environ}
    row.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    row["PYTHONDONTWRITEBYTECODE"] = "1"
    return row


def _read_document(path: Path) -> tuple:
    if not path.is_file():
        return None, "submission-absent"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None, "submission-unreadable"
    if not isinstance(payload, dict):
        return None, "submission-unreadable"
    return payload, ""


def load(path: Path) -> Run:
    """Read a submission document that already exists. Nothing is executed."""
    document, reason = _read_document(Path(path))
    if document is None:
        return Run(False, reason)
    return Run(True, "", document)


def execute(entry: Path, timeout: int = RUN_TIMEOUT_SECONDS) -> Run:
    """Copy the entry point alone into a scratch directory, run it, and reap its group."""
    entry = Path(entry)
    if not entry.is_file():
        return Run(False, "submission-absent")
    workspace = Path(tempfile.mkdtemp(prefix="oer14-submission-"))
    process = None
    try:
        staged = workspace / entry.name
        shutil.copy2(entry, staged)
        staged.chmod(0o755)
        target = workspace / SUBMISSION_NAME
        environment = _environment()
        environment["SUBMISSION_PATH"] = str(target)
        process = subprocess.Popen(
            [sys.executable, str(staged)] if staged.suffix == ".py" else ["/bin/bash", str(staged)],
            cwd=str(workspace),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = process.communicate(timeout=timeout)
            code = process.returncode
        except subprocess.TimeoutExpired:
            return Run(False, "submission-timed-out")
        document, reason = _read_document(target)
        if document is None:
            return Run(
                False,
                reason,
                stdout=out.decode("utf-8", "replace"),
                stderr=err.decode("utf-8", "replace"),
                exit_code=code,
            )
        return Run(
            True,
            "",
            document,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
            code,
        )
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        shutil.rmtree(workspace, ignore_errors=True)
