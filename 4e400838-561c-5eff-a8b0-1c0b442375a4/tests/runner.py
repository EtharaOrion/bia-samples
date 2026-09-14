"""Launch the submission in isolation and take back exactly one artifact from it.

The grading process never imports the submission. This module copies the
submission ALONE into a fresh temporary directory, launches it as a new session
leader under a small environment allowlist, captures its output, and kills the
whole process group in a `finally` block whether it succeeded, failed, hung or
was interrupted. A submission that forks, daemonizes or ignores SIGTERM still
loses its whole group.

The one artifact taken back is `scheme.json`, a quantization SPECIFICATION. No
number the submission printed ever reaches the grader: stdout is captured for the
trajectory record and is never parsed for a metric.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

ARTIFACT = "scheme.json"

# The whole environment the submission is launched under. Nothing else crosses.
ENVIRONMENT_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL")

DEFAULT_TIMEOUT_SECONDS = 300


class SubmissionError(RuntimeError):
    """The submission produced no readable scheme. Graded, never crashed on."""


def _command(target: Path) -> list:
    if target.suffix == ".py":
        return ["python3", target.name]
    if target.suffix in (".sh", ""):
        return ["bash", target.name]
    return ["bash", target.name]


def run_submission(submission: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    submission = Path(submission).resolve()
    if not submission.is_file():
        raise SubmissionError("submission-absent")
    workspace = tempfile.mkdtemp(prefix="oer21-submission-")
    process = None
    try:
        staged = Path(workspace) / submission.name
        shutil.copy2(submission, staged)
        environment = {name: os.environ[name] for name in ENVIRONMENT_ALLOWLIST if name in os.environ}
        environment["OUT_DIR"] = workspace
        process = subprocess.Popen(
            _command(staged),
            cwd=workspace,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            code = process.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, code = "", "submission-timeout", None
        artifact = Path(workspace) / ARTIFACT
        if not artifact.is_file():
            raise SubmissionError("submission-emitted-no-scheme")
        try:
            scheme = json.loads(artifact.read_text(encoding="utf-8"))
        except ValueError:
            raise SubmissionError("submission-scheme-unreadable") from None
        if not isinstance(scheme, dict):
            raise SubmissionError("submission-scheme-unreadable")
        return {
            "scheme": scheme,
            "exit_code": code,
            "stdout_bytes": len(stdout or ""),
            "stderr_bytes": len(stderr or ""),
        }
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        shutil.rmtree(workspace, ignore_errors=True)
