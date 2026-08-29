"""Execute the submission in isolation. The grading process never imports it.

An import would run submission bytes inside the grader, where they could reach
the checkers, the reference, the telemetry the verdict rests on, and the reward
carrier. So the submission is copied alone into a fresh temporary directory,
launched as a NEW SESSION LEADER under a small environment allowlist, and its
whole process group is killed in a `finally` block whether it exited, timed out,
or crashed. Nothing it prints is ever read as a graded number; its stdout is
captured for the trajectory and for nothing else.

This file writes no reward. tests/grade.py owns that, and it reads telemetry the
verifier's own process produced rather than anything captured here.
"""

import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile

# The whole environment the submission gets. Every other variable is dropped, so
# a secret, a token or a path pointing back at the grading tree cannot be read
# out of the environment by the bytes under test.
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONHASHSEED", "OER_LEDGER_DIR")

SUBMISSION = pathlib.Path(os.environ.get("OER_SUBMISSION", "/workspace/submission.py"))
TELEMETRY = pathlib.Path(os.environ.get("OER_TELEMETRY", "/logs/verifier"))
TRAIN = pathlib.Path(os.environ.get("OER_TRAIN", "/workspace/environment/train.py"))

# The per-attempt bound, 7.2 minutes, transcribed from task.toml [optimization]
# budget_hours = 0.12. It is a wall-clock kill, not a schedule the run reads.
ATTEMPT_TIMEOUT_SECONDS = 432


def isolated_copy(source, into):
    """The submission alone. No checker, no reference, no telemetry, no manifest."""
    target = pathlib.Path(into) / "submission.py"
    shutil.copyfile(str(source), str(target))
    return target


def child_environment():
    env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def execute(submission, workdir):
    """Run it under its own session and take the whole group down afterwards."""
    argv = [sys.executable, str(TRAIN), "--submission", str(submission)]
    process = None
    captured = {"returncode": None, "stdout": "", "stderr": "", "timed_out": False}
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(workdir),
            env=child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        try:
            out, err = process.communicate(timeout=ATTEMPT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            captured["timed_out"] = True
            out, err = "", "attempt exceeded the bound per-attempt wall clock"
        captured["stdout"] = out or ""
        captured["stderr"] = err or ""
        captured["returncode"] = process.returncode
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
    return captured


def main():
    TELEMETRY.mkdir(parents=True, exist_ok=True)
    if not SUBMISSION.is_file():
        (TELEMETRY / "submission_run.json").write_text(
            json.dumps({"ran": False, "detail": "no submission bytes present"}, sort_keys=True),
            encoding="utf-8",
        )
        return 0
    workdir = tempfile.mkdtemp(prefix="oer07-submission-")
    try:
        copied = isolated_copy(SUBMISSION, workdir)
        captured = execute(copied, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    captured["ran"] = True
    (TELEMETRY / "submission_run.json").write_text(
        json.dumps(captured, sort_keys=True, indent=1), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
