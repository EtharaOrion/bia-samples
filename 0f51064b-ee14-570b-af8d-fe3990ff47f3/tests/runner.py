#!/usr/bin/env python3
"""Execute the submission in isolation. The grading process never imports it.

The submission is copied alone into a fresh temporary directory, launched as a new session
leader under a small environment allowlist, its output captured, and the whole process group
killed in a `finally` block whether it exited, hung or raised. Nothing it writes reaches the
grader's import path, and nothing it prints is read as a graded number.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# The whole environment the submission is given. Anything not on this list is not inherited.
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONHASHSEED", "OER_SLOT", "OER_FAMILY")

DEFAULT_TIMEOUT_SECONDS = 420  # the bound per-attempt budget, 0.12 h = 7.2 min


def child_environment() -> dict:
    env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_submission(entry: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    workspace = Path(tempfile.mkdtemp(prefix="oer09-submission-"))
    process = None
    try:
        staged = workspace / entry.name
        shutil.copy2(entry, staged)
        staged.chmod(0o755)
        process = subprocess.Popen(
            ["/bin/bash", str(staged)],
            cwd=str(workspace),
            env=child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        try:
            out, err = process.communicate(timeout=timeout)
            code = process.returncode
        except subprocess.TimeoutExpired:
            out, err, code = "", "submission exceeded the bound per-attempt timeout", 124
        return {"exit_code": code, "stdout": out, "stderr": err, "graded": False}
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        shutil.rmtree(workspace, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="run the submission in isolation")
    parser.add_argument("--entry", required=True)
    parser.add_argument("--record", default="/logs/verifier/submission_run.json")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    entry = Path(args.entry)
    if not entry.is_file():
        record = {"exit_code": 127, "stdout": "", "stderr": "submission entry absent", "graded": False}
    else:
        record = run_submission(entry, args.timeout)

    target = Path(args.record)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The exit code of the submission is recorded and deliberately not propagated: a run that
    # failed is graded, with a reason, rather than reported as an infrastructure fault.
    return 0


if __name__ == "__main__":
    sys.exit(main())
