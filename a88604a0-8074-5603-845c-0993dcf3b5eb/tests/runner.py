#!/usr/bin/env python3
"""Execute the submission without importing it, and never let it outlive the run.

The grading process must not import the submission, because an import runs the
submission's bytes inside the process that decides its score. This runner copies the
submission alone into a fresh temporary directory, launches it as a NEW SESSION
LEADER under a small environment allowlist, captures its output, and kills the whole
process group in a `finally` block so a backgrounded child cannot survive the run and
keep writing.

Nothing the submission prints reaches a graded quantity. The only thing carried
forward is the submission.json artifact, which is inert data that tests/grade.py
re-reads and re-derives every graded number from.
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

# A small allowlist. Nothing else from the verifier's environment crosses.
ENVIRONMENT_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONHASHSEED")

SUBMISSION_NAME = "submission.json"


def _environment(workspace: Path) -> dict:
    env = {key: os.environ[key] for key in ENVIRONMENT_ALLOWLIST if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONHASHSEED"] = "0"
    env["OER22_WORKSPACE"] = str(workspace)
    env["OER22_SUBMISSION"] = str(workspace / SUBMISSION_NAME)
    return env


def run_submission(script: Path, workspace: Path, timeout: float = 300.0) -> dict:
    """Launch the submission entry point in isolation and reap its whole group."""
    script = Path(script)
    workspace = Path(workspace)
    if not script.is_file():
        return {"launched": False, "reason": "submission-entry-point-absent", "returncode": None}
    holder = tempfile.mkdtemp(prefix="oer22-submission-")
    process = None
    group = None
    try:
        staged = Path(holder) / script.name
        shutil.copy2(script, staged)
        staged.chmod(0o755)
        process = subprocess.Popen(
            ["/usr/bin/env", "bash", str(staged)],
            cwd=holder,
            env=_environment(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        # start_new_session=True makes the child its own group leader, so the group id
        # equals its pid. It is captured HERE, before communicate() reaps the leader:
        # once the leader is gone os.getpgid raises, and a backgrounded grandchild would
        # then never be signalled and would outlive the run.
        group = process.pid
        try:
            captured = process.communicate(timeout=timeout)[0]
            code = process.returncode
        except subprocess.TimeoutExpired:
            captured, code = "", None
        return {
            "launched": True,
            "reason": "submission-executed" if code == 0 else "submission-nonzero-exit",
            "returncode": code,
            "stdout_bytes": len(captured or ""),
        }
    finally:
        if group is not None:
            # The whole process group, not just the leader, so a backgrounded child
            # cannot survive the run and keep writing into the workspace.
            try:
                os.killpg(group, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        shutil.rmtree(holder, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--entry", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if args.entry:
        entry = Path(args.entry)
    else:
        # The agent surface leaves its entry point at the workspace root. The bundle's own
        # reference ships at solution/solve.sh, so a control that submits the reference
        # unchanged resolves there. Without this fallback the reference is reported
        # submission-entry-point-absent and scores a zero it did not earn.
        entry = workspace / "solve.sh"
        if not entry.is_file():
            entry = workspace / "solution" / "solve.sh"
    outcome = run_submission(entry, workspace)
    outcome["submission_artifact_present"] = (workspace / SUBMISSION_NAME).is_file()
    text = json.dumps(outcome, indent=2, sort_keys=True) + "\n"
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
