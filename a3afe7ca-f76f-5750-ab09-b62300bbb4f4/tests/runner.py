"""Run the submission in isolation. The grading process never imports it.

Everything in here exists so that one sentence stays true. A submission imported into
the grading process can monkey-patch a checker, replace `math.fsum`, rewrite a handle
in memory, or register an `atexit` hook that edits the reward. So the submission is
never imported: it is copied alone into a fresh temporary directory, launched as a new
session leader in a separate process group under a small environment allowlist, its
output is captured as bytes, and the whole process group is killed in a `finally` block
whether it exited, hung, or crashed.

This module starts a process on purpose, which is why it is separate from
`tests/checkers.py`. Nothing in `checkers.py` starts anything.
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

# The whole environment the submission sees. Anything not named here is absent rather
# than inherited, so a secret in the verifier's environment cannot reach the submission
# and a value the submission sets cannot reach the grader.
ENVIRONMENT_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "OER11_CORPUS_ROOT", "OER11_HARNESS_LOGS")

# A submission that never terminates is a submission that produced nothing. The bound
# is a bound on this runner, not a grading threshold, and it is passed in by the caller.
DEFAULT_WALL_BOUND_SECONDS = 900


def _allowed_environment() -> dict:
    return {name: os.environ[name] for name in ENVIRONMENT_ALLOWLIST if name in os.environ}


def run_submission(submission: Path, argv: list, bound_seconds: int = DEFAULT_WALL_BOUND_SECONDS) -> dict:
    """Copy the submission alone into a scratch tree, run it, kill its whole group.

    Returns a record of what was observed. The record is evidence about the run and is
    never a score: no field of it reaches the reward, and `tests/checkers.py` never
    reads it.
    """
    scratch = Path(tempfile.mkdtemp(prefix="oer11-submission-"))
    process = None
    try:
        target = scratch / submission.name
        if submission.is_dir():
            shutil.copytree(submission, target, symlinks=False)
        else:
            shutil.copy2(submission, target)

        process = subprocess.Popen(
            [sys.executable, str(target)] + list(argv),
            cwd=str(scratch),
            env=_allowed_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = process.communicate(timeout=bound_seconds)
            code = process.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            out, err, code, timed_out = b"", b"", None, True
        return {
            "exit_code": code,
            "timed_out": timed_out,
            "stdout_bytes": len(out or b""),
            "stderr_bytes": len(err or b""),
            "note": "captured as bytes and never parsed for a score; a number in this stream reaches no graded path",
        }
    finally:
        if process is not None and process.pid:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv: list) -> int:
    if not argv:
        print(json.dumps({"ran": False, "reason": "no-submission-named"}, sort_keys=True))
        return 0
    record = run_submission(Path(argv[0]), argv[1:])
    record["ran"] = True
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
