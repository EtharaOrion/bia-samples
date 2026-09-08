"""Launch the submission out-of-process, and never inside the grading process.

The grading process does not import the submission. It copies the submitted file
alone into a fresh temporary directory, launches it as a NEW SESSION LEADER under
a small environment allowlist, captures its output, and kills the whole process
group in a `finally` block whatever happens. Importing it would put submission
bytes inside the interpreter that decides the score, which makes every checker in
tests/checkers.py an assertion the submission could rewrite.

Nothing here scores anything. It produces a run record: what was launched, how it
ended, how many optimizer steps the pinned harness observed, and where the
harness-owned weight ledger landed. Every number the grader later grades is read
off the verifier's own filesystem, never off this process's stdout capture. The
captured stdout is retained only so a divergence between what the submission
claimed and what the verifier measured can be graded rather than absorbed.
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
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# The environment the child sees. Everything else is stripped, so a secret, a
# proxy setting or a locale cannot reach the run and change what it does.
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")

# The verifier surface has no egress. This is bound in task.toml under
# [verifier] egress = "denied" and re-asserted here so the child cannot inherit a
# proxy the parent happened to carry.
EGRESS_DENIED_ENV = {
    "http_proxy": "",
    "https_proxy": "",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "no_proxy": "*",
    "NO_PROXY": "*",
}

# The file the pinned harness writes inside the verifier-owned run directory.
# It is harness output, not submission output, and it is the only step ledger the
# grader reads.
LEDGER_NAME = "harness_weight_ledger.json"

# Terminal statuses, mirrored from tests/checkers.py RUN_STATUSES. A status
# outside that set is unreadable rather than lenient.
STATUS_COMPLETED = "completed"
STATUS_TERMINATED_EARLY = "terminated-early"
STATUS_CRASHED = "crashed"
STATUS_TIMEOUT = "timeout"


@dataclass
class RunRecord:
    """What one launch of one submission actually did, as the verifier saw it."""

    status: str
    exit_code: int
    stopped_by: str
    schedule_length: int
    steps_executed: int
    run_dir: str
    ledger_present: bool
    stdout_bytes: int
    stderr_tail: str = ""
    submission_sha256: str = ""
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def child_environment() -> Dict[str, str]:
    env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    env.update(EGRESS_DENIED_ENV)
    env.setdefault("PYTHONHASHSEED", "0")
    env["BIA_HARNESS_LEDGER"] = LEDGER_NAME
    return env


def read_ledger(run_dir: pathlib.Path) -> Dict[str, Any]:
    """The harness-owned ledger, or an empty mapping when the harness wrote none.

    An empty mapping is returned rather than an exception, because a missing
    ledger is a gradable fact about the run and not an infrastructure error. It
    reaches the grader as zero evaluations, which the checker chain grades as a
    failure with a reason.
    """
    path = run_dir / LEDGER_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def launch(
    submission: pathlib.Path,
    run_root: pathlib.Path,
    timeout_sec: float,
    schedule_hint: int = 0,
) -> RunRecord:
    """Copy, launch, capture, and kill the process group. Never import."""
    import hashlib

    run_root.mkdir(parents=True, exist_ok=True)
    digest = ""
    if submission.is_file():
        digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    else:
        return RunRecord(
            status=STATUS_CRASHED,
            exit_code=-1,
            stopped_by="submission-absent",
            schedule_length=int(schedule_hint),
            steps_executed=0,
            run_dir=str(run_root),
            ledger_present=False,
            stdout_bytes=0,
            stderr_tail="no submission at " + str(submission),
            notes=["submission-absent"],
        )

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="bia-oer03-", dir=str(run_root)))
    target = workdir / "submission.py"
    shutil.copyfile(submission, target)

    process: Optional[subprocess.Popen] = None
    status, exit_code, stopped_by = STATUS_CRASHED, -1, "launch-failed"
    out, err = b"", b""
    try:
        process = subprocess.Popen(
            [sys.executable, str(target)],
            cwd=str(workdir),
            env=child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = process.communicate(timeout=timeout_sec)
            exit_code = int(process.returncode)
            status = STATUS_COMPLETED if exit_code == 0 else STATUS_CRASHED
            stopped_by = "schedule-end" if exit_code == 0 else "nonzero-exit"
        except subprocess.TimeoutExpired:
            status, exit_code, stopped_by = STATUS_TIMEOUT, -9, "verifier-timeout"
            out, err = b"", b"timed out after " + str(timeout_sec).encode() + b" seconds"
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

    ledger = read_ledger(workdir)
    steps = 0
    try:
        steps = int(ledger.get("steps_executed", 0))
    except (TypeError, ValueError):
        steps = 0
    declared = int(schedule_hint)
    try:
        declared = int(ledger.get("schedule_length", schedule_hint))
    except (TypeError, ValueError):
        declared = int(schedule_hint)
    if status == STATUS_COMPLETED and 0 < steps < declared:
        status, stopped_by = STATUS_TERMINATED_EARLY, "submission-early-stop"

    return RunRecord(
        status=status,
        exit_code=exit_code,
        stopped_by=stopped_by,
        schedule_length=declared,
        steps_executed=steps,
        run_dir=str(workdir),
        ledger_present=bool(ledger),
        stdout_bytes=len(out),
        stderr_tail=err.decode("utf-8", errors="replace")[-512:],
        submission_sha256=digest,
        notes=[],
    )
