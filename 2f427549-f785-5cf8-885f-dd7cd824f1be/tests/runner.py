"""Run the submission in isolation. The grading process never imports it.

The generator is a foreign program, so it is copied ALONE into a fresh temporary
directory and executed there as a new session leader under a small environment
allowlist. Nothing else is placed in that directory: not the benchmark, not the
checkers, not the grading tree, not this bundle. Isolation is therefore a property of
what was assembled rather than of what a filter removed, which is the difference
between a boundary and a hope.

The process group is killed in a `finally` on every path, including the timeout path
and the exception path, because a generator that forks and returns would otherwise
leave a child alive to write into the workspace after the listing that grades it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

SUBMISSION_NAME = "generator.py"
SAMPLES_NAME = "samples.jsonl"

# The whole environment the generation process receives. A secret, a token, a path
# into this bundle and a hint about the grading tree are all absent by construction.
ENV_ALLOWLIST = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LC_ALL": "C",
    "LANG": "C",
    "HOME": "/tmp",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def _listing(root: Path) -> list:
    return sorted(
        item.relative_to(root).as_posix()
        for item in Path(root).rglob("*")
        if "__pycache__" not in item.parts
    )


def _kill_group(process) -> None:
    """Kill the whole process group. Never raises; a dead group is the good case."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def read_samples(path: Path, cap: int) -> list:
    """Parse the generated corpus. A malformed line is dropped, never guessed at."""
    rows = []
    if not Path(path).is_file():
        return rows
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "prompt" in row and "answer" in row:
                rows.append({"prompt": str(row.get("prompt")), "answer": str(row.get("answer"))})
            if len(rows) >= cap:
                break
    return rows


def reported_score(stdout: str):
    """The number the submission printed. Recorded for divergence, graded never."""
    for line in reversed(str(stdout).splitlines()):
        line = line.strip()
        if not line.startswith("REPORTED_SCORE="):
            continue
        try:
            return float(line.split("=", 1)[1].strip())
        except (TypeError, ValueError):
            return None
    return None


def reported_smoothing(stdout: str):
    for line in reversed(str(stdout).splitlines()):
        line = line.strip()
        if line.startswith("REPORTED_SMOOTHING="):
            return line.split("=", 1)[1].strip() or None
    return None


def run_submission(submission: Path, timeout_sec: float, sample_cap: int) -> dict:
    """Copy the submission alone into a fresh temp dir, run it, kill the group, report."""
    workspace = Path(tempfile.mkdtemp(prefix="oer17-generation-"))
    record = {
        "workspace": workspace.as_posix(),
        "env_allowlist": sorted(ENV_ALLOWLIST),
        "workspace_before": [],
        "workspace_after": [],
        "exit_code": None,
        "timed_out": False,
        "stdout_sha256": "",
        "samples": [],
        "reported_score": None,
        "reported_smoothing": None,
        "stderr_tail": "",
        "stdout_tail": "",
    }
    process = None
    try:
        shutil.copy2(Path(submission), workspace / SUBMISSION_NAME)
        record["workspace_before"] = _listing(workspace)
        process = subprocess.Popen(
            [sys.executable, SUBMISSION_NAME, "--out", SAMPLES_NAME],
            cwd=str(workspace),
            env=dict(ENV_ALLOWLIST),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        try:
            out, err = process.communicate(timeout=float(timeout_sec))
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            _kill_group(process)
            out, err = process.communicate()
        record["exit_code"] = process.returncode
        record["stdout_sha256"] = hashlib.sha256((out or "").encode("utf-8")).hexdigest()
        record["reported_score"] = reported_score(out or "")
        record["reported_smoothing"] = reported_smoothing(out or "")
        record["stderr_tail"] = (err or "")[-800:]
        record["stdout_tail"] = (out or "")[-800:]
        record["workspace_after"] = _listing(workspace)
        record["samples"] = read_samples(workspace / SAMPLES_NAME, int(sample_cap))
    finally:
        if process is not None:
            _kill_group(process)
        shutil.rmtree(workspace, ignore_errors=True)
    return record
