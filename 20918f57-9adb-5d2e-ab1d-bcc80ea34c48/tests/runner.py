#!/usr/bin/env python3
"""Isolation. The grading process never imports the submission.

The submission is data: one serving configuration. It is copied ALONE into a fresh temporary
directory, and the harness-owned simulator is launched against it as a NEW SESSION LEADER
under a small environment allowlist. Output is captured, and the whole process group is
killed in a `finally` block whether the run succeeded, failed or timed out.

Nothing the submission contains is imported, exec'd or evaluated by the grading interpreter.
`tests/grade.py` imports the checkers and the bound constants and never imports this
submission, so there is no module a submission could rebind that the scoring process reaches.

The telemetry this returns is the harness's own: it is emitted by the simulator process this
module started, over the BUNDLE's frozen trace and envelope, never over a copy the submission
supplied. That is what makes the graded throughput and the graded p99 quantities the verifier
computed rather than quantities the submission reported.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# The whole environment the simulator process is given. Nothing else crosses.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONHASHSEED")

DEFAULT_TIMEOUT_SECONDS = 300


class RunFailed(Exception):
    """The harness run did not produce telemetry. Never a score, always a reason."""


def _environment():
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    # Fixed so dictionary iteration inside the child cannot vary between runs.
    env["PYTHONHASHSEED"] = "0"
    return env


def replay(bundle: Path, config: dict, timeout: int = DEFAULT_TIMEOUT_SECONDS):
    """Re-run the frozen trace under one configuration and return harness telemetry rows."""
    # Resolved because the child runs with its cwd inside the scratch directory, so a
    # relative bundle path would resolve against the wrong root.
    bundle = Path(bundle).resolve()
    simulator = bundle / "environment" / "serving_sim.py"
    trace = bundle / "environment" / "trace.json"
    envelope = bundle / "environment" / "envelope.json"
    for required in (simulator, trace, envelope):
        if not required.is_file():
            raise RunFailed("harness input absent: " + required.as_posix())

    scratch = tempfile.mkdtemp(prefix="oer24-replay-")
    process = None
    try:
        config_path = Path(scratch) / "config.json"
        config_path.write_text(
            json.dumps(config if isinstance(config, dict) else {}, sort_keys=True), encoding="utf-8"
        )
        telemetry_path = Path(scratch) / "telemetry.jsonl"
        argv = [
            sys.executable,
            "-I",
            "-S",
            str(simulator),
            "--trace",
            str(trace),
            "--envelope",
            str(envelope),
            "--config",
            str(config_path),
            "--telemetry",
            str(telemetry_path),
        ]
        process = subprocess.Popen(
            argv,
            cwd=scratch,
            env=_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RunFailed("the harness replay exceeded " + str(timeout) + " seconds")
        if process.returncode != 0:
            raise RunFailed(
                "the harness replay exited "
                + str(process.returncode)
                + ": "
                + err.decode("utf-8", "replace").strip()[:400]
            )
        if not telemetry_path.is_file():
            raise RunFailed("the harness replay produced no telemetry")
        rows = []
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
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
        for entry in sorted(Path(scratch).rglob("*"), reverse=True):
            try:
                entry.unlink() if entry.is_file() else entry.rmdir()
            except OSError:
                pass
        try:
            Path(scratch).rmdir()
        except OSError:
            pass
