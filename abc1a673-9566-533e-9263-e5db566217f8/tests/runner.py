#!/usr/bin/env python3
"""Isolation. The grading process never imports the submission and never traverses the fence.

The submission is data: one JSON document. The traversal that establishes the crossing count is
performed by the HARNESS, in a separate process: `environment/fence_probe.py` is launched
against the BUILT fence state as a NEW SESSION LEADER under a small environment allowlist, its
observations are captured as a record stream, and the whole process group is killed in a
`finally` block whether the run succeeded, failed or timed out.

Nothing the submission contains is imported, exec'd or evaluated by the grading interpreter.
`tests/grade.py` imports the checkers and the bound constants and never imports this submission,
so there is no module a submission could rebind that the scoring process reaches.

The telemetry this returns is the harness's own: it is emitted by the probe process this module
started, over the fence state the IMAGE was built with, never over a copy the submission
supplied. That is what makes the crossing count and the admitted set quantities the verifier
established rather than quantities the submission reported.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# The whole environment the probe process is given. Nothing else crosses.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONHASHSEED", "OER26_FENCE_STATE")

DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_PROBE = "/opt/fence/fence_probe.py"


class RunFailed(Exception):
    """The harness probe did not produce telemetry. Never a score, always a reason."""


def _environment():
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONHASHSEED"] = "0"
    return env


def probe_path() -> Path:
    return Path(os.environ.get("OER26_FENCE_PROBE", DEFAULT_PROBE))


def observe(timeout: int = DEFAULT_TIMEOUT_SECONDS):
    """Walk the built fence state once, in an isolated subprocess, and return its record rows."""
    probe = probe_path()
    if not probe.is_file():
        raise RunFailed("the harness probe is absent: " + probe.as_posix())

    scratch = tempfile.mkdtemp(prefix="oer26-probe-")
    process = None
    try:
        telemetry_path = Path(scratch) / "telemetry.jsonl"
        argv = [sys.executable, "-I", "-S", str(probe), "--telemetry", str(telemetry_path)]
        state = os.environ.get("OER26_FENCE_STATE")
        if state:
            argv.extend(["--state", state])
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
            raise RunFailed("the harness probe exceeded " + str(timeout) + " seconds")
        if process.returncode != 0:
            raise RunFailed(
                "the harness probe exited "
                + str(process.returncode)
                + ": "
                + err.decode("utf-8", "replace").strip()[:400]
            )
        if not telemetry_path.is_file():
            raise RunFailed("the harness probe produced no telemetry")
        rows = []
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if not rows:
            raise RunFailed("the harness probe produced an empty record stream")
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
