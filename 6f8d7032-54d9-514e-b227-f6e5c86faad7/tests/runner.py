#!/usr/bin/env python3
"""Isolation. The grading process never imports the submission.

The submission is a kernel module, which is code, so it is never imported by the grading
interpreter. It is handed to the BUNDLE's own resume driver in a fresh temporary directory, as a
NEW SESSION LEADER, under a five-key environment allowlist, and the whole process group is
killed in a `finally` block whether the run succeeded, failed or timed out.

`tests/grade.py` imports the checkers and the bound constants and never imports this submission,
so there is no module a submission could rebind that the scoring process reaches.

The telemetry this returns is the VERIFIER's own. It is emitted by the resume process this
module started, over the BUNDLE's frozen stream and parameters, never over a copy the submission
supplied. That is what makes the graded charge a number the verifier counted, and it is also
what makes this the second half of a two-phase unit rather than a re-reading of the first half:
the resume is SEEDED from the carry the agent phase handed over, so the verifier's own run
depends on state it did not produce.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# The whole environment the resume process is given. Nothing else crosses.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONHASHSEED")

DEFAULT_TIMEOUT_SECONDS = 300


class RunFailed(Exception):
    """The resume did not produce telemetry. Never a score, always a reason."""


def _environment():
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    # Fixed so dictionary iteration inside the child cannot vary between runs.
    env["PYTHONHASHSEED"] = "0"
    return env


def resume(bundle: Path, kernel: Path, carry: dict, offset: int, timeout: int = DEFAULT_TIMEOUT_SECONDS):
    """Re-run the segment after the splice under the submitted kernel, and count the reads."""
    # Resolved because the child runs with its cwd inside the scratch directory, so a relative
    # bundle path would resolve against the wrong root.
    bundle = Path(bundle).resolve()
    driver = bundle / "environment" / "phase_b.py"
    stream = bundle / "environment" / "stream.json"
    params = bundle / "environment" / "params.json"
    for required in (driver, stream, params):
        if not required.is_file():
            raise RunFailed("harness input absent: " + required.as_posix())
    if not Path(kernel).is_file():
        raise RunFailed("no kernel at " + Path(kernel).as_posix())

    scratch = tempfile.mkdtemp(prefix="oer28-resume-")
    process = None
    try:
        carry_path = Path(scratch) / "carry.json"
        carry_path.write_text(
            json.dumps(carry if isinstance(carry, dict) else {}, sort_keys=True),
            encoding="utf-8",
        )
        telemetry_path = Path(scratch) / "resume.jsonl"
        argv = [
            sys.executable,
            "-I",
            "-S",
            str(driver),
            "--stream",
            str(stream),
            "--params",
            str(params),
            "--kernel",
            str(Path(kernel).resolve()),
            "--carry",
            str(carry_path),
            "--splice-offset",
            str(int(offset)),
            "--telemetry",
            str(telemetry_path),
        ]
        environment = _environment()
        process = subprocess.Popen(
            argv,
            cwd=scratch,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RunFailed("the resume exceeded " + str(timeout) + " seconds")
        if process.returncode != 0:
            raise RunFailed(
                "the resume exited "
                + str(process.returncode)
                + ": "
                + err.decode("utf-8", "replace").strip()[:400]
            )
        if not telemetry_path.is_file():
            raise RunFailed("the resume produced no telemetry")
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
