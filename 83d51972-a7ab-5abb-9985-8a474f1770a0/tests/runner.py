#!/usr/bin/env python3
"""Isolation. The grading process never imports the submission.

The submission is data: an adjudication over a frozen corpus. Nothing it contains is
imported, exec'd or evaluated by the grading interpreter. What this module launches is the
HARNESS's own index builder, over the BUNDLE's frozen corpus and frozen index specification,
as a NEW SESSION LEADER under a small environment allowlist, with output captured and the
whole process group killed in a `finally` block whether the build succeeded, failed or timed
out.

The telemetry this returns is therefore an index the verifier built, never an index a
submission supplied. That is what makes the adjudication the verifier grades against a
quantity the verifier computed rather than a quantity the submission reported.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# The whole environment the builder process is given. Nothing else crosses.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONHASHSEED")

DEFAULT_TIMEOUT_SECONDS = 600


class RunFailed(Exception):
    """The harness build did not produce telemetry. Never a score, always a reason."""


def _environment():
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    # Fixed so dictionary iteration inside the child cannot vary between runs.
    env["PYTHONHASHSEED"] = "0"
    return env


def rebuild(bundle: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS):
    """Rebuild the index over the frozen corpus and return harness telemetry rows."""
    bundle = Path(bundle).resolve()
    builder = bundle / "environment" / "build_index.py"
    corpus = bundle / "environment" / "corpus.jsonl"
    spec = bundle / "environment" / "index_spec.json"
    for required in (builder, corpus, spec):
        if not required.is_file():
            raise RunFailed("harness input absent: " + required.as_posix())

    scratch = tempfile.mkdtemp(prefix="oer27-index-")
    process = None
    try:
        telemetry_path = Path(scratch) / "telemetry.jsonl"
        argv = [
            sys.executable,
            "-I",
            "-S",
            str(builder),
            "--corpus",
            str(corpus),
            "--spec",
            str(spec),
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
            raise RunFailed("the harness index build exceeded " + str(timeout) + " seconds")
        if process.returncode != 0:
            raise RunFailed(
                "the harness index build exited "
                + str(process.returncode)
                + ": "
                + err.decode("utf-8", "replace").strip()[:400]
            )
        if not telemetry_path.is_file():
            raise RunFailed("the harness index build produced no telemetry")
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
