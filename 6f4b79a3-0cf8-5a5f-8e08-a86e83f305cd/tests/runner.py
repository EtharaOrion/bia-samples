#!/usr/bin/env python3
"""The isolation boundary. The grading process never imports the submission.

The submission is copied ALONE into a fresh temporary directory, launched as a new
session leader under a small environment allowlist, its output captured, and the
whole process group killed in a `finally` block. Nothing the submission does can
reach a module the grading interpreter holds, because the two never share an
interpreter.

Killing by process group rather than by pid is the part that matters. A submission
that forks and returns leaves its children alive; a wait on the parent alone would
report a clean exit while a child kept running against the shared filesystem, and
the next run would then be graded against state the previous one was still
writing.

This module also owns telemetry acquisition. Telemetry is a VERIFIER-OWNED record:
it is written by tests/harness.py inside the verifier's own process from the
verifier's own evaluations, and it is never a document the submission produced. If
no telemetry exists, this module returns a fail-closed record whose `launched`
flag is false, so the EFFECT checker attributes the zero rather than the grader
inventing a number.
"""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "harness.py"

# The whole environment a submitted recipe is allowed to see. Everything else is
# stripped, including every BIA_ variable, so no path into the grading tree and no
# bound value is reachable from inside the child.
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONHASHSEED")

RUN_TIMEOUT_SEC = float(os.environ.get("BIA_RUN_TIMEOUT_SEC", "900"))


def sha256_of(path) -> str:
    location = Path(path)
    if not location.is_file():
        return ""
    return hashlib.sha256(location.read_bytes()).hexdigest()


def _child_env() -> dict:
    env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = env.get("PYTHONHASHSEED", "0")
    return env


def run_isolated(submission, argv_tail=(), timeout=None):
    """Copy the submission alone into scratch, run it as its own session, reap it.

    Returns a dict describing the run. Nothing from the child is trusted as a
    graded number; the return value exists so the harness can record what happened
    and so a failure is attributable rather than silent.
    """
    source = Path(submission)
    limit = RUN_TIMEOUT_SEC if timeout is None else float(timeout)
    scratch = tempfile.mkdtemp(prefix="oer01-run-")
    process = None
    try:
        target = Path(scratch) / source.name
        shutil.copy2(source, target)
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", str(target), *[str(item) for item in argv_tail]],
            cwd=scratch,
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = process.communicate(timeout=limit)
            status = "completed"
        except subprocess.TimeoutExpired:
            out, err, status = b"", b"", "timeout"
        return {
            "status": status,
            "returncode": process.returncode,
            "stdout": out.decode("utf-8", "replace") if out else "",
            "stderr": err.decode("utf-8", "replace") if err else "",
            "scratch": scratch,
        }
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait(timeout=30)
        shutil.rmtree(scratch, ignore_errors=True)


def _fail_closed(reason: str, screen_record=None) -> dict:
    """The record a missing measurement produces. Never a fabricated series.

    The screen event is carried, because the screen DID run: grade.py performs it
    over bytes before this module is called at all. Dropping it would make an
    absent measurement read as a screen that never happened, which is a different
    defect with a different reason, and the run would be refused for the wrong one.
    """
    record = screen_record or {}
    return {
        "schema": "bia.oer01.telemetry/v1",
        "origin": "absent",
        "absent_reason": reason,
        "events": [
            {
                "seq": 0,
                "kind": "fingerprint-screen",
                "accelerator_seconds_spent_before": 0.0,
                "corpus_revision": record.get("corpus_revision"),
            }
        ],
        "run": {"launched": False, "seeds": [], "halt_step": None, "halt_cause": "", "run_directory_owner": "verifier"},
        "evaluations": [],
        "evaluations_per_seed": {},
        "graded_series": [],
        "graded_loss_provenance": "",
        "checkpoint_custody": "",
        "frozen_axes": {},
    }


def telemetry(path, submission, screen_record, admin, frozen) -> dict:
    """Return the verifier-owned telemetry record for this graded run.

    Resolution order, and each step is deliberate:

    1. A telemetry document already written by the verifier's own harness at the
       verifier-owned path. The harness writes it; the submission cannot, because
       the path is not in the child's environment allowlist.
    2. Otherwise, invoke tests/harness.py to produce one.
    3. Otherwise, fail closed with `launched: false`. A grader that invented a
       series here would be manufacturing the measurement the reward rests on.
    """
    location = Path(path)
    if location.is_file():
        record = json.loads(location.read_text(encoding="utf-8"))
        record.setdefault("origin", "harness")
        return record
    if HARNESS.is_file():
        outcome = subprocess.run(
            [sys.executable, str(HARNESS), str(submission), str(location)],
            capture_output=True,
            text=True,
            check=False,
        )
        if location.is_file():
            record = json.loads(location.read_text(encoding="utf-8"))
            record.setdefault("origin", "harness")
            return record
        return _fail_closed("harness-produced-no-telemetry:" + (outcome.stderr or "").strip()[:200], screen_record)
    return _fail_closed("harness-absent", screen_record)
