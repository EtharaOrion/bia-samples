#!/usr/bin/env python3
"""Run the submission somewhere the grading process is not, and reap it.

This is the isolation boundary the whole grader-integrity design rests on. The
grading process NEVER imports the submission. Instead:

1. The submission file alone is copied into a fresh temporary directory. Nothing
   else is copied there, so nothing under `tests/` and nothing under `solution/`
   is on the submission's import path or on its filesystem walk.
2. A verifier-owned probe driver is written beside it. The driver imports the
   submission INSIDE that child process and prints one transcript to stdout.
3. The child is launched as a NEW SESSION LEADER, with `-I -S` so neither the
   user site directory nor the parent's `sys.path` leaks in, under a small
   environment allowlist, with cwd set to the temporary directory.
4. Whatever happens, the whole PROCESS GROUP is killed in a `finally` block, so
   a child that forked a helper to outlive the timeout does not survive.

What this module deliberately does NOT do: it computes no verdict, it grades
nothing, and it returns raw bytes and a status. Every judgement is made by
`tests/grade.py` afterwards, over telemetry the verifier produced. The ordering
between this module's reap and that judgement is itself graded, by
`check_truth_computed_after_submission_exit`.
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

# The whole environment the child is allowed to see. Anything not named here is
# absent from the child, including every secret the verifier holds.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONHASHSEED")

# A fixed value, so the child cannot vary its behaviour by hash seed.
FIXED_HASH_SEED = "0"

DEFAULT_TIMEOUT_SEC = 120

DRIVER = '''"""Verifier-owned probe driver. Runs inside the submission's process, never the grader's."""
import importlib.util
import json
import sys

PROBE = json.loads(sys.argv[1])

# The interpreter runs under -I, which removes the script directory from
# sys.path, so the submission is loaded from an explicit file location rather
# than by name. That is deliberate: nothing on sys.path can shadow it and it
# cannot reach anything else on sys.path either.


def main():
    spec = importlib.util.spec_from_file_location("submission_update_rule", "update_rule.py")
    update_rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(update_rule)

    rule = update_rule.build_update_rule((PROBE["width"], PROBE["width"]), {})
    params = [float(v) for v in PROBE["params0"]]
    state = {}
    rows = []
    for grads in PROBE["grads"]:
        before = list(params)
        params, state = rule.step(params, [float(g) for g in grads], state)
        params = [float(v) for v in params]
        state = state if isinstance(state, dict) else {}
        rows.append([round(float(a) - float(b) + 0.0, PROBE["quantum"]) for b, a in zip(before, params)])
    sys.stdout.write(json.dumps({"transcript": rows}, sort_keys=True, separators=(",", ":")))
    return 0


sys.exit(main())
'''


def _env() -> dict:
    out = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    out["PYTHONHASHSEED"] = FIXED_HASH_SEED
    out.setdefault("PATH", "/usr/bin:/bin")
    return out


def run_probe(submission: Path, probe_spec: dict, timeout_sec: int = DEFAULT_TIMEOUT_SEC) -> dict:
    """Copy the submission alone into a scratch dir, drive it, reap the group."""
    scratch = Path(tempfile.mkdtemp(prefix="oer05-submission-"))
    process = None
    try:
        shutil.copy2(str(submission), str(scratch / "update_rule.py"))
        (scratch / "probe_driver.py").write_text(DRIVER, encoding="utf-8")
        payload = json.dumps(probe_spec, sort_keys=True, separators=(",", ":"))
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "probe_driver.py", payload],
            cwd=str(scratch),
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = process.communicate(timeout=timeout_sec)
            status = process.returncode
        except subprocess.TimeoutExpired:
            out, err, status = b"", b"submission-timeout", None
        return {
            "status": status,
            "stdout": out.decode("utf-8", errors="replace"),
            "stderr": err.decode("utf-8", errors="replace"),
            "scratch": str(scratch),
        }
    finally:
        # Every exit path reaps the whole process group, including the paths
        # taken by an exception raised above.
        if process is not None:
            _reap(process)
        shutil.rmtree(scratch, ignore_errors=True)


def _reap(process) -> None:
    try:
        group = os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError, OSError):
        group = None
    if group is not None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(group, sig)
            except (ProcessLookupError, PermissionError, OSError):
                break
    try:
        process.wait(timeout=5)
    except Exception:
        pass


def order_marks(base_seq: int = 1) -> list:
    """The ordering marks this module contributes to the verifier's order log."""
    return [
        {"event": "submission-exit", "seq": base_seq},
        {"event": "process-group-killed", "seq": base_seq + 1},
    ]
