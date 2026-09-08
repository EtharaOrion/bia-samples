#!/usr/bin/env python3
"""Isolation. The grading process never imports the submission and never opens the warehouse.

The submission is data: one SQL text. It is copied ALONE into a fresh temporary directory, and
the harness-owned executor is launched against it as a NEW SESSION LEADER under a small
environment allowlist. Output is captured, and the whole process group is killed in a `finally`
block whether the run succeeded, failed or timed out.

Nothing the submission contains is imported, exec'd or evaluated by the grading interpreter. The
plan text is never concatenated into another statement; it is handed to `environment/planrun.py`
whole, which opens the database read-only through a URI and installs an authorizer that refuses
every action outside the read allowlist.

The three frozen tables are dumped through the same executor, so `tests/checkers.py` receives
plain lists and never opens a database of its own. That is what lets the checker module hold to
its import allowlist and stay a pure function of recorded state.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# The whole environment the executor process is given. Nothing else crosses.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONHASHSEED")

DEFAULT_TIMEOUT_SECONDS = 120

TABLE_DUMPS = {
    "lineage": (
        "SELECT node_id, parent_id, p_alpha, p_beta, p_gamma, declared_check "
        "FROM lineage ORDER BY node_id"
    ),
    "derives": "SELECT edge_id, source_node, target_node FROM derives ORDER BY edge_id",
    "reading": "SELECT reading_id, node_id, value FROM reading ORDER BY reading_id",
}

TABLE_COLUMNS = {
    "lineage": ("node_id", "parent_id", "p_alpha", "p_beta", "p_gamma", "declared_check"),
    "derives": ("edge_id", "source_node", "target_node"),
    "reading": ("reading_id", "node_id", "value"),
}


class RunFailed(Exception):
    """The harness run did not produce a report. Never a score, always a reason."""


def _environment():
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    # Fixed so dictionary iteration inside the child cannot vary between runs.
    env["PYTHONHASHSEED"] = "0"
    return env


def execute(bundle: Path, plan_text: str, timeout: int = DEFAULT_TIMEOUT_SECONDS):
    """Run one plan through the harness executor and return its report."""
    bundle = Path(bundle).resolve()
    executor = bundle / "environment" / "planrun.py"
    warehouse = bundle / "environment" / "warehouse.db"
    for required in (executor, warehouse):
        if not required.is_file():
            raise RunFailed("harness input absent: " + required.as_posix())

    scratch = tempfile.mkdtemp(prefix="oer29-plan-")
    process = None
    try:
        plan_path = Path(scratch) / "plan.sql"
        plan_path.write_text(str(plan_text), encoding="utf-8")
        report_path = Path(scratch) / "report.json"
        argv = [
            sys.executable,
            "-I",
            "-S",
            str(executor),
            "--warehouse",
            str(warehouse),
            "--plan",
            str(plan_path),
            "--out",
            str(report_path),
            "--rows",
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
            raise RunFailed("the harness plan run exceeded " + str(timeout) + " seconds")
        if process.returncode != 0:
            raise RunFailed(
                "the harness plan run exited "
                + str(process.returncode)
                + ": "
                + err.decode("utf-8", "replace").strip()[:400]
            )
        if not report_path.is_file():
            raise RunFailed("the harness plan run produced no report")
        return json.loads(report_path.read_text(encoding="utf-8"))
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


def snapshot(bundle: Path):
    """Dump the three frozen tables through the same isolated executor."""
    tables = {}
    for name, sql in TABLE_DUMPS.items():
        report = execute(bundle, sql)
        if not report.get("ok"):
            raise RunFailed(
                "the frozen table dump for " + name + " was refused: "
                + str(report.get("reason", ""))
            )
        columns = TABLE_COLUMNS[name]
        tables[name] = [dict(zip(columns, row)) for row in report["rows"]]
    return tables
