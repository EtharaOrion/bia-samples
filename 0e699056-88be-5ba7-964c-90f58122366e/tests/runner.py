#!/usr/bin/env python3
"""Execute the submission in isolation and record what the verifier's own process observed.

This module produces evidence and computes no verdict. It never decides whether a recipe
is a replay, whether a budget was respected or what a loss was; it records the selection
the recipe produced, the corpus that was actually fed, and the training and evaluation
record the frozen harness wrote, and hands all of it to tests/grade.py.

Isolation, in the order it happens:

  1. the submission file ALONE is copied into a fresh temporary directory. No driver, no
     fixture, no grader byte and no bundle path is placed beside it.
  2. it is launched with `python3 -I -S` from that directory, as a NEW SESSION LEADER, under
     a five-key environment allowlist, with the driver passed on the command line rather
     than written next to the submission.
  3. its stdout is captured and parsed as one JSON line.
  4. the whole process group is killed in a `finally` block, so a submission that forks,
     daemonises or sleeps cannot outlive the measurement.

Stage sequence numbers are written here, by this process, which is what makes the ORDERING
checker a statement about what ran rather than about what a submission claimed. The
exclusion set is pinned at stage 2 and the fingerprint screen runs at stage 3; training
does not begin before stage 5, so a rejected replay costs zero accelerator time.
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

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
ENVIRONMENT = BUNDLE / "environment"

TELEMETRY_SCHEMA = "oer10.telemetry/v1"
TELEMETRY_PATH = "/logs/verifier/telemetry.json"

ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONHASHSEED")
SUBMISSION_TIMEOUT_SEC = 120

# The driver is passed on the command line so the temporary directory holds the
# submission and nothing else.
DRIVER = (
    "import json,sys,importlib.util;"
    "spec=importlib.util.spec_from_file_location('recipe','recipe.py');"
    "mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);"
    "docs=json.load(open(sys.argv[1]));"
    "rows=[[d['id'],max(0,min(8,int(round(float(mod.weight(d))*8.0))))] for d in docs];"
    "plan=mod.plan() if hasattr(mod,'plan') else {};"
    "print(json.dumps({'rows':rows,'plan':plan}))"
)


def _read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _child_env() -> dict:
    env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONHASHSEED"] = "0"
    return env


def execute(submission: Path, docs: list):
    """Run the submission alone over one document list. Returns (rows, plan, status)."""
    if not submission.is_file():
        return None, {}, "submission-absent"
    scratch = tempfile.mkdtemp(prefix="oer10-submission-")
    process = None
    try:
        shutil.copy2(submission, Path(scratch) / "recipe.py")
        pool = Path(scratch).parent / (Path(scratch).name + "-pool.json")
        pool.write_text(json.dumps(docs), encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", DRIVER, str(pool)],
            cwd=scratch,
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        try:
            out, _err = process.communicate(timeout=SUBMISSION_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            return None, {}, "submission-timeout"
        if process.returncode != 0:
            return None, {}, "submission-failed"
        try:
            payload = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return None, {}, "submission-output-unreadable"
        return payload.get("rows") or [], payload.get("plan") or {}, "ok"
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError:
                pass
            process.wait(timeout=10)
        shutil.rmtree(scratch, ignore_errors=True)
        stale = Path(scratch).parent / (Path(scratch).name + "-pool.json")
        if stale.exists():
            stale.unlink()


def observe(submission: Path) -> dict:
    """Walk the frozen stage sequence and record what each stage produced."""
    stages = [{"seq": 1, "stage": "assemble-agent-surface"}]

    exclusion = _read_json(ENVIRONMENT / "exclusion_set.json") or {}
    stages.append({"seq": 2, "stage": "pin-exclusion-set"})

    probe = (_read_json(ENVIRONMENT / "probe_pool.json") or {}).get("docs") or []
    rows, plan, status = execute(submission, probe)
    stages.append({"seq": 3, "stage": "recipe-fingerprint-screen"})

    record = {
        "schema": TELEMETRY_SCHEMA,
        "stages": stages,
        "submission_status": status,
        "declared_plan": plan,
        "recipe_fingerprint": _fingerprint(rows, exclusion),
        "exclusion_set": _exclusion_block(exclusion),
    }

    run = _read_json(Path(os.environ.get("OER10_RUN_RECORD", "/telemetry/run_record.json")))
    if run is None:
        # Nothing trained. That is a recordable state, not a silent one: the checkers
        # that need a training record fail with their own reasons.
        record["training"] = {}
        record["corpus"] = {}
        record["evaluation"] = {}
        record["checkpoints"] = []
        record["ladder"] = {}
        return record

    raw_rows, _plan, raw_status = execute(submission, (_read_json(ENVIRONMENT / "raw_pool.json") or {}).get("docs") or [])
    stages.append({"seq": 4, "stage": "curate-corpus"})
    stages.append({"seq": 5, "stage": "train"})
    stages.append({"seq": 6, "stage": "verifier-evaluation"})

    fed = [str(name) for name, weight in (raw_rows or []) if int(weight) >= 1] if raw_status == "ok" else []
    held = _read_json(ENVIRONMENT / "held_out_split.json") or {}
    record["training"] = run.get("training") or {}
    record["corpus"] = {
        "held_out_split_id": held.get("id"),
        "bound_held_out_split_id": run.get("bound_held_out_split_id"),
        "held_out_doc_ids": held.get("doc_ids") or [],
        "fed_doc_ids": run.get("fed_doc_ids") if run.get("fed_doc_ids") is not None else fed,
    }
    record["evaluation"] = run.get("evaluation") or {}
    record["checkpoints"] = run.get("checkpoints") or []
    record["ladder"] = run.get("ladder") or {}
    return record


def _fingerprint(rows, exclusion) -> dict:
    import hashlib

    if not rows:
        return {}
    vector = [int(weight) for _name, weight in rows]
    payload = json.dumps(vector, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "probe_pool_id": exclusion.get("probe_pool_id"),
        "probe_pool_size": len(vector),
        "selection_vector": vector,
        "digest": hashlib.sha256(payload).hexdigest(),
    }


def _exclusion_block(exclusion) -> dict:
    if not exclusion:
        return {"pinned": False, "entries": [], "size": 0}
    return {
        "pinned": bool(exclusion.get("pinned")),
        "id": exclusion.get("id"),
        "digest": exclusion.get("digest"),
        "size": exclusion.get("size"),
        "l1_floor": exclusion.get("l1_floor"),
        "entries": exclusion.get("entries") or [],
    }


def main(argv) -> int:
    submission = Path(argv[0]) if argv else Path("/workspace/submission/recipe.py")
    target = Path(argv[1]) if len(argv) > 1 else Path(TELEMETRY_PATH)
    record = observe(submission)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("telemetry written to " + str(target) + ", submission status " + record["submission_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
