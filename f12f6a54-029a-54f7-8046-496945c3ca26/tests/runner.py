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

IT ALSO OWNS THE RUN. The delivered version of this module READ `/telemetry/run_record.json`
and, finding nothing there, recorded empty training, corpus, evaluation, checkpoint and ladder
blocks. Nothing in the bundle ever wrote that file: `environment/frozen_train.py` described the
record it would write and carried no trainer at all. Six of the nine checkers therefore failed
on every run and the verifier published `graded-loss-not-verifier-computed` with reward 0.0.
This module now DRIVES the frozen trainer for all three arms and writes that record itself.

That is also a containment fix and not only a completeness one. The agent image creates
/telemetry, so a submission could have left a finished, favourable run record there and had it
graded as the verifier's own measurement. The record is overwritten here on every run, so what
a checker reads is always evidence this process produced.
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

FROZEN_TRAIN = ENVIRONMENT / "frozen_train.py"
# The harness-owned reference recipe. It sits verifier-side, so it is absent from the agent
# surface by construction, and solution/recompute.py generates it and solution/reference.py
# from the one `reference_recipe` block, so the floor and the oracle can never drift apart.
REFERENCE_RECIPE = HERE / "reference_recipe.py"

QUANTISATION_CEILING = 8

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


def _quantise(weight) -> int:
    """The single quantisation rule. The screen, the corpus and the trainer all use it."""
    return max(0, min(QUANTISATION_CEILING, int(round(float(weight) * float(QUANTISATION_CEILING)))))


def _screen_matches(fingerprint: dict, exclusion: dict) -> bool:
    """Whether the pinned exclusion set catches this recipe, at either level.

    This is an ECONOMICS decision and not a verdict: it decides whether the accelerator is
    spent, and instruction.md binds that a rejected replay costs zero accelerator time. The
    verdict stays with checkers.check_recipe_not_a_replay, which re-derives it from the
    recorded fingerprint without consulting anything decided here.
    """
    vector = fingerprint.get("selection_vector")
    if not vector or not exclusion.get("pinned"):
        return False
    digest = fingerprint.get("digest")
    try:
        floor = int(exclusion.get("l1_floor"))
    except (TypeError, ValueError):
        return False
    for row in exclusion.get("entries") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("digest", "")) == str(digest):
            return True
        other = row.get("selection_vector") or []
        if len(other) == len(vector):
            if sum(abs(int(a) - int(b)) for a, b in zip(vector, other)) < floor:
                return True
    return False


def _reference_weights(documents: list) -> dict:
    """The harness-owned reference recipe's quantised weight over every raw-pool document."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("oer10_reference_recipe", REFERENCE_RECIPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {str(document.get("id")): _quantise(module.weight(document)) for document in documents}


def _reported_loss(plan) -> dict:
    """Whatever the submission said about its own loss, carried so a checker can prove it
    was not the graded number."""
    if not isinstance(plan, dict):
        return None
    for key in ("reported_loss", "validation_loss", "loss"):
        if key in plan:
            return {"loss": plan[key], "smoothing": plan.get("smoothing"), "step": plan.get("step")}
    return None


def train_arms(documents: list, rows, plan, held_split_id) -> dict:
    """Invoke the frozen trainer for the graded, control and reference arms.

    The record is written by environment/frozen_train.py, which owns every graded number, and
    it is written to a path this process chooses and overwrites, so a run record a submission
    left behind is never what gets read back.
    """
    submission_weights = {str(name): int(weight) for name, weight in (rows or [])}
    control_weights = {str(document.get("id")): 1 for document in documents}
    arms = {
        "documents": documents,
        "submission": submission_weights,
        "control": control_weights,
        "reference": _reference_weights(documents),
        "submission_reported": _reported_loss(plan),
        "bound_held_out_split_id": held_split_id,
    }

    scratch = tempfile.mkdtemp(prefix="oer10-arms-")
    try:
        arms_path = Path(scratch) / "arms.json"
        arms_path.write_text(json.dumps(arms, sort_keys=True), encoding="utf-8")
        target = Path(os.environ.get("OER10_RUN_RECORD", "/telemetry/run_record.json"))
        if target.exists():
            target.unlink()
        outcome = subprocess.run(
            [sys.executable, str(FROZEN_TRAIN), "--arms", str(arms_path), "--out", str(target)],
            capture_output=True,
            text=True,
        )
        if outcome.returncode != 0:
            return None
        return _read_json(target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def observe(submission: Path) -> dict:
    """Walk the frozen stage sequence and record what each stage produced."""
    stages = [{"seq": 1, "stage": "assemble-agent-surface"}]

    exclusion = _read_json(ENVIRONMENT / "exclusion_set.json") or {}
    stages.append({"seq": 2, "stage": "pin-exclusion-set"})

    probe = (_read_json(ENVIRONMENT / "probe_pool.json") or {}).get("docs") or []
    rows, plan, status = execute(submission, probe)
    stages.append({"seq": 3, "stage": "recipe-fingerprint-screen"})

    fingerprint = _fingerprint(rows, exclusion)
    record = {
        "schema": TELEMETRY_SCHEMA,
        "stages": stages,
        "submission_status": status,
        "declared_plan": plan,
        "recipe_fingerprint": fingerprint,
        "exclusion_set": _exclusion_block(exclusion),
    }

    held = _read_json(ENVIRONMENT / "held_out_split.json") or {}
    empty = {
        "training": {},
        "corpus": {
            "held_out_split_id": held.get("id"),
            "bound_held_out_split_id": None,
            "held_out_doc_ids": held.get("doc_ids") or [],
            "fed_doc_ids": [],
        },
        "evaluation": {},
        "checkpoints": [],
        "ladder": {},
    }

    # Stage 3 of 6 decides whether stage 5 ever happens. A submission that did not run, and a
    # recipe the pinned exclusion set catches, both stop here at zero accelerator cost. That
    # is a recordable state, not a silent one: the checkers that need a training record fail
    # with their own reasons.
    if status != "ok" or not rows or _screen_matches(fingerprint, exclusion):
        record.update(empty)
        return record

    raw_rows, _plan, raw_status = execute(
        submission, (_read_json(ENVIRONMENT / "raw_pool.json") or {}).get("docs") or []
    )
    if raw_status != "ok":
        record.update(empty)
        return record
    stages.append({"seq": 4, "stage": "curate-corpus"})

    documents = (_read_json(ENVIRONMENT / "raw_pool.json") or {}).get("docs") or []
    run = train_arms(documents, raw_rows, plan, held.get("id"))
    if run is None:
        record.update(empty)
        return record
    stages.append({"seq": 5, "stage": "train"})
    stages.append({"seq": 6, "stage": "verifier-evaluation"})

    fed = [str(name) for name, weight in (raw_rows or []) if int(weight) >= 1]
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
