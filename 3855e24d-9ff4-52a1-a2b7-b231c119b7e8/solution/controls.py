"""Private negative controls for bia slot S06.

A checker that can only accept is indistinguishable from a checker that is hardcoded
to pass, so every checker in this bundle must be shown to reject as well as accept.
This script proves both halves. It runs the oracle at the smoke operating point to
obtain a genuine accepting run record, then plants one defect at a time into a copy of
that record and requires the matching checker to close on it.

It is private, it never runs in the graded path, and it consumes no accelerator.

  python3 solution/controls.py            run the smoke oracle, then the controls
  python3 solution/controls.py --record R --submission S   reuse an existing record
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

# The bundle's canonical content hash covers every file under it, so this private
# tool leaves no byte code caches behind in the tree it is auditing.
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)


def run_smoke(workdir):
    sub = os.path.join(workdir, "submission")
    logs = os.path.join(workdir, "logs")
    os.makedirs(sub, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ)
    env.update({"BIA_SMOKE": "1", "BIA_SUBMISSION_DIR": sub, "BIA_LOG_DIR": logs,
                "BIA_BUNDLE": BUNDLE})
    proc = subprocess.run(["bash", os.path.join(HERE, "solve.sh")],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit("smoke oracle failed: " + proc.stderr[-2000:])
    return os.path.join(logs, "run_record.jsonl"), os.path.join(sub, "estimator.py"), logs


def load_grader(bundle, logs, submission):
    os.environ["BIA_BUNDLE"] = bundle
    os.environ["BIA_LOG_DIR"] = logs
    os.environ["BIA_SUBMISSION"] = submission
    path = os.path.join(bundle, "tests", "grade.py")
    spec = importlib.util.spec_from_file_location("bia_grade_under_control", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_records(path):
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def make_ctx(grade, recs):
    summary = next(r for r in recs if r.get("event") == "summary")
    ctx = {"summary": summary,
           "per_seed_map": {b["seed"]: b for b in summary.get("per_seed", [])}}
    raw, _ = grade.raw_from_summary(summary)
    ctx["score_candidate"] = min(max(raw, 0.0), 1.0) if raw is not None else 0.0
    ctx["private_harness_tree"] = grade.tree_digest(grade.PRIVATE_HARNESS)
    ctx["private_fixture_tree"] = grade.tree_digest(grade.PRIVATE_FIXTURES)
    ctx["agent_harness_tree"] = grade.agent_tree(os.path.join("environment", "harness"))
    ctx["agent_fixture_tree"] = grade.agent_tree(os.path.join("environment", "fixtures"))
    return ctx


def plant_agent_harness_edit(recs, ctx):
    ctx["agent_harness_tree"] = "2" * 64


def plant_graded_run_used_another_harness(recs, ctx):
    start = next(r for r in recs if r["event"] == "run_start")
    start["harness_tree_sha256"] = "3" * 64


def plant_substrate(recs, ctx):
    recs[3]["substrate_digest"] = "0" * 64


def plant_fixture_bytes(recs, ctx):
    start = next(r for r in recs if r["event"] == "run_start")
    start["noise_fixture_sha256"] = "1" * 64


def plant_backward_count(recs, ctx):
    ctx["summary"]["backward_calls"] = int(ctx["summary"]["steps_expected"]) - 1


def plant_sentinel(recs, ctx):
    ctx["summary"]["true_grad_sentinel_hits"] = 3
    ctx["summary"]["true_grad_sentinel_names"] = ["grads"]


def plant_probe(recs, ctx):
    start = next(r for r in recs if r["event"] == "run_start")
    start["noise_probe_digest"] = "2" * 64


def plant_arm_order(recs, ctx):
    a = next(i for i, r in enumerate(recs) if r["event"] == "arm_end" and r["arm"] == "baseline")
    b = next(i for i, r in enumerate(recs) if r["event"] == "arm_start" and r["arm"] == "agent")
    recs[a], recs[b] = recs[b], recs[a]


def plant_unsustained_crossing(recs, ctx):
    seed = next(iter(ctx["per_seed_map"]))
    block = ctx["per_seed_map"][seed]
    if block.get("agent_steps") is None:
        raise SystemExit("control needs a crossing to unmake")
    for r in recs:
        if (r.get("event") == "eval" and r.get("arm") == "agent"
                and int(r.get("step", 0)) > int(block["agent_steps"])):
            r["val_loss"] = float(block["target_loss"]) + 1.0
            return
    raise SystemExit("control found no post crossing eval to raise")


def plant_estimator_digest(recs, ctx):
    for r in recs:
        if r.get("arm") == "agent":
            r["estimator_digest"] = "3" * 64
            return


def plant_score_divergence(recs, ctx):
    ctx["score_candidate"] = ctx["score_candidate"] + 0.25


CONTROLS = [
    ("control_substrate_moved", "substrate_digest_constant", plant_substrate),
    ("control_edited_fixture", "frozen_fixture_bytes_match_bound", plant_fixture_bytes),
    ("control_agent_harness_tree_edited", "frozen_fixture_bytes_match_bound",
     plant_agent_harness_edit),
    ("control_graded_run_used_another_harness", "frozen_fixture_bytes_match_bound",
     plant_graded_run_used_another_harness),
    ("control_extra_backward", "one_backward_per_step", plant_backward_count),
    ("control_sentinel_touch", "no_true_gradient_access", plant_sentinel),
    ("control_reseeded_noise", "noise_realization_reproduces", plant_probe),
    ("control_arm_order_swapped", "arm_order_and_sustained_crossing", plant_arm_order),
    ("control_crossing_given_back", "arm_order_and_sustained_crossing", plant_unsustained_crossing),
    ("control_forged_telemetry", "submission_bound_to_run", plant_estimator_digest),
    ("control_score_divergence", "score_matches_telemetry", plant_score_divergence),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--record")
    ap.add_argument("--submission")
    args = ap.parse_args(argv)

    tmp = None
    if args.record and args.submission:
        record, submission = args.record, args.submission
        logs = os.path.dirname(record)
    else:
        tmp = tempfile.mkdtemp(prefix="s06controls_")
        record, submission, logs = run_smoke(tmp)

    grade = load_grader(BUNDLE, logs, submission)
    base = read_records(record)
    bound = grade.load_bound()

    rows = []
    ctx0 = make_ctx(grade, base)
    for cid in grade.CHECKER_IDS:
        ok, why = grade.CHECKERS[cid](base, bound, dict(ctx0))
        rows.append(("accept", cid, cid, bool(ok), why))

    for name, cid, plant in CONTROLS:
        recs = copy.deepcopy(base)
        ctx = make_ctx(grade, recs)
        plant(recs, ctx)
        ok, why = grade.CHECKERS[cid](recs, bound, ctx)
        rows.append(("reject", name, cid, bool(ok), why))

    failures = []
    print("half      control                        checker                            verdict")
    for half, name, cid, ok, why in rows:
        want_pass = (half == "accept")
        good = (ok == want_pass)
        if not good:
            failures.append((half, name, cid, why))
        print("%-9s %-30s %-34s %s" % (half, name, cid, "OK" if good else "INERT_OR_WRONG"))
    print("")
    if failures:
        print("FAILED: " + json.dumps(failures))
        return 1
    print("both halves proven for all %d checkers over %d controls" %
          (len(grade.CHECKER_IDS), len(CONTROLS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
