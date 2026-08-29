"""Verifier for bia S09 multi-objective-frontier.

Reads the frozen spec and the runner-sealed frontier record, runs the six
checkers declared in tests/checkers.yaml, recomputes the reward from the raw
achieved objective values through an independent implementation, and writes
the score. Every checker is hard-pass: one failure scores exactly 0.0.

The grader never trusts the score the runner printed. It recomputes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import secrets
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from checkers.hypervolume import hypervolume_grid, reward_from_raw_points  # noqa: E402

EPS_AGREE = 1e-9
EPS_EFFECT = 1e-4

# The telemetry directory a clean run leaves behind, declared HERE in the verifier tree rather
# than read out of the record. The previous ABSENCE checker validated the directory against a
# manifest carried inside the very artifact under test, which is circular: a forged record
# declared its own clean bill of health. This set is the outside anchor that breaks the circle.
EXPECTED_TELEMETRY = frozenset({"frontier_record.json"})

FROZEN_BYTES_PIN = pathlib.Path(__file__).resolve().parent / "checkers" / "frozen_bytes.json"

RUNNER = pathlib.Path(
    os.environ.get("BIA_S09_RUNNER",
                   str(pathlib.Path(os.environ.get("BIA_S09_BUNDLE", "/workspace"))
                       / "environment" / "harness" / "run_frontier.py"))
)
SUBMISSION = pathlib.Path(
    os.environ.get("BIA_S09_SUBMISSION", "/workspace/submission/frontier.py")
)
VERIFIER_RUN_TIMEOUT = float(os.environ.get("BIA_S09_VERIFIER_RUN_TIMEOUT", "1500"))

# CONTROL-ONLY INPUT. When set, the named directory is used as the authoritative record
# instead of running the frontier, and the authoritative source is recorded as `external`. The
# origin checker requires the source to be `verifier_execution`, so this switch can only ever
# produce a zero. It exists so the rejecting half of that checker is executable under frozen
# bytes.
EXTERNAL_AUTHORITATIVE = os.environ.get("BIA_S09_AUTHORITATIVE_TELEMETRY_DIR", "")

# One-element cell holding the record the reward was read off, so emit() can publish it
# for the private reference driver without threading it through every call site.
GRADED_RECORD_PATH = [""]

# The kebab reason code each checker emits alongside the zero it explains. `detail` below is
# prose for a human reader and cannot be branched on; these are what tests/checkers.yaml
# declares as zero_reason, so the attribution the manifest claims is the one this file prints.
ZERO_REASON = {
    "frontier_cardinality": "frontier-cardinality-mismatched",
    "frozen_substrate_invariant": "frozen-substrate-violated",
    "no_unmanifested_telemetry": "unmanifested-telemetry-present",
    "runs_precede_seal": "runs-not-ordered-before-seal",
    "training_moved_live_weights": "live-weights-unmoved",
    "hypervolume_independently_agrees": "hypervolume-recomputation-diverged",
    "graded_record_authored_by_the_verifier": "graded-record-not-authored-by-the-verifier",
    "submission_ran_outside_the_measuring_process": "submission-ran-inside-the-measuring-process",
    "frozen_harness_bytes_unmodified": "frozen-harness-bytes-modified",
}


def _fail(name, kind, detail):
    return {
        "name": name,
        "kind": kind,
        "passed": False,
        "reason": ZERO_REASON[name],
        "detail": detail,
    }


def _pass(name, kind, detail):
    return {"name": name, "kind": kind, "passed": True, "reason": None, "detail": detail}


# --- VALUE ------------------------------------------------------------------


def check_frontier_cardinality(spec, record):
    kind = "VALUE"
    name = "frontier_cardinality"
    runs = record.get("runs", [])
    want = int(spec["frontier_points"])
    if record.get("proposed_points") != want:
        return _fail(name, kind, f"propose_frontier returned {record.get('proposed_points')}, required {want}")
    if len(runs) != want:
        return _fail(name, kind, f"{len(runs)} runs completed, required {want}")
    idx = sorted(r["point_index"] for r in runs)
    if idx != list(range(want)):
        return _fail(name, kind, f"point indices are {idx}, required {list(range(want))}")
    return _pass(name, kind, f"{want} completed runs with point indices {idx}")


# --- INVARIANT --------------------------------------------------------------


def check_frozen_substrate(spec, record):
    kind = "INVARIANT"
    name = "frozen_substrate_invariant"
    runs = record.get("runs", [])
    if not runs:
        return _fail(name, kind, "no runs to hold the invariant over")
    sigs = {r["architecture_signature"] for r in runs}
    if len(sigs) != 1:
        return _fail(name, kind, f"architecture signature moved across runs: {sorted(sigs)}")
    ref = record.get("untrained_reference", {}).get("architecture_signature")
    if ref is not None and ref not in sigs:
        return _fail(name, kind, "architecture signature differs from the untrained frozen reference")
    for r in runs:
        if int(r["max_steps"]) != int(spec["max_steps"]):
            return _fail(name, kind, f"run {r['point_index']} declared max_steps {r['max_steps']}")
        if int(r["forward_calls"]) != int(r["steps_taken"]):
            return _fail(
                name, kind,
                f"run {r['point_index']} made {r['forward_calls']} forward passes over {r['steps_taken']} steps",
            )
        if int(r["backward_calls"]) != int(r["steps_taken"]):
            return _fail(
                name, kind,
                f"run {r['point_index']} made {r['backward_calls']} backward passes over {r['steps_taken']} steps",
            )
        if float(r["elapsed_seconds"]) > float(spec["per_run_seconds"]) + 10.0:
            return _fail(
                name, kind,
                f"run {r['point_index']} ran {r['elapsed_seconds']:.1f}s, past the frozen per-run cap plus one step of slack",
            )
    if record.get("spec_id") != spec["spec_id"]:
        return _fail(name, kind, f"record spec_id {record.get('spec_id')} is not {spec['spec_id']}")
    return _pass(
        name, kind,
        "one architecture signature, one forward and one backward per step, and the frozen per-run cap held across every run",
    )


# --- ABSENCE ----------------------------------------------------------------


def check_no_unmanifested_telemetry(spec, record, tele_dir):
    kind = "ABSENCE"
    name = "no_unmanifested_telemetry"
    manifest = set(record.get("manifest", []))
    if manifest != set(EXPECTED_TELEMETRY):
        return _fail(name, kind, f"the sealed manifest is {sorted(manifest)}, not {sorted(EXPECTED_TELEMETRY)}")
    present = {e.name for e in os.scandir(tele_dir) if e.is_file()}
    extra = sorted(present - set(EXPECTED_TELEMETRY))
    if extra:
        return _fail(name, kind, f"files the runner did not write are present in the telemetry directory: {extra}")
    missing = sorted(set(EXPECTED_TELEMETRY) - present)
    if missing:
        return _fail(name, kind, f"expected files are absent: {missing}")
    return _pass(name, kind, f"telemetry directory holds exactly the expected set {sorted(EXPECTED_TELEMETRY)}")


# --- ORDERING ---------------------------------------------------------------


def check_runs_precede_seal(spec, record):
    kind = "ORDERING"
    name = "runs_precede_seal"
    runs = record.get("runs", [])
    if not runs:
        return _fail(name, kind, "no runs to order")
    order = [r["order_index"] for r in runs]
    if order != sorted(order) or order != list(range(len(runs))):
        return _fail(name, kind, f"append order is {order}, required {list(range(len(runs)))}")
    sealed = float(record["sealed_at"])
    prev_end = None
    for r in runs:
        if float(r["t_start"]) > float(r["t_end"]):
            return _fail(name, kind, f"run {r['point_index']} ended before it started")
        if prev_end is not None and float(r["t_start"]) < prev_end - 1e-6:
            return _fail(name, kind, f"run {r['point_index']} started before run {r['point_index'] - 1} ended")
        if float(r["t_end"]) > sealed + 1e-6:
            return _fail(name, kind, f"run {r['point_index']} ended after the record was sealed")
        prev_end = float(r["t_end"])
    return _pass(name, kind, "every run started after its predecessor ended and every run ended before the seal")


# --- EFFECT -----------------------------------------------------------------


def check_training_moved_weights(spec, record):
    kind = "EFFECT"
    name = "training_moved_live_weights"
    runs = record.get("runs", [])
    if not runs:
        return _fail(name, kind, "no runs to observe an effect in")
    for r in runs:
        if int(r["steps_taken"]) < 1:
            return _fail(name, kind, f"run {r['point_index']} took no optimizer step")
        d_loss = abs(float(r["post"]["val_loss"]) - float(r["pre"]["val_loss"]))
        d_dens = abs(float(r["post"]["density"]) - float(r["pre"]["density"]))
        if d_loss < EPS_EFFECT and d_dens < EPS_EFFECT:
            return _fail(
                name, kind,
                f"run {r['point_index']} left the live weights unmoved: val_loss delta {d_loss:.3e}, density delta {d_dens:.3e}",
            )
    deltas = [abs(float(r["post"]["val_loss"]) - float(r["pre"]["val_loss"])) for r in runs]
    return _pass(name, kind, f"every run moved the live weights, minimum val_loss delta {min(deltas):.6f}")


# --- DIVERGENCE -------------------------------------------------------------


def check_hypervolume_divergence(spec, record):
    kind = "DIVERGENCE"
    name = "hypervolume_independently_agrees"
    runs = record.get("runs", [])
    if not runs:
        return _fail(name, kind, "no achieved points to derive a hypervolume from"), 0.0
    anc = record["anchors"]
    raw_points = [(r["objectives"]["val_loss"], r["objectives"]["density"]) for r in runs]
    mine = reward_from_raw_points(raw_points, anc)
    theirs_norm = [tuple(p) for p in record["normalized_points"]]
    for i, (a, b) in enumerate(zip(mine["normalized_points"], theirs_norm)):
        if abs(a[0] - b[0]) > EPS_AGREE or abs(a[1] - b[1]) > EPS_AGREE:
            return _fail(name, kind, f"normalized point {i} disagrees: {a} against {b}"), 0.0
    if abs(mine["hypervolume_raw"] - float(record["hypervolume_raw"])) > EPS_AGREE:
        return _fail(
            name, kind,
            f"hypervolume disagrees: grid {mine['hypervolume_raw']!r} against sweep {record['hypervolume_raw']!r}",
        ), 0.0
    front_hv = hypervolume_grid([tuple(p) for p in record["pareto_front"]])
    if abs(front_hv - mine["hypervolume_raw"]) > EPS_AGREE:
        return _fail(
            name, kind,
            f"the recorded Pareto front does not carry the full hypervolume: {front_hv!r} against {mine['hypervolume_raw']!r}",
        ), 0.0
    return _pass(
        name, kind,
        f"grid union, sweep and Pareto-front hypervolume all agree at {mine['hypervolume_raw']:.12f}",
    ), mine["score"]


def check_record_origin(spec, record, nonce, source):
    """VALUE. Live-state read: the run_nonce and graded_submission_sha256 the record carries,
    against the nonce this verifier process minted after it started and hashlib.sha256 over the
    deliverable on the verifier filesystem at grade time.

    Nothing written before this process existed can carry that nonce, so a hand-authored record,
    a replayed genuine record with mutated objectives, and a record an atexit hook overwrote
    after the seal all fail here regardless of how well formed they are.
    """
    kind = "VALUE"
    name = "graded_record_authored_by_the_verifier"
    if source != "verifier_execution":
        return _fail(name, kind, f"authoritative record source is {source}")
    got = str(record.get("run_nonce", ""))
    if not nonce or got != nonce:
        return _fail(name, kind, "the record carries no nonce this verifier issued")
    if not SUBMISSION.is_file():
        return _fail(name, kind, f"no deliverable at {SUBMISSION}")
    live = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    if record.get("graded_submission_sha256") != live:
        return _fail(name, kind, "the measured deliverable is not the deliverable on disk at grade time")
    return _pass(name, kind, f"record stamped by this verifier run over deliverable {live[:16]}")


def check_submission_isolation(spec, record):
    """INVARIANT. Live-state read: the submission_isolation block the runner wrote, comparing
    runner_pid against worker_pid.

    The submission's optimizer must be a live object, so the training loop necessarily sits
    beside it. What must not sit beside it is any measurement. This checker holds the property
    that makes the objectives evidence: the process that measured them is not the process that
    ran submitted bytes.
    """
    kind = "INVARIANT"
    name = "submission_ran_outside_the_measuring_process"
    iso = record.get("submission_isolation") or {}
    runner, worker = iso.get("runner_pid"), iso.get("worker_pid")
    if iso.get("mode") != "subprocess" or iso.get("worker") != "frontier_worker.py":
        return _fail(name, kind, f"isolation mode is {iso.get('mode')!r} with worker {iso.get('worker')!r}")
    if not isinstance(worker, int) or worker <= 0 or worker == runner:
        return _fail(name, kind, f"worker pid {worker} against runner pid {runner}")
    for r in record.get("runs", []):
        if r.get("objectives_measured_by") != "runner" or r.get("elapsed_measured_by") != "runner":
            return _fail(name, kind, f"run {r.get('point_index')} does not attribute its objectives to the runner")
    return _pass(name, kind, f"submission hosted in pid {worker}, measured by runner pid {runner}")


def check_frozen_harness_bytes(spec, record):
    """VALUE. Live-state read: sha256 over the three executable harness files on the verifier
    filesystem at grade time, against the pin this bundle froze and against what the run
    recorded at load.

    The runner, the worker and the engine are agent-visible files. Nothing hashed them before,
    so a submission could have rewritten the density measurement itself rather than earning it.
    """
    kind = "VALUE"
    name = "frozen_harness_bytes_unmodified"
    if not FROZEN_BYTES_PIN.is_file():
        return _fail(name, kind, "frozen byte pin absent from the verifier tree")
    pinned = json.loads(FROZEN_BYTES_PIN.read_text())["files"]
    harness_dir = RUNNER.parent
    live, bad = {}, []
    for rel, want in sorted(pinned.items()):
        path = harness_dir / rel
        if not path.is_file():
            bad.append(f"{rel}:absent")
            continue
        live[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        if live[rel] != want:
            bad.append(f"{rel}:{live[rel][:12]}_is_not_{want[:12]}")
    recorded = record.get("harness_digests") or {}
    for rel, digest in sorted(live.items()):
        if recorded.get(rel) not in (None, digest):
            bad.append(f"{rel}:changed_after_load")
    if bad:
        return _fail(name, kind, "frozen harness bytes moved: " + ", ".join(bad))
    return _pass(name, kind, f"all {len(live)} frozen harness files match the pin at load and at grading")


def run_frontier_under_the_verifier(nonce, logs_dir):
    """Start one frontier run under this process and return the directory it wrote into.

    The directory is named from the same entropy as the nonce, so nothing that existed before
    this process ran can have pre-planted a record there.
    """
    rundir = logs_dir / f"verifier_run_{nonce}"
    if EXTERNAL_AUTHORITATIVE:
        return pathlib.Path(EXTERNAL_AUTHORITATIVE), "external"
    if not SUBMISSION.is_file():
        return rundir, "submission_absent"
    if not RUNNER.is_file():
        return rundir, "runner_absent"
    rundir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["BIA_S09_RUN_NONCE"] = nonce
    env["BIA_S09_TELEMETRY_DIR"] = str(rundir)
    env["BIA_S09_SUBMISSION"] = str(SUBMISSION)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("BIA_S09_AUTHORITATIVE_TELEMETRY_DIR", None)
    try:
        proc = subprocess.run([sys.executable, str(RUNNER)], capture_output=True, text=True,
                              env=env, timeout=VERIFIER_RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return rundir, "timeout"
    (logs_dir / "verifier-run-stdout.md").write_text(
        (proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or "")
    )
    if not (rundir / "frontier_record.json").is_file():
        return rundir, "run_produced_no_record"
    return rundir, "verifier_execution"


# --- driver -----------------------------------------------------------------


def write_reward(score) -> None:
    """Write the one float the runtime reads from /logs/verifier/reward.txt.

    Never raises: a verifier that dies while reporting its own score turns a graded zero into an
    unscored run, which is the failure this write exists to prevent.
    """
    reward_path = pathlib.Path(os.environ.get("BIA_S09_REWARD", "/logs/verifier/reward.txt"))
    try:
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text("%.12f\n" % min(max(float(score), 0.0), 1.0))
    except (OSError, TypeError, ValueError) as exc:
        print(f"reward path {reward_path} unwritable: {exc}", file=sys.stderr)


def run_checker(ident, kind, fn, *args):
    """Run one checker, turning a raise into that checker's own attributed failure.

    A checker that throws is a failing checker, never a pass and never an unscored run: an
    escaping exception would leave the reward absent, which reads as an infrastructure fault
    rather than as the zero the bundle actually earned.
    """
    try:
        return fn(*args)
    except Exception as exc:
        return _fail(ident, kind, f"checker raised: {exc!r}")


def main() -> int:
    spec_path = os.environ.get("BIA_S09_SPEC")
    tele_dir = pathlib.Path(os.environ.get("BIA_S09_TELEMETRY_DIR", "/telemetry"))
    score_path = pathlib.Path(os.environ.get("BIA_S09_SCORE", "/logs/verifier/score.json"))
    outcomes_path = pathlib.Path(os.environ.get("BIA_S09_OUTCOMES", "/logs/verifier/outcomes.json"))
    score_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(score, reason, results, extra=None):
        zero_reasons = sorted({r["reason"] for r in results if not r["passed"] and r["reason"]})
        payload = {"score": float(score), "reason": reason, "zero_reasons": zero_reasons}
        if extra:
            payload.update(extra)
        score_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        outcomes = {r["name"]: bool(r["passed"]) for r in results}
        outcomes["_results"] = results
        outcomes["_score"] = float(score)
        outcomes["_reason"] = reason
        outcomes["_zero_reasons"] = zero_reasons
        outcomes["_graded_record"] = str(GRADED_RECORD_PATH[0])
        outcomes_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True))
        write_reward(score)
        print(json.dumps(payload, indent=2, sort_keys=True))
        for r in results:
            verdict = "PASS" if r["passed"] else f"FAIL {r['reason']}"
            print(f"  [{verdict}] {r['kind']:<11} {r['name']}: {r['detail']}")
        return 0

    if not spec_path or not pathlib.Path(spec_path).is_file():
        emit(0.0, "frozen_spec_absent", [])
        return 2
    spec = json.loads(pathlib.Path(spec_path).read_text())

    # The graded record is the one this process produced. The verifier runs the frozen runner
    # itself, in a directory it names from system entropy, and stamps it with a nonce nothing
    # outside this process has seen. Whatever the agent left in its own telemetry directory is
    # not read here, because a record the submission can write is not evidence.
    nonce = secrets.token_hex(16)
    graded_dir, source = run_frontier_under_the_verifier(nonce, score_path.parent)
    if source not in ("verifier_execution", "external"):
        emit(0.0, f"verifier_execution_failed:{source}", [])
        return 2
    tele_dir = graded_dir

    record_path = tele_dir / "frontier_record.json"
    if not record_path.is_file():
        emit(0.0, "frontier_record_absent", [])
        return 2
    GRADED_RECORD_PATH[0] = str(record_path)
    record = json.loads(record_path.read_text())
    if record.get("schema") != "bia.s09.frontier_record/v1":
        emit(0.0, "frontier_record_schema_unrecognized", [])
        return 2
    if "fault" in record:
        return emit(0.0, "frontier_run_faulted", [], {"fault_head": record["fault"].splitlines()[-1][:200]})

    results = [
        run_checker("frontier_cardinality", "VALUE", check_frontier_cardinality, spec, record),
        run_checker("frozen_substrate_invariant", "INVARIANT", check_frozen_substrate, spec, record),
        run_checker("no_unmanifested_telemetry", "ABSENCE", check_no_unmanifested_telemetry, spec, record, tele_dir),
        run_checker("runs_precede_seal", "ORDERING", check_runs_precede_seal, spec, record),
        run_checker("training_moved_live_weights", "EFFECT", check_training_moved_weights, spec, record),
        run_checker("graded_record_authored_by_the_verifier", "VALUE", check_record_origin, spec, record, nonce, source),
        run_checker("submission_ran_outside_the_measuring_process", "INVARIANT", check_submission_isolation, spec, record),
        run_checker("frozen_harness_bytes_unmodified", "VALUE", check_frozen_harness_bytes, spec, record),
    ]
    try:
        div, recomputed = check_hypervolume_divergence(spec, record)
    except Exception as exc:
        div = _fail("hypervolume_independently_agrees", "DIVERGENCE", f"checker raised: {exc!r}")
        recomputed = 0.0
    results.append(div)

    failed = [r["name"] for r in results if not r["passed"]]
    if failed:
        return emit(0.0, "checker_failed:" + ",".join(failed), results)

    score = min(max(float(recomputed), 0.0), 1.0)
    single_best = max(
        (u1 * u2 for u1, u2 in (tuple(p) for p in record["normalized_points"])),
        default=0.0,
    )
    return emit(
        score,
        "ok",
        results,
        {
            "hypervolume": score,
            "best_single_point_hypervolume": single_best,
            "pareto_front_size": len(record["pareto_front"]),
            "reference_point_normalized": [0.0, 0.0],
            "ideal_point_normalized": [1.0, 1.0],
            "reference_point_raw": [record["anchors"]["loss_ref"], record["anchors"]["density_ref"]],
            "ideal_point_raw": [record["anchors"]["loss_ideal"], record["anchors"]["density_ideal"]],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
