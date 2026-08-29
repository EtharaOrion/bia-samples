#!/usr/bin/env python3
"""Grader for the BIA-GSN-1 kernel throughput task.

Private verifier entry point. It runs the correctness gate first, then the timing
protocol, then the six checkers, and only then computes a reward. The order matters:
a submission that diverges from the reference never reaches a timing number, so a
fast wrong implementation cannot bank a partial reward.

ISOLATION MODEL. The submitted module does not execute in this interpreter. Two worker
processes are started from tests/checkers/worker.py, one hosting the frozen reference
and one hosting the submission, and this process speaks to both over a line protocol.
It follows that:

  the checker table cannot be rebound, because it lives in a process the submission
  never enters;

  the clock cannot be moved, because every trial is timed on this side of the pipe;

  the timing inputs cannot be reused, because this process chooses the chain that folds
  each trial's output back into the next trial's input;

  device time cannot be deferred out of the measurement, because every round ends with a
  timed barrier that this process verifies against the reference worker's answer at the
  same positions, so a worker that answers its trials without waiting for the accelerator
  pays the outstanding time inside the interval being timed;

  the reference output tree cannot be scavenged, because the submission's fixtures are
  computed and written BEFORE the reference's are, so at the moment the submission runs
  there is nothing to read;

  and no number is scraped out of a worker's own report, because the only thing a worker
  says is that it finished.

Every zero score written here carries a machine readable reason.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from checkers import kinds, timing  # noqa: E402

WORKER = HERE / "checkers" / "worker.py"


def _env_path(name, default):
    return pathlib.Path(os.environ.get(name, default))


def _write_json(path, payload):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _kebab(reason):
    return str(reason).replace("_", "-")


def _fail(score_path, report_path, reason, **detail):
    _write_json(score_path, {"score": 0.0})
    _write_json(
        report_path,
        {"score": 0.0, "reason": reason, "zero_reason": _kebab(reason), "detail": detail},
    )
    print("SCORE 0.0 reason=%s" % reason)
    print(json.dumps(detail, indent=2, sort_keys=True)[:4000])
    return 1


def worker_env(work_dir):
    """A minimal environment this process builds from scratch for both workers.

    Nothing the solving container exported reaches a graded process. The relaxed
    precision override variables are absent by construction rather than by policy, and
    the ABSENCE checker reads each worker's /proc environ to prove it.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(work_dir),
        "TMPDIR": str(work_dir),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for passthrough in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LD_LIBRARY_PATH"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]
    return env


def start_workers(device, chunk_rows, submission_path, env_dir, work_dir, logs_dir):
    env = worker_env(work_dir)
    common = [sys.executable, "-E", "-s", "-B", str(WORKER),
              "--device", str(device), "--chunk-rows", str(int(chunk_rows))]
    reference = timing.Worker(
        "reference", common + ["--impl", "reference"], env,
        str(pathlib.Path(logs_dir) / "worker-reference.err"))
    submission = timing.Worker(
        "submission", common + ["--impl", "submission",
                                "--submission-path", str(submission_path),
                                "--env-dir", str(env_dir)], env,
        str(pathlib.Path(logs_dir) / "worker-submission.err"))
    return reference, submission


def compare_trees(sub_dir, ref_dir, names):
    """Byte comparison performed here, over files two separate processes wrote."""
    rows = []
    ok = True
    for name in names:
        sub_path = pathlib.Path(sub_dir) / name
        ref_path = pathlib.Path(ref_dir) / name
        if not ref_path.is_file():
            rows.append({"artifact": name, "status": "reference_missing"})
            ok = False
            continue
        if not sub_path.is_file():
            rows.append({"artifact": name, "status": "submission_missing"})
            ok = False
            continue
        sub_bytes = sub_path.read_bytes()
        ref_bytes = ref_path.read_bytes()
        if len(sub_bytes) != len(ref_bytes):
            rows.append({"artifact": name, "status": "length_mismatch",
                         "submission_bytes": len(sub_bytes), "reference_bytes": len(ref_bytes)})
            ok = False
            continue
        if sub_bytes == ref_bytes:
            rows.append({"artifact": name, "status": "identical"})
            continue
        first = -1
        for i in range(0, len(ref_bytes), 4):
            if sub_bytes[i:i + 4] != ref_bytes[i:i + 4]:
                first = i // 4
                break
        rows.append({"artifact": name, "status": "bit_pattern_mismatch", "first_word": first})
        ok = False
    return ok, rows


def main():
    bundle_root = _env_path("BIA_BUNDLE", "/")
    tests_root = _env_path("BIA_TESTS", str(HERE))
    env_dir = _env_path("BIA_ENV_DIR", str(bundle_root / "environment"))
    submission_path = _env_path("BIA_SUBMISSION", "/workspace/submission/impl.py")
    logs_dir = _env_path("BIA_LOGS", "/logs/verifier")
    artifacts_dir = _env_path("BIA_ARTIFACTS", "/workspace/artifacts")
    score_path = _env_path("SCORE_PATH", str(logs_dir / "score.json"))
    report_path = _env_path("BIA_REPORT", str(logs_dir / "report.json"))
    outcomes_path = _env_path("BIA_OUTCOMES", str(logs_dir / "outcomes.json"))
    events_path = _env_path("BIA_EVENTS", str(logs_dir / "harness_events.jsonl"))
    summary_path = _env_path("BIA_TIMING_SUMMARY", str(logs_dir / "timing_summary.json"))
    smoke = os.environ.get("BIA_SMOKE", "0") == "1"

    logs_dir.mkdir(parents=True, exist_ok=True)
    sub_out = artifacts_dir / "submission_out"
    ref_out = artifacts_dir / "reference_out"
    timing_out = artifacts_dir / "timing_out"
    for target in (sub_out, ref_out, timing_out):
        target.mkdir(parents=True, exist_ok=True)

    spec_path = tests_root / "checkers" / "spec.json"
    manifest_path = tests_root / "checkers" / "frozen_manifest.json"
    if not spec_path.is_file() or not manifest_path.is_file():
        return _fail(score_path, report_path, "verifier_tree_incomplete", spec=str(spec_path))
    if not WORKER.is_file():
        return _fail(score_path, report_path, "verifier_worker_absent", worker=str(WORKER))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    profile_name = "smoke" if smoke else "graded"
    profile = spec["profiles"][profile_name]

    try:
        import torch
    except Exception as exc:
        return _fail(score_path, report_path, "torch_unavailable", error=repr(exc))

    for setter in (
        lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", False),
        lambda: setattr(torch.backends.cudnn, "allow_tf32", False),
        lambda: setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False),
        lambda: setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False),
        lambda: torch.set_float32_matmul_precision("highest"),
    ):
        try:
            setter()
        except Exception:
            pass

    device = profile["device"]
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return _fail(score_path, report_path, "config_fault_cuda_unavailable", device=device)

    frozen_start = {}
    for rel in manifest:
        path = bundle_root / rel
        frozen_start[rel] = kinds._sha256_file(path) if path.is_file() else None

    log = timing.EventLog(events_path)
    log.emit("run_start", profile=profile_name, device=str(device), smoke=smoke)

    chunk_rows = int(profile["chunk_rows"])
    work_dir = artifacts_dir / "worker_home"
    work_dir.mkdir(parents=True, exist_ok=True)

    marker = artifacts_dir / ".invocation_start"
    marker.write_text(str(time.time_ns()), encoding="utf-8")
    invocation_start_ns = marker.stat().st_mtime_ns

    try:
        reference, submission = start_workers(
            device, chunk_rows, submission_path, env_dir, work_dir, logs_dir)
    except Exception as exc:
        log.emit("submission_load_failed", error=repr(exc))
        return _fail(score_path, report_path, "submission_unloadable", error=repr(exc))

    artifact_names = []
    for fixture in profile["fixtures"]:
        artifact_names.extend(["%s.y.bin" % fixture["name"], "%s.r.bin" % fixture["name"]])

    try:
        # The submission computes every fixture FIRST. The reference output tree does not
        # exist while the submitted implementation is running, so there is nothing on
        # disk for it to return instead of computing.
        for who, out_dir in (("submission", sub_out), ("reference", ref_out)):
            worker = submission if who == "submission" else reference
            for fixture in profile["fixtures"]:
                worker.call(cmd="fixture", kind=fixture["kind"], rows=int(fixture["rows"]),
                            cols=int(fixture["cols"]), seed=int(fixture["seed"]),
                            stem=str(out_dir / fixture["name"]))
                log.emit("fixture_computed", impl=who, fixture=fixture["name"])
    except Exception as exc:
        log.emit("correctness_gate_failed", error=repr(exc))
        reference.close()
        submission.close()
        return _fail(score_path, report_path, "submission_raised_during_correctness_gate",
                     error=repr(exc))

    fixtures_agree, fixture_rows = compare_trees(sub_out, ref_out, artifact_names)
    log.emit("correctness_gate_complete", fixtures=len(profile["fixtures"]), agree=fixtures_agree)

    if not fixtures_agree:
        # The declared order is correctness first, timing second. A submission that
        # diverges from the reference never reaches a timing number, so the run stops
        # here and attributes the zero to the checker that decided it rather than to
        # whatever the timing phase would have raised on a malformed output.
        reference.close()
        submission.close()
        results = [{"checker": "divergence_bitwise", "kind": "DIVERGENCE", "passed": False,
                    "reason": "divergence_from_reference_output",
                    "zero_reason": kinds.ZERO_REASONS["divergence_bitwise"],
                    "detail": {"artifacts": fixture_rows}}]
        for cid, _fn in kinds.CHECKERS:
            if cid == "divergence_bitwise":
                continue
            results.append({"checker": cid, "kind": "NOT_EVALUATED", "passed": False,
                            "reason": "not_evaluated_correctness_gate_closed_first",
                            "zero_reason": kinds.ZERO_REASONS.get(cid), "detail": {}})
        _write_json(outcomes_path, {r["checker"]: r["passed"] for r in results})
        _write_json(report_path, {
            "score": 0.0,
            "reason": "checker_failed:divergence_bitwise",
            "zero_reason": kinds.ZERO_REASONS["divergence_bitwise"],
            "profile": profile_name,
            "smoke": smoke,
            "fixtures": fixture_rows,
            "checkers": results,
        })
        _write_json(score_path, {"score": 0.0})
        print("fixtures      : %d, all agree = False" % len(profile["fixtures"]))
        for r in results:
            print("checker %-30s %-11s %s" % (r["checker"], r["kind"], "FAIL " + r["reason"]))
        print("SCORE 0.000000 reason=checker_failed:divergence_bitwise")
        return 0

    declared_order = list(spec["timing_protocol"]["order_within_round"])
    workload_seed = 7777777 if not smoke else 7777779
    rows, cols = int(profile["rows"]), int(profile["cols"])

    # The chain columns are drawn here, from a generator this process seeds, so the
    # positions a worker must fold are not derivable from anything it holds.
    rng = random.Random(workload_seed ^ 0x5F3759DF)

    def chain_columns(round_index):
        dst = [rng.randrange(cols) for _ in range(rows)]
        src = [rng.randrange(cols) for _ in range(rows)]
        return dst, src

    probe_width = min(512, rows)

    def probe_positions(round_index):
        y_idx = [rng.randrange(rows) * cols + rng.randrange(cols) for _ in range(probe_width)]
        r_idx = [rng.randrange(rows) for _ in range(probe_width)]
        return y_idx, r_idx

    def dump_stem(name, round_index):
        # One stem per implementation, overwritten every round. The bytes are compared
        # the moment both workers have written them, so the tree never holds more than
        # one round's output and a long measurement cannot fill the disk.
        return str(timing_out / ("%s.current" % name))

    round_agreement = []

    def on_round_dumped(round_index, probe_values):
        names = ["submission.current.y.bin", "submission.current.r.bin"]
        refs = ["reference.current.y.bin", "reference.current.r.bin"]
        status = "identical"
        for sub_name, ref_name in zip(names, refs):
            sub_path = timing_out / sub_name
            ref_path = timing_out / ref_name
            if not sub_path.is_file() or not ref_path.is_file():
                status = "missing"
                break
            if sub_path.read_bytes() != ref_path.read_bytes():
                status = "bit_pattern_mismatch"
                break
        probe_status = "identical"
        ref_probe = probe_values.get("reference") or {}
        sub_probe = probe_values.get("submission") or {}
        for part in ("y", "r"):
            if ref_probe.get(part) != sub_probe.get(part):
                probe_status = "barrier_value_mismatch"
                break
        round_agreement.append({"round": round_index, "outputs": status, "barrier": probe_status})
        log.emit("round_outputs_compared", round=round_index, outputs=status, barrier=probe_status)

    workers = {"reference": reference, "submission": submission}
    try:
        for name, worker in workers.items():
            worker.call(cmd="prepare", rows=rows, cols=cols, seed=workload_seed)
        durations, aborted = timing.run_measurement(
            log, workers, profile, device, float(profile["measurement_budget_s"]),
            chain_columns, probe_positions, dump_stem, on_round_dumped)
    except Exception as exc:
        log.emit("measurement_failed", error=repr(exc))
        reference.close()
        submission.close()
        return _fail(score_path, report_path, "submission_raised_during_measurement",
                     error=repr(exc))

    worker_environ = {name: w.environ for name, w in workers.items()}
    worker_flags = {}
    for name, worker in workers.items():
        try:
            worker_flags[name] = worker.call(cmd="flags").get("flags")
        except Exception as exc:
            worker_flags[name] = {"error": repr(exc)}
    reference.close()
    submission.close()

    rounds_used = min(len(durations["reference"]), len(durations["submission"]))
    timing_rows = []
    timing_agree = bool(round_agreement)
    for entry in round_agreement:
        timing_rows.append({"artifact": "timing round %02d" % entry["round"],
                            "status": entry["outputs"], "barrier": entry["barrier"]})
        if entry["outputs"] != "identical" or entry["barrier"] != "identical":
            timing_agree = False
    for stale in timing_out.glob("*.bin"):
        try:
            stale.unlink()
        except OSError:
            pass

    summary = timing.summarize(durations, declared_order)
    baseline = float(spec["reward"]["baseline_metric"])
    target = float(spec["reward"]["target_metric"])
    raw = (summary["speedup"] - baseline) / (target - baseline)
    reward = min(max(raw, 0.0), 1.0)
    summary["raw"] = raw
    summary["reward"] = reward
    summary["aborted"] = aborted
    summary["profile"] = profile_name
    _write_json(summary_path, summary)

    ctx = {
        "bundle_root": str(bundle_root),
        "submission_out": str(sub_out),
        "reference_out": str(ref_out),
        "timing_out": str(timing_out),
        "fixtures": profile["fixtures"],
        "invocation_start_ns": invocation_start_ns,
        "frozen_manifest": manifest,
        "frozen_start": frozen_start,
        "events_path": str(events_path),
        "timing_summary_path": str(summary_path),
        "declared_order": declared_order,
        "expected_timed_trials": int(profile["timed_trials"]),
        "baseline_metric": baseline,
        "target_metric": target,
        "fixture_agreement": {"ok": fixtures_agree, "rows": fixture_rows},
        "timing_agreement": {"ok": timing_agree, "rows": timing_rows},
        "worker_environ": worker_environ,
        "worker_flags": worker_flags,
        "rounds_used": rounds_used,
    }
    results = kinds.run_all(ctx)
    outcomes = {r["checker"]: r["passed"] for r in results}
    _write_json(outcomes_path, outcomes)

    failed = [r for r in results if not r["passed"]]
    if aborted:
        score = 0.0
        reason = aborted
        zero_reason = _kebab(aborted)
    elif failed:
        score = 0.0
        reason = "checker_failed:" + ",".join(r["checker"] for r in failed)
        zero_reason = ",".join(r["zero_reason"] or "checker-raised" for r in failed)
    else:
        score = reward
        reason = "scored"
        zero_reason = None

    report = {
        "score": score,
        "reason": reason,
        "zero_reason": zero_reason,
        "profile": profile_name,
        "smoke": smoke,
        "speedup": summary["speedup"],
        "raw": raw,
        "reward_before_gate": reward,
        "fixtures": fixture_rows,
        "timing_outputs": timing_rows,
        "checkers": results,
        "timing": {"rounds_used": summary["rounds_used"], "per_round": summary["per_round"]},
    }
    _write_json(report_path, report)
    _write_json(score_path, {"score": float(score)})

    print("operator      : %s" % spec["operator_id"])
    print("profile       : %s" % profile_name)
    print("device        : %s" % device)
    print("fixtures      : %d, all agree = %s" % (len(profile["fixtures"]), fixtures_agree))
    print("timing outputs: %d compared, all agree = %s" % (len(timing_rows), timing_agree))
    print("speedup       : %.6f" % summary["speedup"])
    print("raw           : %.6f" % raw)
    for r in results:
        print("checker %-30s %-11s %s" % (r["checker"], r["kind"], "PASS" if r["passed"] else "FAIL " + r["reason"]))
    print("SCORE %.6f reason=%s" % (score, reason))
    if smoke:
        print("SMOKE MODE: the timing number above is not a claim about the operating point.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
