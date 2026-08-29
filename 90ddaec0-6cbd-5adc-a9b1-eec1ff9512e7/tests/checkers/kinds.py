"""The six graded checkers for the BIA-GSN-1 throughput task.

Private verifier code. Every checker here reduces to exactly one of the six kinds and
every checker names the live state it reads. The mapping from checker identifier to
kind and to live state read is declared once in tests/checkers.yaml and this module is
the executable side of that declaration.

None of these functions reads a clock, a random source, a network, or a value the
submission reported about itself. Each reads either bytes on the filesystem that the
run produced, or process state the grader read out of the kernel about the worker
processes it started.

The submitted implementation runs in its own process, so nothing here can be rebound by
it: the previous design executed the submission in this interpreter, which let a
submission replace this module's CHECKERS table with six closures that returned true.
That is not a defect these functions could have detected, and it is not one they are
asked to detect now. It is closed by where the submission runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

RESULT_KEYS = ("checker", "kind", "passed", "reason", "zero_reason", "detail")

# The machine readable code each checker attributes its zero to, keyed by checker
# identifier. `reason` stays the specific diagnosis this run produced; this code is
# the stable lowercase kebab a downstream grader branches on, and it is the code
# tests/checkers.yaml declares as that checker's zero_reason. The two must stay in
# step: a zero_reason the carrier never emits is a zero that still arrives bare.
ZERO_REASONS = {
    "divergence_bitwise": "divergence-from-reference-output",
    "effect_outputs_materialized": "submission-did-not-materialize-declared-outputs",
    "absence_relaxed_precision": "relaxed-precision-switch-present",
    "invariant_frozen_surface": "frozen-surface-moved-during-run",
    "ordering_measurement_protocol": "measurement-protocol-out-of-order",
    "value_reward_recomputed": "recorded-reward-not-reproducible-from-durations",
}


def _result(checker, kind, passed, reason, **detail):
    return {
        "checker": checker,
        "kind": kind,
        "passed": bool(passed),
        "reason": reason,
        "zero_reason": None if passed else ZERO_REASONS.get(checker),
        "detail": detail,
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_events(path):
    events = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def check_divergence_bitwise(ctx):
    """DIVERGENCE. The submitted implementation and the private reference agree.

    Live state read: the artifact trees on the filesystem, each written by a different
    process. ctx['submission_out'] holds what the submission worker returned for every
    correctness fixture, ctx['reference_out'] holds what the reference worker returned
    for the same fixtures, and ctx['timing_out'] holds the raw output both workers were
    made to dump at the end of every timing round. The checker reads raw bytes and
    compares them, so it never consults any pass or fail claim recorded by the code that
    produced them.

    The timing round comparison is what makes a cached answer visible. Each timing trial
    folds the previous output back into the input, so a memo that serves a stored tuple
    produces an output that is correct for an input the round no longer holds, and the
    end of round bytes diverge. Each round also carries a barrier answer, the exact
    values at grader chosen positions, and the two workers' barrier answers must agree
    for the same reason and by the same comparison.
    """
    sub_dir = pathlib.Path(ctx["submission_out"])
    ref_dir = pathlib.Path(ctx["reference_out"])
    per_fixture = []
    ok = True
    for row in ctx.get("timing_agreement", {}).get("rows", []):
        per_fixture.append(dict(row, scope="timing_round"))
        if row.get("status") != "identical" or row.get("barrier") != "identical":
            ok = False
    if not ctx.get("timing_agreement", {}).get("ok", False):
        ok = False
    for fixture in ctx["fixtures"]:
        for part in ("y", "r"):
            name = "%s.%s.bin" % (fixture["name"], part)
            sub_path = sub_dir / name
            ref_path = ref_dir / name
            if not ref_path.is_file():
                per_fixture.append({"artifact": name, "status": "reference_missing"})
                ok = False
                continue
            if not sub_path.is_file():
                per_fixture.append({"artifact": name, "status": "submission_missing"})
                ok = False
                continue
            sub_bytes = sub_path.read_bytes()
            ref_bytes = ref_path.read_bytes()
            if len(sub_bytes) != len(ref_bytes):
                per_fixture.append(
                    {
                        "artifact": name,
                        "status": "length_mismatch",
                        "submission_bytes": len(sub_bytes),
                        "reference_bytes": len(ref_bytes),
                    }
                )
                ok = False
                continue
            if sub_bytes == ref_bytes:
                per_fixture.append({"artifact": name, "status": "identical"})
                continue
            first = -1
            for i in range(0, len(ref_bytes), 4):
                if sub_bytes[i : i + 4] != ref_bytes[i : i + 4]:
                    first = i // 4
                    break
            per_fixture.append(
                {"artifact": name, "status": "bit_pattern_mismatch", "first_word": first}
            )
            ok = False
    reason = "ok" if ok else "divergence_from_reference_output"
    return _result("divergence_bitwise", "DIVERGENCE", ok, reason, artifacts=per_fixture)


def check_effect_outputs_materialized(ctx):
    """EFFECT. Invoking the submitted implementation changed the artifact tree.

    Live state read: os.stat over every declared artifact under ctx['submission_out'],
    compared against the pre invocation instant ctx['invocation_start_ns'] the harness
    recorded before the first call into the submitted implementation. The named change
    is that the directory gained exactly the declared artifacts, each of exactly the
    declared byte length, each stamped after that instant.
    """
    sub_dir = pathlib.Path(ctx["submission_out"])
    start_ns = int(ctx["invocation_start_ns"])
    rows = []
    ok = True
    for fixture in ctx["fixtures"]:
        expected = {
            "y": int(fixture["rows"]) * int(fixture["cols"]) * 4,
            "r": int(fixture["rows"]) * 4,
        }
        for part, want in expected.items():
            name = "%s.%s.bin" % (fixture["name"], part)
            path = sub_dir / name
            if not path.is_file():
                rows.append({"artifact": name, "status": "absent"})
                ok = False
                continue
            stat = path.stat()
            fresh = int(stat.st_mtime_ns) >= start_ns
            sized = int(stat.st_size) == want
            if not (fresh and sized):
                ok = False
            rows.append(
                {
                    "artifact": name,
                    "status": "present",
                    "bytes": int(stat.st_size),
                    "expected_bytes": want,
                    "mtime_after_invocation": fresh,
                }
            )
    reason = "ok" if ok else "submission_did_not_materialize_declared_outputs"
    return _result("effect_outputs_materialized", "EFFECT", ok, reason, artifacts=rows)


def check_absence_relaxed_precision(ctx):
    """ABSENCE. No relaxed precision switch exists in the graded processes.

    Live state read, in three parts.

    First, /proc/<pid>/environ of each worker process, read by the grader out of the
    kernel while that worker was alive. This is the load bearing read: it is the one
    fact about a worker process that code inside it cannot rewrite for the grader's
    benefit, and the grader builds each worker's environment from scratch so an override
    variable is absent by construction rather than by request.

    Second, the torch backend precision flags in this interpreter, which must still be
    strict.

    Third, the flags and the live environment table each worker reports about itself.
    That third read catches a switch a process turned on for itself after starting, which
    /proc environ does not show. It is recorded rather than trusted, because a worker can
    say anything. It does not need to be trusted: an implementation that actually relaxed
    precision would produce different bit patterns, and divergence_bitwise reads those
    bytes.
    """
    findings = []
    for name, environ in (ctx.get("worker_environ") or {}).items():
        if environ is None:
            findings.append({"probe": "worker:%s:environ" % name, "state": "unreadable"})
            continue
        for var in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                    "PYTORCH_ENABLE_FAST_MATH", "TORCH_CUDNN_V8_API_ENABLED"):
            raw = environ.get(var)
            if raw not in (None, "", "0"):
                findings.append({"probe": "worker:%s:env:%s" % (name, var), "state": raw})
    for name, flags in (ctx.get("worker_flags") or {}).items():
        if not isinstance(flags, dict) or "error" in flags:
            findings.append({"probe": "worker:%s:flags" % name, "state": "unreported"})
            continue
        if flags.get("allow_tf32_matmul") or flags.get("allow_tf32_cudnn"):
            findings.append({"probe": "worker:%s:allow_tf32" % name, "state": "enabled"})
        if flags.get("autocast_enabled"):
            findings.append({"probe": "worker:%s:autocast" % name, "state": "enabled"})
        if str(flags.get("float32_matmul_precision")) != "highest":
            findings.append({"probe": "worker:%s:float32_matmul_precision" % name,
                             "state": str(flags.get("float32_matmul_precision"))})
        for var, raw in (flags.get("live_env") or {}).items():
            if raw not in (None, "", "0"):
                findings.append({"probe": "worker:%s:live_env:%s" % (name, var), "state": raw})
    try:
        import torch
    except Exception as exc:
        return _result(
            "absence_relaxed_precision",
            "ABSENCE",
            False,
            "torch_unavailable_in_verifier",
            error=repr(exc),
        )
    probes = []
    probes.append(("torch.backends.cuda.matmul.allow_tf32", bool(torch.backends.cuda.matmul.allow_tf32)))
    probes.append(("torch.backends.cudnn.allow_tf32", bool(torch.backends.cudnn.allow_tf32)))
    reduced = getattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", None)
    if reduced is not None:
        probes.append(("torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction", bool(reduced)))
    bf16_reduced = getattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", None)
    if bf16_reduced is not None:
        probes.append(("torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction", bool(bf16_reduced)))
    for name, value in probes:
        if value:
            findings.append({"probe": name, "state": "enabled"})
    precision = str(torch.get_float32_matmul_precision())
    if precision != "highest":
        findings.append({"probe": "torch.get_float32_matmul_precision", "state": precision})
    try:
        if bool(torch.is_autocast_enabled()):
            findings.append({"probe": "torch.is_autocast_enabled", "state": "enabled"})
    except Exception:
        pass
    for var in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "PYTORCH_ENABLE_FAST_MATH"):
        raw = os.environ.get(var)
        if raw not in (None, "", "0"):
            findings.append({"probe": "env:%s" % var, "state": raw})
    ok = not findings
    reason = "ok" if ok else "relaxed_precision_switch_present"
    return _result(
        "absence_relaxed_precision",
        "ABSENCE",
        ok,
        reason,
        findings=findings,
        float32_matmul_precision=precision,
        worker_flags=ctx.get("worker_flags"),
    )


def check_invariant_frozen_surface(ctx):
    """INVARIANT. The frozen operator surface held for the whole run.

    Live state read: the sha256 of every path named in the frozen manifest, recomputed
    from the filesystem at checker time, compared against both the manifest value and
    the snapshot ctx['frozen_start'] the harness took before the submission was loaded.
    A file that was correct at the start, mutated during the run, and restored would
    still have to hold the manifest digest at both ends, which is the property this
    checker asserts rather than the weaker one that the file is correct right now.
    """
    manifest = ctx["frozen_manifest"]
    start = ctx["frozen_start"]
    rows = []
    ok = True
    for rel, want in sorted(manifest.items()):
        path = pathlib.Path(ctx["bundle_root"]) / rel
        if not path.is_file():
            rows.append({"path": rel, "status": "absent"})
            ok = False
            continue
        live = _sha256_file(path)
        began = start.get(rel)
        agree = (live == want) and (began == want)
        if not agree:
            ok = False
        rows.append(
            {
                "path": rel,
                "status": "held" if agree else "moved",
                "manifest": want,
                "at_start": began,
                "at_end": live,
            }
        )
    reason = "ok" if ok else "frozen_surface_moved_during_run"
    return _result("invariant_frozen_surface", "INVARIANT", ok, reason, paths=rows)


def check_ordering_measurement_protocol(ctx):
    """ORDERING. The measurement events occurred in the required sequence.

    Live state read: the append only event log at ctx['events_path'], which the timing
    harness wrote one line at a time as the run progressed. The required sequence is
    the correctness gate closing before any warmup, a warmup completing before the
    first timed trial of every implementation and round, timed trial indices ascending
    contiguously from zero within each implementation and round, and the declared
    within round order of implementations repeating identically in every round.
    """
    events = _read_events(ctx["events_path"])
    seqs = [int(e["seq"]) for e in events]
    if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
        return _result(
            "ordering_measurement_protocol",
            "ORDERING",
            False,
            "event_sequence_not_monotonic",
            count=len(events),
        )
    gate_seq = None
    for e in events:
        if e["event"] == "correctness_gate_complete":
            gate_seq = int(e["seq"])
            break
    if gate_seq is None:
        return _result(
            "ordering_measurement_protocol", "ORDERING", False, "correctness_gate_event_absent"
        )
    warmups = {}
    trials = {}
    durations = {}
    totals = {}
    barriers = {}
    dumps = set()
    round_order = {}
    problems = []
    for e in events:
        if e["event"] == "warmup_complete":
            key = (e["impl"], int(e["round"]))
            warmups[key] = int(e["seq"])
            round_order.setdefault(int(e["round"]), []).append(e["impl"])
            if int(e["seq"]) < gate_seq:
                problems.append({"issue": "warmup_before_correctness_gate", "key": list(key)})
        elif e["event"] == "trial":
            key = (e["impl"], int(e["round"]))
            trials.setdefault(key, []).append((int(e["seq"]), int(e["index"])))
            durations.setdefault(key, []).append(int(e["duration_ns"]))
        elif e["event"] == "probe":
            barriers[(e["impl"], int(e["round"]))] = int(e["duration_ns"])
        elif e["event"] == "round_impl_complete":
            totals[(e["impl"], int(e["round"]))] = int(e["total_ns"])
        elif e["event"] == "round_output_dumped":
            dumps.add((e["impl"], int(e["round"])))
    if not trials:
        problems.append({"issue": "no_timed_trials_recorded"})
    for key, recorded_total in sorted(totals.items()):
        if key not in barriers:
            problems.append({"issue": "round_barrier_never_recorded", "key": list(key)})
            continue
        if recorded_total != sum(durations.get(key, [])) + barriers[key]:
            problems.append({"issue": "round_total_is_not_its_trials_plus_its_barrier",
                             "key": list(key)})
        if key not in dumps:
            problems.append({"issue": "round_output_never_dumped", "key": list(key)})
    for key in sorted(trials):
        if key not in totals:
            problems.append({"issue": "round_without_recorded_total", "key": list(key)})
    for key, entries in sorted(trials.items()):
        warm = warmups.get(key)
        if warm is None:
            problems.append({"issue": "trials_without_warmup", "key": list(key)})
            continue
        if min(seq for seq, _ in entries) < warm:
            problems.append({"issue": "trial_before_warmup", "key": list(key)})
        indices = [idx for _, idx in sorted(entries)]
        if indices != list(range(len(indices))):
            problems.append({"issue": "trial_indices_not_contiguous", "key": list(key)})
    declared = list(ctx["declared_order"])
    for rnd, seen in sorted(round_order.items()):
        if seen != declared:
            problems.append({"issue": "round_order_deviates", "round": rnd, "seen": seen})
    ok = not problems
    reason = "ok" if ok else "measurement_protocol_out_of_order"
    return _result(
        "ordering_measurement_protocol",
        "ORDERING",
        ok,
        reason,
        problems=problems,
        rounds=len(round_order),
    )


def check_value_reward_recomputed(ctx):
    """VALUE. The recorded reward equals the value recomputed from the raw durations.

    Live state read: the per trial duration entries in the append only event log at
    ctx['events_path'], and the summary the timing phase wrote to
    ctx['timing_summary_path']. The checker recomputes the trial count, the per round
    totals, the speedup, and the clamped reward straight from the logged durations and
    requires the recorded summary to equal what it recomputed. The statistic it
    recomputes is the one the reward uses: the median across rounds of the per round
    ratio of TOTAL time.
    """
    import statistics

    events = _read_events(ctx["events_path"])
    summary = json.loads(pathlib.Path(ctx["timing_summary_path"]).read_text(encoding="utf-8"))
    declared = list(ctx["declared_order"])
    expected_trials = int(ctx["expected_timed_trials"])
    buckets = {}
    barriers = {}
    for e in events:
        if e["event"] == "trial":
            buckets.setdefault((e["impl"], int(e["round"])), []).append(int(e["duration_ns"]))
        elif e["event"] == "probe":
            barriers[(e["impl"], int(e["round"]))] = int(e["duration_ns"])
    problems = []
    for key, durations in sorted(buckets.items()):
        if len(durations) != expected_trials:
            problems.append(
                {"issue": "trial_count_mismatch", "key": list(key), "found": len(durations), "expected": expected_trials}
            )
        if any(d <= 0 for d in durations):
            problems.append({"issue": "non_positive_duration", "key": list(key)})
    rounds = sorted({int(e["round"]) for e in events if e["event"] == "trial"})
    ratios = []
    for rnd in rounds:
        ref = buckets.get((declared[0], rnd))
        sub = buckets.get((declared[1], rnd))
        if not ref or not sub:
            problems.append({"issue": "round_missing_implementation", "round": rnd})
            continue
        if (declared[0], rnd) not in barriers or (declared[1], rnd) not in barriers:
            problems.append({"issue": "round_barrier_absent", "round": rnd})
            continue
        sub_total = float(sum(sub) + barriers[(declared[1], rnd)])
        if sub_total <= 0.0:
            problems.append({"issue": "non_positive_round_total", "round": rnd})
            continue
        ratios.append(float(sum(ref) + barriers[(declared[0], rnd)]) / sub_total)
    recomputed_speedup = float(statistics.median(ratios)) if ratios else 0.0
    baseline = float(ctx["baseline_metric"])
    target = float(ctx["target_metric"])
    raw = (recomputed_speedup - baseline) / (target - baseline)
    recomputed_reward = min(max(raw, 0.0), 1.0)
    recorded_speedup = float(summary.get("speedup", -1.0))
    recorded_reward = float(summary.get("reward", -1.0))
    if abs(recorded_speedup - recomputed_speedup) > 1e-9:
        problems.append(
            {"issue": "speedup_disagrees", "recorded": recorded_speedup, "recomputed": recomputed_speedup}
        )
    if abs(recorded_reward - recomputed_reward) > 1e-12:
        problems.append(
            {"issue": "reward_disagrees", "recorded": recorded_reward, "recomputed": recomputed_reward}
        )
    if not (0.0 <= recomputed_reward <= 1.0):
        problems.append({"issue": "reward_out_of_range", "value": recomputed_reward})
    ok = not problems
    reason = "ok" if ok else "recorded_reward_not_reproducible_from_durations"
    return _result(
        "value_reward_recomputed",
        "VALUE",
        ok,
        reason,
        problems=problems,
        recomputed_speedup=recomputed_speedup,
        recomputed_reward=recomputed_reward,
    )


CHECKERS = (
    ("divergence_bitwise", check_divergence_bitwise),
    ("effect_outputs_materialized", check_effect_outputs_materialized),
    ("absence_relaxed_precision", check_absence_relaxed_precision),
    ("invariant_frozen_surface", check_invariant_frozen_surface),
    ("ordering_measurement_protocol", check_ordering_measurement_protocol),
    ("value_reward_recomputed", check_value_reward_recomputed),
)

RED_LINES = ("divergence_bitwise", "absence_relaxed_precision", "invariant_frozen_surface")


def run_all(ctx):
    results = []
    for name, fn in CHECKERS:
        try:
            results.append(fn(ctx))
        except Exception as exc:
            results.append(
                _result(name, "UNRESOLVED", False, "checker_raised", error=repr(exc))
            )
    return results


def load(path=None):
    """Read the outcome map the grader wrote, for the compiled rubric tests."""
    target = path or os.environ.get("BIA_OUTCOMES", "/logs/verifier/outcomes.json")
    return json.loads(pathlib.Path(target).read_text(encoding="utf-8"))
