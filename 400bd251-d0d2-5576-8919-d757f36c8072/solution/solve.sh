#!/bin/bash
# OER-28 reference oracle. This is PHASE A. It runs inside the AGENT image, writes the
# graded artifact across the splice, and journals what it did for a human reader.
#
# The splice is /workspace/submission/plan.json. /workspace is the only path shared with
# the verifier; /app is this image's own and a plan left there is never graded.
#
# NOTHING THIS PHASE MEASURES CROSSES THE SPLICE AS EVIDENCE. The journal it writes to
# /logs/agent/phase_a.jsonl is for a reader; the verifier re-establishes equivalence on
# its own weights and re-times every plan from scratch. That is deliberate: a phase that
# could hand its own timings forward would be a phase that could assert its own reward.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURFACE="${OER_SURFACE:-/app}"
OUT="${OER28_SUBMISSION:-/workspace/submission/plan.json}"
JOURNAL="${OER28_JOURNAL:-/logs/agent/phase_a.jsonl}"

mkdir -p "$(dirname "$OUT")" "$(dirname "$JOURNAL")"

echo "[phase-a] writing reference execution plan to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module and run the SURFACE's own equivalence
# check. If the shipped schema would refuse this plan, or the plan is not equivalent to
# the frozen reference kernel, the oracle must fail here and be fixed rather than hand
# the verifier something it will refuse.
echo "[phase-a] validating the plan and re-establishing equivalence locally"
python3 - "$OUT" "$SURFACE" "$JOURNAL" <<'PY'
import json, sys, time
path, surface, journal = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, surface)
import harness, kernel_plan

doc = kernel_plan.load(path)
plan = kernel_plan.as_execution(doc)
spec = harness.load_spec()
print("[phase-a] plan validates against", kernel_plan.SCHEMA_ID)
print(json.dumps(plan, indent=2))

rows = [{"event": "plan_selected", "plan": plan, "at": time.time()}]

import torch
if not torch.cuda.is_available():
    print("[phase-a] no accelerator on this surface; skipping the local equivalence run.")
    print("[phase-a] the verifier establishes equivalence itself regardless.")
    rows.append({"event": "local_equivalence_skipped", "reason": "no-accelerator"})
else:
    harness.configure_backends()
    tokens = harness.to_device_tokens(
        harness.load_shard(f"{surface}/data/devset_slice.bin"), "cuda")
    bench = harness.Bench(spec, "cuda")
    control = kernel_plan.as_execution(kernel_plan.load(f"{surface}/default_plan.json"))
    report = bench.equivalence(plan, tokens)
    rows.append({"event": "local_equivalence", "worst_loss_abs_delta": report["worst_loss_abs_delta"],
                 "worst_grad_relative_delta": report["worst_grad_relative_delta"],
                 "equivalent": report["equivalent"]})
    print(f"[phase-a] equivalence loss_delta={report['worst_loss_abs_delta']:.6f} "
          f"(tol {report['loss_tolerance']})  grad_rel="
          f"{report['worst_grad_relative_delta']:.6f} (tol {report['grad_relative_tolerance']})")
    if not report["equivalent"]:
        raise SystemExit("[phase-a] REFUSING: this plan is not equivalent to the frozen "
                         "reference kernel; the verifier would refuse it at 0.0")
    lat = harness.interleaved_latencies(bench, {"control": control, "plan": plan},
                                        tokens, verbose=False)
    speedup = lat["control"]["min_ms"] / lat["plan"]["min_ms"]
    rows.append({"event": "local_timing", "control_min_ms": lat["control"]["min_ms"],
                 "plan_min_ms": lat["plan"]["min_ms"], "local_speedup": speedup,
                 "note": "phase-A measurement, for a reader; the verifier re-times from scratch"})
    print(f"[phase-a] local timing: control {lat['control']['min_ms']:.3f}ms, "
          f"plan {lat['plan']['min_ms']:.3f}ms, speedup {speedup:.3f}x")

with open(journal, "w", encoding="utf-8") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\n")
print(f"[phase-a] journal written to {journal}")
PY

echo "[phase-a] done; the plan is across the splice at $OUT"
