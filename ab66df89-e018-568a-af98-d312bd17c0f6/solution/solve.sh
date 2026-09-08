#!/bin/bash
# Reference oracle. Runs inside the AGENT image and writes the graded artifact.
#
# The whole job is to put one validated plan at /workspace/submission/allocation.json.
# /workspace is the volume the verifier reads; /app is not shared with it.
#
# This oracle does not train at solve time. The plan it installs is the OUTPUT of a
# search run during authoring over this exact harness and this exact pool, and
# reference.py records that search, its measurements and the reasoning, so the
# artifact is traceable to evidence rather than asserted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURFACE="${OER16_SURFACE:-/app}"
OUT="${OER16_SUBMISSION:-/workspace/submission/allocation.json}"

mkdir -p "$(dirname "$OUT")"

echo "[solve] writing reference allocation to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module, not a copy carried in solution/.
# If the shipped schema would refuse this plan, the oracle must fail here and be
# fixed, rather than hand the verifier something it will refuse at grade time.
echo "[solve] validating against the shipped schema"
python3 - "$OUT" "$SURFACE" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[2])
import harness, allocation_schema
spec = harness.load_spec()
alloc = allocation_schema.load(sys.argv[1], spec, harness)
per = harness.micro_batch_flops(spec, alloc["model"])
spent = alloc["micro_steps"] * per
budget = int(spec["budget"]["flop_budget"])
print("[solve] allocation validates against", allocation_schema.SCHEMA_ID)
print(f"[solve] {alloc['model']}  micro_steps {alloc['micro_steps']}  "
      f"grad_accum {alloc['grad_accum']}")
print(f"[solve] spends {spent} of {budget} FLOPs, {100.0 * spent / budget:.1f}% of the budget")
print(json.dumps(alloc, indent=2))
PY

echo "[solve] done"
