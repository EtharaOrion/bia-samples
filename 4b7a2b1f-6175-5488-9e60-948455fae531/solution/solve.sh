#!/bin/bash
# Reference oracle. Runs inside the AGENT image and writes the graded artifact.
#
# The whole job is to put one validated plan at /workspace/submission/plan.json.
# /workspace is the volume the verifier reads; /app is not shared with it.
#
# This oracle does not train at solve time. The plan it installs is the OUTPUT of a
# search run during authoring over this exact harness and this exact pool, and
# reference.py records that search, its measurements and the reasoning, so the
# artifact is traceable to evidence rather than asserted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURFACE="${OER08_SURFACE:-/app}"
OUT="${OER08_SUBMISSION:-/workspace/submission/plan.json}"

mkdir -p "$(dirname "$OUT")"

echo "[solve] writing reference plan to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module, not a copy carried in solution/.
# If the shipped schema would refuse this plan, the oracle must fail here and be
# fixed, rather than hand the verifier something it will refuse at grade time.
echo "[solve] validating against the shipped schema"
python3 - "$OUT" "$SURFACE" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[2])
import harness, plan_schema
spec = harness.load_spec()
plan = plan_schema.load(sys.argv[1], spec)
total = sum(d["tokens"] for d in plan["draws"])
print("[solve] plan validates against", plan_schema.SCHEMA_ID)
print(f"[solve] {len(plan['draws'])} draws, {total} tokens, budget "
      f"{spec['budget']['total_train_tokens']}")
print("[solve] order:", [d["source"] for d in plan["draws"]])
print(json.dumps(plan, indent=2))
PY

echo "[solve] done"
