#!/bin/bash
# Reference oracle. Runs inside the AGENT image and writes the graded artifact.
#
# The whole job is to put one validated batch-geometry plan at
# /workspace/submission/packing.json. /workspace is the volume the verifier reads;
# /app is not shared with it.
#
# This oracle does not train. The plan it installs is the OUTPUT of a search that was
# run during authoring -- twenty training runs of this exact harness on this exact
# corpus at this exact budget -- and reference.py records that search, its
# measurements and the reasoning, so the artifact is traceable to evidence rather
# than asserted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OER_SUBMISSION:-/workspace/submission/packing.json}"

mkdir -p "$(dirname "$OUT")"

echo "[solve] writing reference plan to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module, not a copy carried in solution/.
# If the shipped schema would refuse this plan, the oracle must fail here and be
# fixed, rather than hand the verifier something it will refuse at grade time.
echo "[solve] validating against the shipped schema"
python3 - "$OUT" <<'PY'
import json, sys
sys.path.insert(0, "/app")
import harness, packing_schema
spec = harness.load_spec()
total = int(spec["budget"]["total_train_tokens"])
plan = packing_schema.load(sys.argv[1], total)
phases = harness.expand(plan, spec)
consumed = sum(p["tokens"] for p in phases)
print("[solve] plan validates against", packing_schema.SCHEMA_ID)
print("[solve] phases:", " -> ".join(
    f"{p['rows']}x{p['seq_len']}x{p['grad_accum']}:{p['optimizer_steps']}st" for p in phases))
print("[solve] consumes", consumed, "of", total, "budget tokens")
assert consumed > 0.99 * total, "the oracle plan does not consume the budget"
print(json.dumps(plan, indent=2))
PY

echo "[solve] done"
