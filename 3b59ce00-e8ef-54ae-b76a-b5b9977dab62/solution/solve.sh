#!/bin/bash
# Reference oracle. Runs inside the AGENT image and writes the graded artifact.
#
# The whole job is to put one validated policy at
# /workspace/submission/policy.json. /workspace is the volume the verifier reads;
# /app is not shared with it.
#
# This oracle does not train. The plan it installs is the OUTPUT of a search that was
# run during authoring -- seventeen training runs of this exact harness on this exact
# corpus at this exact budget -- and reference.py records that search, its
# measurements and the reasoning, so the artifact is traceable to evidence rather
# than asserted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OER_SUBMISSION:-/workspace/submission/policy.json}"

mkdir -p "$(dirname "$OUT")"

echo "[solve] writing reference policy to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module, not a copy carried in solution/.
# If the shipped schema would refuse this plan, the oracle must fail here and be
# fixed, rather than hand the verifier something it will refuse at grade time.
echo "[solve] validating against the shipped schema"
python3 - "$OUT" <<'PY'
import json, sys
sys.path.insert(0, "/app")
import policy_schema
policy = policy_schema.load(sys.argv[1])
print("[solve] policy validates against", policy_schema.SCHEMA_ID)
print(json.dumps(policy, indent=2))
PY

echo "[solve] done"
