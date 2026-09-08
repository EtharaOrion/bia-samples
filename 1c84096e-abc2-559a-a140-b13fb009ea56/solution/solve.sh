#!/bin/bash
# Reference oracle. Runs inside the AGENT image and writes the graded artifact.
#
# The whole job is to put one validated policy at /workspace/submission/policy.json.
# /workspace is the volume the verifier reads; /app is not shared with it.
#
# This oracle does not train at solve time. The policy it installs is the OUTPUT of a
# search that was run during authoring over the curves probe_local.py produces on this
# exact substrate, for every declared ramp start, and reference.py records that search,
# its measurements and the reasoning, so the artifact is traceable to evidence rather
# than asserted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURFACE="${OER12_SURFACE:-/app}"
OUT="${OER12_SUBMISSION:-/workspace/submission/policy.json}"

mkdir -p "$(dirname "$OUT")"

echo "[solve] writing reference policy to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module, not a copy carried in solution/.
# If the shipped schema would refuse this policy, the oracle must fail here and be
# fixed, rather than hand the verifier something it will refuse at grade time.
echo "[solve] validating against the shipped schema"
python3 - "$OUT" "$SURFACE" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[2])
import policy_schema
policy = policy_schema.load(sys.argv[1])
spent = sum(p["tokens"] for p in policy["probes"])
print("[solve] policy validates against", policy_schema.SCHEMA_ID)
print(f"[solve] {len(policy['probes'])} probes, {spent}/{policy_schema.EVAL_TOKEN_BUDGET} tokens, "
      f"rule {policy['selection']['rule']}")
print(json.dumps(policy, indent=2))
PY

echo "[solve] done"
