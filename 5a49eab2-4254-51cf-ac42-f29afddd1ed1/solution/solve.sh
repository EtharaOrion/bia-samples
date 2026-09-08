#!/bin/bash
# Reference oracle for OER-05. Runs inside the AGENT image and writes the graded
# artifact.
#
# The whole job is to put one validated update rule at
# /workspace/submission/update_rule.json. /workspace is the volume the verifier reads;
# /app is not shared with it.
#
# This oracle does not train. What it installs is the OUTPUT of a search that was run
# during authoring over this exact harness, this exact corpus and this exact budget,
# and solution/reference.py records that search and its measurements, so the artifact
# is traceable to evidence rather than asserted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OER05_SUBMISSION:-/workspace/submission/update_rule.json}"

mkdir -p "$(dirname "$OUT")"

echo "[solve] writing the reference update rule to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module, not a copy carried in solution/.
# If the shipped schema would refuse this document, the oracle must fail here and be
# fixed, rather than hand the verifier something it will refuse at grade time.
echo "[solve] validating against the shipped schema"
python3 - "$OUT" <<'PY'
import json, sys
sys.path.insert(0, "/app")
import rule_schema as schema
document = schema.load(sys.argv[1])
print("[solve] validates against", schema.SCHEMA_ID)
print(json.dumps(document, indent=2))
PY

echo "[solve] done"
