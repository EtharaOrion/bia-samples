#!/bin/bash
# Reference oracle. Runs inside the AGENT image and writes the graded artifact.
#
# The whole job is to put one validated retention selection at
# /workspace/submission/selection.json. /workspace is the volume the verifier reads;
# /app is not shared with it, so a selection left in /app is never graded.
#
# This oracle does not train and does not evaluate. The selection it installs is the
# OUTPUT of a search that was run during authoring over this exact harness and this exact
# frozen run, and reference.py records that search, what it measured and what it
# concluded, so the artifact is traceable to evidence rather than asserted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURFACE="${OER_SURFACE:-/app}"
OUT="${OER_SUBMISSION:-/workspace/submission/selection.json}"

mkdir -p "$(dirname "$OUT")"

echo "[solve] writing reference selection to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module, not a copy carried in solution/.
# If the shipped schema would refuse this selection -- for exceeding the storage budget,
# for weights that do not sum to one -- the oracle must fail HERE and be fixed, rather
# than hand the verifier something it will refuse at grade time.
echo "[solve] validating against the shipped schema"
python3 - "$OUT" "$SURFACE" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[2])
import selection_schema
selection = selection_schema.load(sys.argv[1])
print("[solve] selection validates against", selection_schema.SCHEMA_ID)
print(json.dumps({"keep": selection["keep"], "weights": selection["weights"]}, indent=2))
PY

echo "[solve] done"
