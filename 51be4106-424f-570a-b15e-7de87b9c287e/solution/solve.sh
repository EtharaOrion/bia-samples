#!/bin/bash
# Reference oracle. Runs inside the AGENT image and writes the graded artifact.
#
# The whole job is to put one validated recipe at /workspace/submission/recipe.json.
# /workspace is the volume the verifier reads; /app is not shared with it.
#
# This oracle does not train. The recipe it installs is the OUTPUT of a search that
# was run during authoring -- roughly forty training runs of this exact harness on
# this exact corpus -- and reference.py records that search, its measurements and the
# reasoning, so the artifact is traceable to evidence rather than asserted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURFACE="${OER_SURFACE:-/app}"
OUT="${OER_SUBMISSION:-/workspace/submission/recipe.json}"

mkdir -p "$(dirname "$OUT")"

echo "[solve] writing reference recipe to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module, not a copy carried in solution/.
# If the shipped schema would refuse this recipe, the oracle must fail here and be
# fixed, rather than hand the verifier something it will refuse at grade time.
echo "[solve] validating against the shipped schema"
python3 - "$OUT" <<'PY'
import json, sys
sys.path.insert(0, "/app")
import recipe_schema
path = sys.argv[1]
recipe = recipe_schema.load(path)
print("[solve] recipe validates against", recipe_schema.SCHEMA_ID)
print(json.dumps(recipe, indent=2))
PY

echo "[solve] done"
