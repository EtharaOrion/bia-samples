#!/bin/bash
# OER-01 reference oracle. Runs inside the AGENT image and writes the graded artifact.
#
# The whole job is to put one validated recipe at /workspace/submission/recipe.json.
# /workspace is the volume the verifier reads; /app is not shared with it.
#
# This oracle does not train. The recipe it installs is the OUTPUT of a search run
# during authoring over this exact harness; solution/grounding.md records that
# search and its measurements.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURFACE="${OER_SURFACE:-/app}"
OUT="${OER01_SUBMISSION:-/workspace/submission/recipe.json}"

mkdir -p "$(dirname "$OUT")"

echo "[solve] writing reference recipe to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module AND its own record corpus. If the
# shipped schema would refuse this recipe, or if it restates a published record, the
# oracle must fail here and be fixed rather than hand the verifier something it will
# refuse at grade time.
echo "[solve] validating against the shipped schema and the published-record corpus"
python3 - "$OUT" "$SURFACE" <<'PY'
import json, sys
path, surface = sys.argv[1], sys.argv[2]
sys.path.insert(0, surface)
import recipe_schema, records
recipe = recipe_schema.load(path)
corpus = records.load_records(f"{surface}/published_records.json")
replay = records.find_replay(recipe, corpus)
if replay is not None:
    raise SystemExit(f"[solve] REFUSING: this recipe restates published record "
                     f"{replay.get('record_id')!r}; the verifier would refuse it")
print("[solve] recipe validates against", recipe_schema.SCHEMA_ID)
print(f"[solve] digest {records.digest(recipe)} is not among the "
      f"{len(corpus)} published records")
print(json.dumps({k: v for k, v in recipe.items() if k != "notes"}, indent=2))
PY

echo "[solve] done"
