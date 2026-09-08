#!/bin/bash
# OER-09 reference oracle. Runs inside the AGENT image and writes the graded artifact.
#
# The whole job is to put one validated curation chain at
# /workspace/submission/filter.json. /workspace is the volume the verifier reads;
# /app is not shared with it, so a chain left in /app is never graded.
#
# This oracle does not train. The chain it installs is the OUTPUT of a search that
# was run during authoring against this exact register and harness, and
# reference.py plus solution/grounding.md record that search, its measurements and
# the reasoning, so the artifact is traceable to evidence rather than asserted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURFACE="${OER_SURFACE:-/app}"
OUT="${OER09_SUBMISSION:-/workspace/submission/filter.json}"

mkdir -p "$(dirname "$OUT")"

echo "[solve] writing reference curation chain to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module, not a copy carried in
# solution/. If the shipped schema would refuse this chain, the oracle must fail
# here and be fixed, rather than hand the verifier something it will refuse.
echo "[solve] validating against the shipped schema and running the shipped filter runner"
python3 - "$OUT" "$SURFACE" <<'PY'
import json, sys
path, surface = sys.argv[1], sys.argv[2]
sys.path.insert(0, surface)
import filter_schema, harness
chain = filter_schema.load(path)
register = filter_schema.load_register(f"{surface}/pool/source_register.jsonl")
admitted, report = filter_schema.apply(chain, register)
budget = harness.load_spec()["budget"]["blocks"]
print("[solve] chain validates against", filter_schema.SCHEMA_ID)
print("[solve] runner report:", json.dumps(report))
if report["rules_unresolved"]:
    raise SystemExit("[solve] REFUSING: the chain has unresolved predicates; "
                     "the runner would have failed open and curated nothing")
if len(admitted) < budget:
    raise SystemExit(f"[solve] REFUSING: {len(admitted)} blocks admitted, "
                     f"the budget consumes {budget}")
print(f"[solve] {len(admitted)} blocks admitted against a budget of {budget}")
PY

echo "[solve] done"
