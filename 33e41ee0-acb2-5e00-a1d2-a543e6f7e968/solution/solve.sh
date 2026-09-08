#!/bin/bash
# OER-04 reference oracle. Runs inside the AGENT image and writes the graded artifact.
#
# The whole job is to put one validated shape at /workspace/submission/shape.json.
# /workspace is the volume the verifier reads; /app is not shared with it.
#
# This oracle does not train. The shape it installs is the OUTPUT of a search run
# during authoring over this exact substrate; solution/grounding.md records that
# search and its measurements.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURFACE="${OER_SURFACE:-/app}"
OUT="${OER04_SUBMISSION:-/workspace/submission/shape.json}"

mkdir -p "$(dirname "$OUT")"

echo "[solve] writing reference shape to $OUT"
python3 "${HERE}/reference.py" --out "$OUT"

# Validate through the SURFACE's own schema module and, critically, INSTANTIATE the
# shape and count its parameters the same way the verifier does. If the shipped
# schema or the band would refuse this shape, the oracle must fail here and be
# fixed, rather than hand the verifier something it will refuse at grade time.
echo "[solve] validating against the shipped schema and counting the instantiated model"
python3 - "$OUT" "$SURFACE" <<'PY'
import json, sys
path, surface = sys.argv[1], sys.argv[2]
sys.path.insert(0, surface)
import harness, shape_schema
spec = harness.load_spec()
shape = shape_schema.load(path, spec["shape_bounds"])
model, count = harness.build_model(shape, spec, device="cpu")
shape_schema.parameter_band_check(count, spec["parameter_band"])
band = spec["parameter_band"]
print("[solve] shape validates against", shape_schema.SCHEMA_ID)
print(json.dumps({k: v for k, v in shape.items() if k != "notes"}, indent=2))
print(f"[solve] instantiated parameters {count} inside [{band['min']}, {band['max']}]")
print(f"[solve] num_heads {shape['model_dim'] // shape['head_dim']}")
PY

echo "[solve] done"
