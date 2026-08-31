#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml
# Run the reference pipeline over the frozen corpus and emit the graded deliverables.
set -euo pipefail

ENV_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../environment" && pwd)}"
OUT_DIR="${2:-/app}"
SCHEMA_VERSION="${OER12_PARSE_SCHEMA_VERSION:-2}"
mkdir -p "$OUT_DIR"
python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/reference.py" "$ENV_DIR" "$OUT_DIR" "$SCHEMA_VERSION"
