# FORGE-CANARY-BEGIN
# 0: c90600d8d43a5556724168ab940056b382daba30ba57053c99eb468d634418d1
# 1: c184f95cc96981e758138402657e14ba9e8eb3d8e60a69f872a12cb9f6f4f517
# 2: 4c6162b182b6b9bb60a556c294389a7b605fa8f05f33b1bf934b28325b164bea
# 3: 999bec1a2d626019d1c07131eb4794467a703d799cbdd53a48d0db7f798f2870
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml
# Run the reference pipeline over the frozen corpus and emit the graded deliverables.
set -euo pipefail

ENV_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../environment" && pwd)}"
OUT_DIR="${2:-/app}"
SCHEMA_VERSION="${OER12_PARSE_SCHEMA_VERSION:-2}"
mkdir -p "$OUT_DIR"
python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/reference.py" "$ENV_DIR" "$OUT_DIR" "$SCHEMA_VERSION"
