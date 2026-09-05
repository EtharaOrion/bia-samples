# FORGE-CANARY-BEGIN
# 0: 294e888ffd036059966722c59a8a0c6fee950b32ce494311ee94fda11ac0fd8c
# 1: 3a6af117ee548bb0f244f605e68db421c4007387d06727675ce633716a6256e7
# 2: 835bf507335e62beec7bcab02ac7e5260a1c5e3913a62bd2ce163f8554776d94
# 3: ff7149a1bb0e7002d4c8c6b2a3930886e69ed8bcf140463ab00ffc4c7709cbab
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
