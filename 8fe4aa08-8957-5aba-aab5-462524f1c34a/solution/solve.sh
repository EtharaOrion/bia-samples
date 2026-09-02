# FORGE-CANARY-BEGIN
# 0: 5f47109db1b624f4b1d94c654ef1bf8bbaa45b4e058b65abc5121b2a5091139e
# 1: c4c493c30628b0c55659c79940f5e6f80000d2deb3452b9832a507f47e79d16d
# 2: 81a65aba03c1ea6faa3e3878633d004e52886f7d7ef3866ca2ce4dde4723b581
# 3: 760a14dd13b75953866e41565859282204eef81ad78b8604012b9dc828419ec7
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
