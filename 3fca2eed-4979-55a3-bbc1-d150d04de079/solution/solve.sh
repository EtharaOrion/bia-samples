# FORGE-CANARY-BEGIN
# 0: 7a314bb4c4b2353fc5f0e0be4e0c3c353bcb9d6fba3c7cf3c4299d476d981130
# 1: 1ec08f0d42698dc7733e694467a26766ed6060b09d4868c1a31af7de72850d30
# 2: 65c5ba8a374bdc535e8a8e8218c80579528bf5c43ecbda7c6b6929605f040367
# 3: 8582b42ddeac5ae9f43e793e171489fe4066d9b2611a273b7f26bf05778fe481
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
