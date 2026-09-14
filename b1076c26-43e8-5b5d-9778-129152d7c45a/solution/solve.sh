# FORGE-CANARY-BEGIN
# 0: d5c99c94032610373c24bd95c97b5e88a892a16bce41e31c3a48c97c04bbf4dc
# 1: c23d6170a1d28445d82665f40a3d9d0317cdc42255a976bcaa7c0c57cfb050bf
# 2: 4c79ce094c5d33ff5107d801bf3ca9d2ede753db0380c0d497a8b4fc8c2661f7
# 3: 3d6948f7b1f5a9f6c978fe0653cafe6ba3f97aaaaa1ca9cb89660f1d44c70f9f
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
