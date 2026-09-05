# FORGE-CANARY-BEGIN
# 0: aedc34a02b126068690e38795d347b2e3553e629082899d0db669f7e865a7acf
# 1: b6ea78ba16ef101d840026209ea03b08b6a5db2210267d4e968bb2c2117d186f
# 2: 22309efe234d47bfe81524b448e2072088793c9f6613bbe0ea84406d3713094d
# 3: 7494f0a86237b6438397787307a6f119122ba37ed7e26c9b36c53a83abbc4453
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
