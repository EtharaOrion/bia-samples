# FORGE-CANARY-BEGIN
# 0: 5b9e9c0cecf191e4b71d1039d91f44bf0ffd9349110d4b143bc2350638adf851
# 1: e044be4f4d136d1cb239969cbbf7db87ad986d7b3537d18aaeda85a350adce07
# 2: 6c867e744983485918b4c6ec5cd3bb8e2901bd8791a2d3cc46846e2ee7ee7ba9
# 3: 8ac7cc1d64f535b7d51843a242a81ad04a7430392c1360bceecb1ed16f3f2aee
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
