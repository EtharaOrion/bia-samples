# FORGE-CANARY-BEGIN
# 0: daf4f2bdef38568e96177986fa8868392c6b61112ef5844c3c76ef04491748f3
# 1: c9cd63dafa2ad2fc3d6772f274c7b2bc464b67df832af144ec39c34758ee8a39
# 2: a53aa679573f70b5b671107305394755d04718e5a6dabc381dbb1efc41b32471
# 3: 5e21b2895b398a4e44ace347f824e35fd28159f1db3f3dd251cb65d8313747f1
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
