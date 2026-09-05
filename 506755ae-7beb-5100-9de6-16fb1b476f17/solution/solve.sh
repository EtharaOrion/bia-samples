# FORGE-CANARY-BEGIN
# 0: 7cd896c86e5d957f385a5bc0392cdc2de87468b0986ed54a4ffb29336218d5f2
# 1: bc0ff85a5952fbff03ce2b49dc906a7146280b18b709c86aa2a0a1ab2b5cd051
# 2: f5ce3f95cd187ccc0d720ba43e221c49fb2abbc9435bf4e69c3ac52fffe22c43
# 3: 29da9211cbdf254ae7a7bbe20998979fae8474d7abe9a0c4c62e6f5e0c75c603
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
