# FORGE-CANARY-BEGIN
# 0: 8d0c90641cd43c2eaab54baf1dd2a599475a0a17aa122154a44d99db6a4d5368
# 1: 2eec4a39c261d565abf91bbe0b388a30869e8941f232e80775d7ce60757e822d
# 2: 2b05a28c05ef589daf442a047ed5d651b6c9904bb4e28fe2a568a534e7c96709
# 3: 8100979d8982a331077a4d8710c49a6120a55e5d2da9d4cbf0cd5e1a6016cfa2
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml.
#
# The reference session policy is a single self-contained file, because the
# harness copies the submission alone into a fresh directory and runs it there.
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${OER06_SUBMISSION:-/app/submission.py}"

mkdir -p "$(dirname "$TARGET")"
cp "$SOLUTION_DIR/reference.py" "$TARGET"
chmod 0644 "$TARGET"

echo "installed reference session policy at $TARGET"
