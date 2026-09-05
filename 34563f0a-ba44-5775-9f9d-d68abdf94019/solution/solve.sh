# FORGE-CANARY-BEGIN
# 0: 00f26333200c1098e49de1c35090e836973d9026bac822e180623c8bfc2938a5
# 1: b782ddafb8402527cabe0efb7ddb382d4110eede39afb93c756352a93496b7c9
# 2: 3cd26b46a6d12905c89d3e50a58a57f6960c472a8ed6c6e1c6e82092638e122e
# 3: 990d138a5a268825cae266bc1c902dbcd1b15843653fea943e0f2dc4f04c9999
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
