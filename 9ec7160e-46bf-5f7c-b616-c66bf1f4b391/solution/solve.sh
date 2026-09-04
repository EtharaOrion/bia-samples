# FORGE-CANARY-BEGIN
# 0: 04eb2f39224e9de1cbc6e288f430de10abffa99d12085ac3f992032c70d781f4
# 1: d9190a1c1b340f451d6abbfb5c5e4db65862b4f707659927945e751e55e5353e
# 2: d1fa32f0897244056e30387ed1777563c15a203cd19849b9cbf57844b4d8afed
# 3: 86b27b0ea9221a5e4ecc358c62f2a85a8b7415fa251e7acdeb3ee16ecc2ca07e
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
