# FORGE-CANARY-BEGIN
# 0: f25086df15a144699e3b96a881c2cbd88eaf47d7c3e2c8b3f796f1ad2b6d3760
# 1: 9169a8c59c60f1c3d41ee3ecfeffb638d3eba3f56cfcb03437f9f33d36fe7f2e
# 2: 1bef91314b6e2674dfcbe7e99d1140c682221549ba109365b042119c58589a3b
# 3: 49b0efd5e5d187cfe8390e9534aebdab46d0c7817ad9601002ce66c7acacf851
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
