# FORGE-CANARY-BEGIN
# 0: a9ff9113b0d0c8846aa733f7668f74a4faa92af36a48b5781623074b27118983
# 1: 158db7e17a53303305a1f3422ada1f22d29042b692b2d6aca97493def22f5164
# 2: 4db90a2dada9d862c14d37d798920b834b128dbb1130b40144e5e735192a111f
# 3: 95cad99235df886eac6ccd9b8618efcc5f42b4bea82997decb452acc6cb5eb48
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml.
#
# The reference session policy is a single self-contained file, because the
# harness copies the submission alone into a fresh directory and runs it there.
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${OER06_SUBMISSION:-/workspace/submission.py}"

mkdir -p "$(dirname "$TARGET")"
cp "$SOLUTION_DIR/reference.py" "$TARGET"
chmod 0644 "$TARGET"

echo "installed reference session policy at $TARGET"
