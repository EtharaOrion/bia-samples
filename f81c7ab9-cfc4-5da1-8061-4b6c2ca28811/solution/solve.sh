# FORGE-CANARY-BEGIN
# 0: 3cb0b5bdc46482e279425152010531d33f1de3470297a1425640343c14980609
# 1: bf9ec05a1558d92b44162573676b979b64d03e253b7d94cbe00744a5d565505d
# 2: 6c5289fb974f8ce7dd5c83368e68e0faa25449a7de8f88bdab818e45d8f798c8
# 3: 168b5e4126cc5c8245eb89e988aa5ea4e0925caae8e6e05feda3d7ef82e873d4
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
