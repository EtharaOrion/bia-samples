# FORGE-CANARY-BEGIN
# 0: cd38bfad0eaf810eb54ce5b626ac212abadd480e72218aeb8b6fb9e559ea8bd0
# 1: c780e9ee7f24dd27df29058202e09bafd8bda5098e114c3aae9823c9ff3c7fc8
# 2: 12e9538f570190078c2532cf77a31d3e550d10bf20742112e70e6b1c5eb236b4
# 3: 5ffdb99cb35c23877bc139c97d71a312e594c973d012de7bf3d343e6b9ee7246
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
