# FORGE-CANARY-BEGIN
# 0: 2d635c3768ffdd23682e4f2ff419db3cf0bb99faa1888f25b28d1695e29e2a3b
# 1: fd245699a038f02bd68bb9a731890cb252f500bac34d4874cdc8cb5d27532343
# 2: 656b2b1f34224608ce40faf4c333ff1580e95e82bc0555e062768e124cac9ba4
# 3: c21d607d2996cd4a5b6b3008ecea169a3ee561182e263644bd6fddb0ac0530aa
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with: python3 solution/recompute.py
#
# Oracle entry point for slot OER-03.
#
# Installs the reference solution as the submission and runs it under the
# pinned harness. The reference crosses at step 2600 on the verifier's
# cadence and carries a schedule of 3000 steps, so 400 steps of schedule sit
# past the graded point and buy exactly nothing. That is the point of the
# reference, not an accident in it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname "$HERE")"

mkdir -p /app
cp "$HERE/reference.py" /app/submission.py
cp "$BUNDLE"/environment/bia_harness.py "$BUNDLE"/environment/bia_loader.py "$BUNDLE"/environment/shape.json /app/env/ 2>/dev/null || true

cd /app
python3 /app/submission.py

# The reference never decides when it crossed and never reports a crossing.
# The verifier recomputes the graded step from its own evaluations.
