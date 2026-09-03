# FORGE-CANARY-BEGIN
# 0: b0f853f55b050f5070c4474393b7d8765a2d72fc946cb60fc97f0b0448066b73
# 1: 5b41c3d20ea448123db38cbc0c0cb707a119d1297e49c1b069aa437654490974
# 2: 93c7768633520b136ded24fa4a8399b5734986f5f0886617b11a7c7fa9cf5675
# 3: 9c9c64bd51e784144f95732f0eba6449f025421489fb0eac963ac222b8fdfa93
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
