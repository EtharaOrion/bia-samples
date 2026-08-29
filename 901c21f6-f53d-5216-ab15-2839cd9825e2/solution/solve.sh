# FORGE-CANARY-BEGIN
# 0: b841af234553d162fde8f8c9ff66c9942c6678c7d2ff614c7db221adfde65966
# 1: 239e3435084d6db402cba47e8fba7b66b4f1407cba0582882cc82af26c116694
# 2: 68465b51c3b012d13e5585f4dcf3729dffbd57fadb86050308d18d571791b2c3
# 3: 4449035164f8850342ccb5aa779970d2bcf58c0cb962f9c765071d22bb43ef47
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
