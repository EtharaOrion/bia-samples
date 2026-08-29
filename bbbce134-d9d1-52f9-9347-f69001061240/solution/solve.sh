# FORGE-CANARY-BEGIN
# 0: 53f07487181ce9e71429a527bd95c30d2f56098609249768e7661aabb51fba4f
# 1: 766c9660043aef74dcae1919ac80ee14752c7760d4f3f528036bdb76aaee3f8b
# 2: 1dbbb0f861417ef4ff1324747a01e2c0a9e4e5d941c9be1d49032db01067d9a8
# 3: a43a85cec6e1448ea3913efa70a2453109f005730aa92b5526e81feb3d283449
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
