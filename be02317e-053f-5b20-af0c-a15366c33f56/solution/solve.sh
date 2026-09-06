# FORGE-CANARY-BEGIN
# 0: a5e8805416b50dd1f7b109be4c3bfb773fc52b8df31cb4ee1d45703403eeec7b
# 1: ab41d437ba8efce3f93a5004b071852bc46dadfd9bcb764e4cbf918edf05709f
# 2: c82c8cf9dc1437e3f88c74a544f226996b551aecc6a41a90544ebd9d81f70d83
# 3: 3314cd0fb7835d9c0835b5ca98d49a76352a6a361116261aa3392c08e4d38e06
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
