# FORGE-CANARY-BEGIN
# 0: dcc242eaf0d6c35bd25158829a6cd99101a6086f4ec11cdd86b9872f62660baa
# 1: dd3ff6f3e7f20d859cc27abbabf49b0f54da6a2795b8593a1f578c2aab8f2d96
# 2: a1c73436461003bb90a6366f8334e56935b8f638127cb53fb419130c4553f071
# 3: 16652a290c585e1c0a9b769b0554a162dbc51a7fa510c22b13a54bdf62cb45d1
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
