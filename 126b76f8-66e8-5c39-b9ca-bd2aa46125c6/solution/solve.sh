# FORGE-CANARY-BEGIN
# 0: 6c5be240343e11d19c547031df50e3895fbb3aac9b31b7949e4cb95737136199
# 1: abbb0625df9829ddbb6c95da1b403478657afa730afb2ce1db0353a9f7867bbc
# 2: 542c651908356bd3e3041191f99b72cb8a797a30bdf2784ce8700bb6b26d7d1e
# 3: 449454f05796cfd420a0d7e97f820341551637e78dcb5181b21afa76776f3fa9
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
