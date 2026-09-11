# FORGE-CANARY-BEGIN
# 0: c00ac766d16ba82061d3da7e5d5069e8a2c7c7a370ca52807fa32a67663dd3f4
# 1: 8b385cb2f5c124b9607a21c2373ece53b9b472c32d432772e171588b58ff47b6
# 2: 5d1a351111ff56ae34bf185ee6716b66e1f95e57ca4fcdc6e071471ce1e33493
# 3: f0f3941d4000144938b2ad245dbc905d65ac46c2617fec8a16c9f4124603fb0b
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It opens the harness handle onto the BUILT fence state, walks the
# producer chain once, enforces the fence over what the traversal presented, and writes
# submission.json. It reads no answer file and hardcodes neither discovery value.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${OER26_WORKSPACE:-/workspace}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
