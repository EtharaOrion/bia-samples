# FORGE-CANARY-BEGIN
# 0: 0d73d1c1c474618d71571bf8afa73896f83dac2e25a0ea2aa3c06718d5d2c05a
# 1: 37dc97f08de8e4278f07aec3bf14a99c3e0425a3ecdbe468fe3ccf4863dd5664
# 2: 640cfce925ed1a0e0c9e3105a554aeae19780b90414a79a14c0e393632b7a69d
# 3: d4bacbee5079f4796e68ef9b06a5df762faeef4fcf946b1b6bb38e72591e5239
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
WORKSPACE="${OER26_WORKSPACE:-${PWD}}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
