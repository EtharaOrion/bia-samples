# FORGE-CANARY-BEGIN
# 0: 33725ea87e99b6a28ea0d1b07582ef402e4672e487f1dbaca5c0ca1bae35e19e
# 1: eb677e04b593fcf0349dda969444109543c2ed2970e4ac5ad0f79e19031684b5
# 2: d2943541d3556b2980bffc11be3e60cd740f023c488f35e8e751a1b6521806e6
# 3: d0e043710b47f5df610127d03087494a8641678b4ce18fc2034e9380d26e5ab9
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
