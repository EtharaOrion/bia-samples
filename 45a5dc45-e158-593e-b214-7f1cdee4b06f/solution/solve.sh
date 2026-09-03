# FORGE-CANARY-BEGIN
# 0: 1767da28f53f4bbd01d88e9e0b02a4ad33876bf087396bc474c29666e5fa6813
# 1: 78f1d70414797df462b8c6fb81a4e07a13b26f4051ae881a4c673bbfbe1b9e8d
# 2: 4287cce1a097dc2da9220eecf266279a268775e81ded46372c2b3d79487dd58d
# 3: d3b38e0080344ae9683cb53917f50812f71cae7034d1892e858d12774708b732
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
