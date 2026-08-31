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
