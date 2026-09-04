# FORGE-CANARY-BEGIN
# 0: 8b30a3657d22b5185654e1467e05f90935200d09a7194cba6e4feec11370dd32
# 1: 2abc11b92fbb27fbf29f0d5baf81c6c33aa2a248733238768d3bf0381dad29ba
# 2: bc660a0ea3f97066d0a55b54c46016c7af17b2d5b1eafa799ca03f7629257cbb
# 3: 99adfe546a60ba5a609e8ae40af30c2868a69aed89c5fdedafa5e5a52b37f7aa
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Derived from solution/grounding.yaml by solution/recompute.py.
#
# The reference solution is a refinement POLICY, not a configuration. The
# harness invokes it once per attempt with state.json beside it, and it
# writes proposal.json. This script only puts it where the harness looks.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${BIA_SUBMISSION:-/workspace/refine.py}"
mkdir -p "$(dirname "${TARGET}")"
cp "${HERE}/reference.py" "${TARGET}"
chmod +x "${TARGET}"
echo "reference policy installed at ${TARGET}"
