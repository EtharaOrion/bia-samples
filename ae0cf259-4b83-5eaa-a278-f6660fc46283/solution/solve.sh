# FORGE-CANARY-BEGIN
# 0: f270f3e0f851ebb05315e90e5ecdc9b09dd69773b4435f4d84ff66a4cc089c21
# 1: 1b151f23979288f47144cdeaef22c70158d0b657acb1904020a55a3a6f382a43
# 2: 26e46c63f07110accfa281930ace62ef32fe962194498e87b612c74407eb5a30
# 3: 9d6745b35434535e24c2aeb59ccc68d1c83023d8533d34e7d5503511a07413db
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
