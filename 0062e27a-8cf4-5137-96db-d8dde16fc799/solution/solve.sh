# FORGE-CANARY-BEGIN
# 0: ddbf3aad2e6b81fa779f2ad1d69397c3a6d5676b69fc985908b4aea85b79fb72
# 1: c1dadc3de63ae6e96c207c030068315c29d920a321afd96a493039999e348c9f
# 2: c552c923c95961de29fc640b7dfd71d1794c63b1b8579634f014a016dd751861
# 3: e03a7f5b75ba71b9b094687b591d202cb857a9578b6da84d3234a30cb3dfe175
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
