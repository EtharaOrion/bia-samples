# FORGE-CANARY-BEGIN
# 0: 6229e5977287a91b9cd03488336b09d371cddc4974243efc6453757124cd7ebf
# 1: cf86ce44760e9a774418462f69c83c1a4e70a04a060ccd34dd33231dcb34c044
# 2: 539a246e3d49c84b4e0284c0f63517903b59bbee3027fb21fb8791b0f2e2bcc9
# 3: db1c65b8327b7a829f69a63aea2839914c1786302330f00f2979e12c938cfc82
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with: python3 solution/recompute.py
#
# The solver entry point. It installs the reference update rule over the
# one file the agent is allowed to edit and does nothing else, because the
# frozen axes are frozen and this script must not touch one.
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${OER05_WORKSPACE:-${BUNDLE}/environment}"

# The rule class is SignGatedMedianAnchorRule, exposing build_update_rule.
install -m 0644 "${BUNDLE}/solution/reference.py" "${WORKSPACE}/update_rule.py"

echo "installed the reference update rule at ${WORKSPACE}/update_rule.py"
