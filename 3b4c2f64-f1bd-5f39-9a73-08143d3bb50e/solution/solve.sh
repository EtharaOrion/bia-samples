# FORGE-CANARY-BEGIN
# 0: 00a5d9cdd074d470cd5063a531502cf7b7775d34afdbe7b332c946a3f4f9d301
# 1: 54984cc0c28e60652f181b50fd3c12a8c33c16fcbb82c7b51408b147e600cbfd
# 2: 73b9aca36f4d18278d372491345922645035c438b3b6bf8536e384615c664c8f
# 3: ae97ac52effb0c0d6680cd3b01065743878dfc8395490c03fc9f8dd0247613a4
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
