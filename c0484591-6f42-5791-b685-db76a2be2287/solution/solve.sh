# FORGE-CANARY-BEGIN
# 0: 8dfab59683a6109af7b225695082ed5c36e0b760f702306a4879304ca8e72260
# 1: b1cdc756c627338cfd1d2b89f49f15ca13a60a3aa231591c6ca578aa88e3febd
# 2: 1160211d13739bf563cc70492a9c5f313ab3e0d1c82cfbf265171d3f30d2a46f
# 3: 83f77687cab068ccd26207d1295d40e74a71d01f6b64828472874edef58cf254
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
