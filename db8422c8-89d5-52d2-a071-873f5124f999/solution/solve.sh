# FORGE-CANARY-BEGIN
# 0: a0fc33d6c80b1f33b2bf1b423ed5fb5edf4e5dc9ca516ecce0cf03d0a3e2ae78
# 1: 23dc233c85dac20b62efa4554cd14b860da4929eb1fbe1d9bdc6f0bcff2c813a
# 2: b88017b2b79d5f98f95b33bacb821132199133ff47e336deeb6030f37afea642
# 3: bb38b96073302685c3cc70cf535addfc2a20edcb36d5130b5d6f9b280b3cc5f4
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
