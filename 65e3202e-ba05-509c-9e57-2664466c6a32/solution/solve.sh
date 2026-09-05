# FORGE-CANARY-BEGIN
# 0: 63b2cdc7932f124d8ef893fcaf6e4822576227c9b0351f480e2abebcec0d1b67
# 1: a53ec68a4303aa78388c3a3e86f16c2401ba3aba38649a0d675e76214d87fcab
# 2: cae4025a2ae34e284437eb2c4c38574952a4f0d6af23d7dc3db1c1bba12aa061
# 3: 31caf4c49874720b6ae3bc6c619df049683f7ed81aed2330e81335ba1b42f72e
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
