# FORGE-CANARY-BEGIN
# 0: a285c345bfba9c7fcf91f1e1d1a65c64f5c2fd069bb5c2de3fb030ce60e9d8ec
# 1: b6bf5bf97851604db9b3b0f89d64453dce5af606ad4fc7ece6ed11cfcecd7759
# 2: f99673f2f6d46de93315f6559e777c8f4cd9423990a22675b5f5e2cf9ebb6f99
# 3: 01fbe531cbdda3bc48f8f1cdfcc7681f42681109208b7faeea38c766b0e053e0
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
