# FORGE-CANARY-BEGIN
# 0: 0abd2ef6b266a6559f309af549ba55345cdbb5391d2d8d17781a977c315806a0
# 1: e8a7547d177b9615e8d74a9aaabd830b032b24c66f52336f594251f8eeacec4c
# 2: 13446bbf10ce87277461782f1d22ce0ce23b20ca702c3b5cba0fef4d69246989
# 3: 30e75efd444dad0d31aa90e9c239761d9e7002203ce1062332a3d86c6c6c13dd
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
