# FORGE-CANARY-BEGIN
# 0: 340424d04fd84d8e83f13c8148b596fd090e4c052737d5705a799402c76f2355
# 1: 5d2663f6f36e06cdac805edd6f5944845fb26bcbe610bff4faba0c3915c53da6
# 2: 7b51d334830ff0386af55fed73175192a7a37d95cfddf9f09feb602c112d1ccc
# 3: 19aabe98bcf6216d54ee7ee01f09ab6c85488231f945aacf3262f85dd75b0561
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
