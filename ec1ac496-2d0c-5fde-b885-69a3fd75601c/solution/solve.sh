# FORGE-CANARY-BEGIN
# 0: 7a3be634bb112dd4b9fe66a061b305cb2d7f4dd5bbdf39d947f072648e8da986
# 1: f6e0bfe8ff81e4c5bcfc089be4217e76f33e244f0c302a4a010477e2d294df5c
# 2: b077f7caf0882a299308fd603f1a71b3495c6cd77fbc31c99a6e14ff13d7fbf7
# 3: 80cd1e19a92babab72f8bebb73d82c0fb3ba95212901927954462ac9204796a0
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
