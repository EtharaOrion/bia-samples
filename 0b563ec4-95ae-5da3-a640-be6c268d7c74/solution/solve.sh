# FORGE-CANARY-BEGIN
# 0: 3da3280ef14eccf90a3d165b88987685da3eeb37449e59d1d74e0e8e9543afba
# 1: 681686a988f7951519ef9bda0b1475e24aabb16c3e4064b15b42242fd8f34fe7
# 2: 693760080769bb940c9fa60fa0ec8dae0e8fc40ea16cadef630a5fd78376477b
# 3: 11b0e590356334e5fe8fc17f7f0a443f595dcbb9b94a78dc62900c23eb203563
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml
#
# Oracle entry point for slot OER-10. It installs the reference curation recipe as the
# submission and stops. It never writes a reward, never reads a checker, and never
# reports a loss: the graded validation loss is the verifier's own number.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMISSION_DIR="${OER10_SUBMISSION_DIR:-/workspace/submission}"

mkdir -p "${SUBMISSION_DIR}"
cp "${BUNDLE_DIR}/solution/reference.py" "${SUBMISSION_DIR}/recipe.py"

printf '%s\n' "installed reference recipe ref-derived-interaction at ${SUBMISSION_DIR}/recipe.py"
printf '%s\n' "declared token budget 419430400, frozen budget is enforced as fed"
