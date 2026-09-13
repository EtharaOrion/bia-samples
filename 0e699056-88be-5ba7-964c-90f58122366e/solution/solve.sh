# FORGE-CANARY-BEGIN
# 0: c875ccfa72938362b8a08dc9f7f29dc3aa8b1697efedd8a962d7ff531cb746dc
# 1: 8fbe05bd14f05d2fed97171209ea672dc2073c56e66ba60214b4ec5df674d65b
# 2: 8ea88f027859c64d7fbf053c88107bddc8c17341a10f79583d61656fcd701ec8
# 3: da44f0f7d5567269e4d50eafbbabcf005c0622639bab8ab4ffb95d20f75f694a
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
