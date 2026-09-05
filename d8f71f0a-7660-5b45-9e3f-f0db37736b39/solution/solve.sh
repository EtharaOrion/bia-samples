# FORGE-CANARY-BEGIN
# 0: e8e046d724ad713b9f3c5964879638bba57bf60ff64423bd5b8bc910ea83646b
# 1: 8ed9b27a1bb038bc7d7ee1ceaaf7e3e5bf196c9128058493af8803eb83f3d81f
# 2: 7abc7c0cb63fdbe298aec04b346775d87b6b9aee68babb14ae96ea5095827325
# 3: 65ce38b559c0159c6477849087dcf71a8fd743b2e488507ac461a0ebe4e9d200
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
printf '%s\n' "declared token budget 1677721600, frozen budget is enforced as fed"
