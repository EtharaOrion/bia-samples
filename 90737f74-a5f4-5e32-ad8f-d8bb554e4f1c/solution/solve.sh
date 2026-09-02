# FORGE-CANARY-BEGIN
# 0: 027e9f1e8123ed60ea97e933d668da67e6be8187f531b24bcc7c170df17d634f
# 1: 8ddd7eddbf0d902dd6553e645ac7384f86dd499707d31af7334d9f519906cf9c
# 2: a88971ef03409d0277a026d66950683005fc9da5d9b8b205a7e731f88beba44c
# 3: 6f5b4ea19a2244df60e8d8cae915054e369c499cc01e6e825e0b8e600fa43ffe
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
