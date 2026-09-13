# FORGE-CANARY-BEGIN
# 0: 94c9b752f2d57dd61b5ce5f30281b1c0c2ee1fb11e7b5bc9abe973c912288eed
# 1: 4086b051eef5c63ccab7c0cfea5694b47fc485f9f2dda757b86571d7748df63e
# 2: 773f2282c1c3b794bf93232206e1662366f1188381bc41d1432832fb6fd726c2
# 3: cc3448ac099e73278af3cda92f0d6c95931b2976a7546dc78d471b6ecba4871c
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
