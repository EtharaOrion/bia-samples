# FORGE-CANARY-BEGIN
# 0: 73830ddf39fa7ef10b16c89f414969f602b28e98aee2e2a8e7dc43c3861bd8df
# 1: 2938606608f8003af147547f50c3b3eb805cf033a8b4000cf788ead105ba7ebe
# 2: fed61ba929513434c7edd08f5edde183b07f80ed453a0083ef39bdb7ef626579
# 3: b64a0ec23d2a6dcafc6b5350d99c08e71e1f734f5313a0181d2c5c8c5b0ecce8
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
