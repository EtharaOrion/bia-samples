# FORGE-CANARY-BEGIN
# 0: fff698f683313fe4711c70fc78552a66178031e89096b5c1d90f3b4b3ff4e295
# 1: 4b4ed9e37a60d26d6459bff071042fe5bf24acf95dee98b632cd4fe5a2822291
# 2: 41b5e6028699c94c1e29c133bac5e93c91dc5fece0eec87e2835407db7b77ab4
# 3: 235adda6de8418b5904b2aaaba0c98842edf525a68b5c6c05a3ec55985789e75
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
