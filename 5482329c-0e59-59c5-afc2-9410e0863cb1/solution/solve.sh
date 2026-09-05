# FORGE-CANARY-BEGIN
# 0: 8de9d2b3e12d74ace5db8ea00702ea15dfdf92e5f3631ac8d62073a3eafdd9fa
# 1: 6d6c2b461caa4ab144dcc41b5ab4ff6ceb20e04474ea08185d7dc99e8b5fdd6b
# 2: ae2ce224a8237f0b602ee79ce00644971c2a50d8df7b0c11f73f3ba534c0d03e
# 3: 145366ce9f7db58057d6f24cb311b568df45c5bc78fb606d54b48891f8647af3
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
