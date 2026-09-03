# FORGE-CANARY-BEGIN
# 0: 4f45c238e81b08b1b3488c1857b498632020930dc571bdde9d8e5af7f16f865d
# 1: 7dca8b5b37182ff9958189d20e7326556aa62fae0f283cf5f3a870d3c0aa69aa
# 2: e16cac804b1acfba62290861ae71c008c29c0aa51563d15b9cffb9f0e2e29f77
# 3: 26a3e705eb92d0541dfec4dc29d210b80085b211987e8eead1dd38a04071cb53
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
