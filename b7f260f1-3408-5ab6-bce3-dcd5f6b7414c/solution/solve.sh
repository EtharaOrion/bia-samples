# FORGE-CANARY-BEGIN
# 0: 4d2ea721f2ee8fe381f781f22da07e5e44fb2d431cae996d0ff86f92045c3716
# 1: 708f4c04a77a87a85293e23c6cb924902f894b18692cfd6574eca8474179b03c
# 2: c55252ed46c06b2c028351f34bec74d90edf7420106609af8fb9358a2ccd38f5
# 3: f252ee5faf82ddc4a80cb560bed92f47ea1ea624d6f6031cdd5ed866ce70a1e5
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
