# FORGE-CANARY-BEGIN
# 0: fc45e58cd8ebe6e61ccefa43859319f940ad11d76892bc682142b2ac10baf77e
# 1: a2ab0f2243c8bf789cd7b193f819516ff79f238470f5b2e1d0daf0db9c6a4619
# 2: 05f7f3d2afdc6810f031e77403cb7bfb573b0542a03e9c1a1bf9867e9ec4db89
# 3: cbee89d2999edcf3415d3d41ab5d3b94ba5b54adf761f2bf14701be5f8b194d1
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
#
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The reference entry point. It installs the reference generator as the submission's
# generator.py in the working directory the verifier stages, and does nothing else. There is
# no network fetch, no model call and no clock read on this path.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-${FORGE_SUBMISSION_DIR:-/workspace/submission}}"

mkdir -p "${TARGET}"
cp "${HERE}/reference.py" "${TARGET}/generator.py"

printf '%s\n' "reference generator installed at ${TARGET}/generator.py"
