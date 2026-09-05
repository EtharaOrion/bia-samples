# FORGE-CANARY-BEGIN
# 0: f79799ff2e3983978fb1e0d49130f475940ee4dd9ea174fae7e473d18c52d7a8
# 1: 4708879094d239d794b2e1dc36d502c80d643f2885382bdd1db29c8a4335ab51
# 2: 0d8d782ca46430b4421c08d905ab28012a4b96062f5f46fe8682bc0f9f6e86bf
# 3: e2f0a9e07fa1ae35c89b77d74fc9f4ecd20da408de3ad5f203c978d2b4b7c1e3
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
