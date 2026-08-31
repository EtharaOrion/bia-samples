# FORGE-CANARY-BEGIN
# 0: bf3c0084eb28cf6354de75c320c68684c6903fd19a96ff2b8928596cfda2fa0c
# 1: 875ae60d93131c87b83c40898ea03e1eaddc1c5c4df9792013130030638f26b7
# 2: ed0b774c0de98d720d3610533f5f87c2f451c4d5fdeac93a21bfa649affe9318
# 3: 71179b7b417f679b6d8de9b23393ec5ff6a330f81bac9644f49b8f736c789a5d
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
