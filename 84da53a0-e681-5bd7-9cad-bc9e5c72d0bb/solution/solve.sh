# FORGE-CANARY-BEGIN
# 0: ddf61f456bd86267b016deb1b4d9f72ec05614f3ea5de1bb831ef2b62f89a113
# 1: 585c9d65fd496c932e86e08f7c6b9fedcca4ddc9bf86815017777f47d1eca49f
# 2: d9ee6f812d45833738ee0a4759d8286adbddd9dd6e0e8b84bcccda778c0e5b31
# 3: 9dacde9953d15f930796b53625c34116d7327233b0b8ccb582d06a6ea5ea61d7
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
