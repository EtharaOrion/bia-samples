# FORGE-CANARY-BEGIN
# 0: 4210cfc009fd2e5d7845de4fa14ae4a891f481a7296af938818fe08716a2e03c
# 1: a9a3ad75f2599a9bbdc3ed8df64abaa95c7757311efe772d92f9ee024dd67a1b
# 2: 7b3503b4ae61fbf9941e9ffccda136606639d586b409520db75924b7af4a36e1
# 3: 5da23602413a5a8899e8e8a84dba3d45d0e5f49ef81e68d0349429d39be1da05
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
