# FORGE-CANARY-BEGIN
# 0: de114644ee74ed1fe41f251978cea581d8ec1c0630d5abdb91a7652c264485bd
# 1: c9a2369ea54117160810212074b935a549d53a3d4209ad42199294a1a7f81387
# 2: cc436c48f483ab08374a5a8e8954657e821fbc24c80d61f86ecd1a535523bc6c
# 3: 585cdb0b409f3280527532bf57c63276c5f52ce2e37ba5e97015221e602cb68a
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
