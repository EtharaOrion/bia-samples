# FORGE-CANARY-BEGIN
# 0: e1e234112db110a84cbb6c2050c6725a1352f06414215e642274464ef45924f7
# 1: 36e61857cb7b74316e0770708449b07d734e7be2f8b04cd2088a28405c104c2a
# 2: a59c86d6fb92f483961f34cd1b27377162df3eda9abbb52cdf757fbb74260ae4
# 3: 061e9684c9dde28cbf33857b560e237de54751eea381947922c978d03ffc4672
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
