# FORGE-CANARY-BEGIN
# 0: a8c2febcb5810c9305dd94f4a2b37b9758cfbfc476a1ad91cadc83a90c3e30f4
# 1: 1ce562724221453c103434a8baa387c776ff222b72d2d5fd245df848bcd42038
# 2: fa8fe6324ffca83ee5cb9ac63cdbd816c61b1de32c4e0aa26ecc9b884d79dbe1
# 3: d8278e8e7c85206d938305c57cd0d680e514091f05cb1cdc8f6822145c9c57b3
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
