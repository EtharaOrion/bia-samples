# FORGE-CANARY-BEGIN
# 0: de3ff2c036a6f2c07f9e8b23a89f3f327c9784167753b0819dbba2f6d520f493
# 1: d23da0f8d13e4fc292219f39b771e87aae0c1701b8ec05a58aca3396f297bdc7
# 2: 99e192c25bfa9af69737c1be1984eef669edd04cf3c52a869b27ea37724683b4
# 3: 7a8f00e55da1d22aefeb3d58b70b205eaa705412e4deaf6e9823c146ad054966
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
