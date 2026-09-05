# FORGE-CANARY-BEGIN
# 0: f1eb90d94c70ef2b67f81aa635df62733a577918b05b5b988239866d6bc0b8b3
# 1: 907051b3bc7e6d1302c8a1c547f16aa43f99b74e072b7245231b81bf8725ea3d
# 2: 2a43d97984eab40033893fbe25c4ec10a21f559d127bae006dfeb31ad5b63788
# 3: 55ad4b5dd2603b381e8db09986cf4684f6435a05566000011aa3e78f355a57a1
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
