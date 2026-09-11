# FORGE-CANARY-BEGIN
# 0: 7decc35090e5fc11cba4b4a7196bf7ddc4aa4f7eaa5261bfa925c93ab0692f8d
# 1: 553c0210e241cc4388036f901e54c3d0394e4ff119bf0d0e7cb357f7672622b6
# 2: cd7fd5c2c45f528178576beecdee307d4615c303100359b2a48bf66d7158190d
# 3: aef4038ed4f1fc44454d8f922018e91f069cf27223cc3509a9f82fc25607b1e0
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It installs the reference producer at the bound
# submission path. The producer does the work at grading time: it reads the seam
# offset back through the harness handle, derives the admissible construction, and
# prints one JSON document on standard output.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${BIA_SUBMISSION:-/workspace/submission.py}"

mkdir -p "$(dirname "${TARGET}")"
cp "${HERE}/fixtures/reference_run/app/submission.py" "${TARGET}"
chmod 0755 "${TARGET}"

echo "reference producer installed at ${TARGET}"
