# FORGE-CANARY-BEGIN
# 0: 0c4acb00aa65545ba44293b8012dc1ff5ff1c2d5d09c69a08fa67143cee37caa
# 1: 6a3599e73de95b8555019c8101eba5936fcbbdd3ad9070eadb9b93a1057fc79f
# 2: 18f3eb7d25648f58a6b9b7181ca67c0204d8d2f12c2005ad26a4bfd9d1376f49
# 3: a20c45f0e74a3859e1496fe369ae1e9d5205060740deef6f9fc95484bc194938
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
TARGET="${BIA_SUBMISSION:-/app/submission.py}"

mkdir -p "$(dirname "${TARGET}")"
cp "${HERE}/fixtures/reference_run/app/submission.py" "${TARGET}"
chmod 0755 "${TARGET}"

echo "reference producer installed at ${TARGET}"
