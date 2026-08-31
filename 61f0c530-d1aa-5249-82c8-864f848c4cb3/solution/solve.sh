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
