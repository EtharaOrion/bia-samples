# FORGE-CANARY-BEGIN
# 0: 15649b4386c293019c1263b64215ad44d4399a5feff56c0ad032c4dd3f89ea96
# 1: dbac865131459d905e79ed134afa269c2ffbc94f152dc48977e7b62b2748ac42
# 2: 359500ba7f7a1c02ad2b52738dfb13e97f35f48b48fefaca12442ceb4bf4b2a5
# 3: 14a334742eb644d9feab5fa26f3013dc0888de49fbccbef73787a2aec60e816c
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
