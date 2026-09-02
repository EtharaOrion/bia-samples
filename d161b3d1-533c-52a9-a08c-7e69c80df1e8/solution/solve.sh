# FORGE-CANARY-BEGIN
# 0: d18b9a21f1ad440176ee25a9fe589ca29e4c1690a80c7f4ae2288c1fc15547c6
# 1: 901bb295c39ca7a3bc08d93070845684d6a9e6227aa46d706a253dc6640af08c
# 2: 638c80fb8dab582d1502dfffc5f163985b0b92ddf171d4d54c1c11db195d678c
# 3: 102e58e0bd18a6fcfae0433b088078137e56888bb7c213fa12eeff66258e3dd0
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
