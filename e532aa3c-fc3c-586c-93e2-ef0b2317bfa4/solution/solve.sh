# FORGE-CANARY-BEGIN
# 0: 9bc4efbd8ec51180978496617ab1ab0de7e897cc3c18ae28e5c116f292237272
# 1: f44ffd1e7e16c8d6675f1c2f3e35b44fcaa415ff734c33c526fdab941ceca939
# 2: dafefd998667e6150cf69d911d475285a1383f772ebae61955e5aed13c2b983d
# 3: 373a16ad6962ff7715cb2aeb39e23aeda214455bae1ec89234f90000a9b9a717
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
