# FORGE-CANARY-BEGIN
# 0: f556dff8e2c45fb3e2a11ef100d30980a097f64ecb761c7b8c83765ac9e16ef9
# 1: f80abe7f7f8ed896298d9832609f4908ca51096a127c40f854abca8b05249180
# 2: 5600be3c85383eb0d108fcac3f5ffb5196adf176b9e16a87beb1647cfa64f2c3
# 3: 99380964149e752a5a454e1a418d8b3033dd4b506f4f9ca056143fcaf7090bc5
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
