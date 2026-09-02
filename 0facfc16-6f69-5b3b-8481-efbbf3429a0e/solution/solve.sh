# FORGE-CANARY-BEGIN
# 0: 18d8ef91e91f40893d517c0acbad10dea4b3e787b234eab02d679c467f3f5b78
# 1: d29d0642e5d53ceaf27cb69d5bbf382b3abbf628f9a87e83956b6b3ecca5aa9b
# 2: ed2c35f7692346c99e277854a2de10e5912ca054a1666c1fb042f3b2ebb0fd5d
# 3: 7bfff57a3377eddb1a9edd1f1dfeb9986542e605585f7035d751719af67e4bc6
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT. Source: solution/grounding.yaml, derived by solution/recompute.py.
#
# Harbor entry point for the reference solution of slot OER-21.
# It runs solution/reference.py, which writes the quantization scheme this slot's
# live checkers accept. It reports no metric and reads no clock.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$PWD}"
export OUT_DIR
mkdir -p "$OUT_DIR"
python3 "$HERE/reference.py"
# measured reference optimum, worst-shard perplexity degradation: 0.107170959
# measured optimum of the handed option set: 1.188591942
