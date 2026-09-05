# FORGE-CANARY-BEGIN
# 0: b94a6714c5930f636ac2c5101af036045d0dcf41e931c92d6d753ac8f33128cb
# 1: 13a95f4414141288e3418430b53f4b7e06310ebcf8fca7fc012f43066d825229
# 2: 4ae9e48d17ff0037120e314594c3b2548cc00025601bd252f5983808eb6071e5
# 3: 457f670d40487df18e34d24c73f9ec59d92ecbce3d81a00502d80fbb2fe9199e
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# source: solution/grounding.yaml
#
# The reference solution entry point. It writes ONE document: the vocabulary
# construction plan. It reports no number, because every number on the graded
# path is the harness's own, produced by training the frozen decoder itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SUBMISSION_PATH:-${HERE}/submission.json}"

python3 "${HERE}/reference.py" --emit "${OUT}"

# The allocation this reference reaches, for a reader who wants it without
# running anything: {"merge_depth": 18000, "span_units": 24000, "numeric_units": 5000, "punct_units": 0}
echo "wrote ${OUT}"
