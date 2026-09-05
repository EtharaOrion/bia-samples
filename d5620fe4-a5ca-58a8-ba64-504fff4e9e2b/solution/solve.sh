# FORGE-CANARY-BEGIN
# 0: e9fd01e96898ea7c19f162805d0a185450c96ecd8ac0de30d33305b425c558d5
# 1: 6acfb9b3a37f97c63886bf1d7b6a42ef82d9f01cf62e1e46dc058a1ceb1d0b4f
# 2: cd1ee2be72fe9aed11410a70f2ed474482266d3c256fa752b1482c7574ebf96f
# 3: 56492788a36fcae3c606a7800f77dd2906feba8856dd283816e0ff7ad84f9bc6
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
