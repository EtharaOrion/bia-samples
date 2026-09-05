# FORGE-CANARY-BEGIN
# 0: 9e9f3feecaa01e03133a9fc1e07b37c299f9510dd773df8360fd8b13479603a1
# 1: c1b470631a0ffbc8a4d7f9bcba6f86c0f4bd2470025201381801ea2f580011fc
# 2: 80220e2223da0839166853d3d37d590cdf7c620adecead02511820bf704186ea
# 3: 8aa48988070c3459f7d6aa533bcd2ad5087bc89637080029f4817dc1d1e6f2d6
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
