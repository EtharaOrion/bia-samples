# FORGE-CANARY-BEGIN
# 0: 38dec558596ff0b7708683e87b9a43ec3cb1e989eabbb7388ec4011ad17f01c5
# 1: a117f0dab25a6b818ca23c5e8bb6eeeacc794ba4e6f0c893078824e405285b19
# 2: f605b54d0e4dc86fe26c1b21285928f46df308e8e1c68f02c3540dc9c57f20d3
# 3: 948229fcb0122545bd81b90ceabf0b07e9e720f11d143ed734925c9114f483b0
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
