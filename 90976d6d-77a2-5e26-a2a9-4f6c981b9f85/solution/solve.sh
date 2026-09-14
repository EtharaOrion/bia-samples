# FORGE-CANARY-BEGIN
# 0: 5997208f39e4b4737e951c8ce9e01c0ddecd7e402683f8a3ed3a1f70aa85d4a7
# 1: d52688a471fc1dd0753de7136569f02945f9d0a352ac3dd6e69d04c45c3b91e1
# 2: b60196718f4ad1d5fd07fd24d7c1fa055e2e39211a878039d8e41a1362ce45a2
# 3: d20f491e088dc1d4157c7c4a3de09c17c0c67ce180e7f7d6e05e87f8c978d97e
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# source: solution/grounding.yaml
#
# The reference solution entry point. It writes ONE document: the vocabulary
# construction plan. It trains nothing, measures nothing and reports no number,
# because every number on the graded path is the harness's own.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SUBMISSION_PATH:-${HERE}/submission.json}"

python3 "${HERE}/reference.py" --emit "${OUT}"

# The allocation this reference reaches, for a reader who wants it without
# running anything: {"merge_depth": 448, "span_units": 500, "numeric_units": 124, "punct_units": 0}
echo "wrote ${OUT}"
