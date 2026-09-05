# FORGE-CANARY-BEGIN
# 0: a60898edead7b77d4d0ad2acc1d4ce153efbf4c4b7e28c5b196d6fed87f0d11a
# 1: 2a837c046b174ff6e0b910263fd6614b7c20aaf62ec9a4c5ac36d916284ec1d8
# 2: 2243082e56370ecef8eac8244229ed7220d2e07af6c0ddc6ba3a77084d398ed8
# 3: 368a7a181e37bd96f8860430e8cd06c24ce27aadd768ea6925c3a0efb9c034df
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
