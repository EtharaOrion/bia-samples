# FORGE-CANARY-BEGIN
# 0: df8866e6d7127d40b4e08b26cc0508afb5aa1363241210a000bd8887e1ed4e91
# 1: cf2cd407de0aef1882e9a080908d3de183c8096b7c642e378de2b5fded929ec6
# 2: bc3eaa9bbb754c4af8af258344124b1d672a9548bdef556a87072a601d8ffa90
# 3: b18809f6d677964c0507087f0684fc357b4f89fbfc3b7d410dc7e88f8deaf3f1
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
