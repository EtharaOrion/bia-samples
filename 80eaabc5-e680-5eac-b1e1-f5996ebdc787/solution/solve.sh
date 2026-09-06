# FORGE-CANARY-BEGIN
# 0: 3f4f3b96cd8fc82e85c7158992a762eab3a4848341d44037d06eb7279eb15a3a
# 1: b4e0d0bf9cc92e20ecd6ff3b983633407964b2377ba5b05320aff8b11b5e29f3
# 2: fc6585592ebdaa7e771426c03858409df0380e96414e1bee815a00dd7678c9c6
# 3: c99a1f3e72cbf54715fa602e3012f76e9b3263a8fe57987fb85a6c3f8714a5a9
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
