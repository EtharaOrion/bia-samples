# FORGE-CANARY-BEGIN
# 0: 8464bde8df3813a67cfafbb44549bd56f5c87c2a789ca59a9eb4659a5e003f7d
# 1: 377208ea55a5c1aca0bd8cabd7cf8848af3ac8fe8ba0d6999a05ba1592ff83f5
# 2: 2812d9cd09ae1849f5d2b360b1fb94decbc7bbd9484ee9ab1ef8e6837fe8f320
# 3: e619ded3d70c26390c2e42202d44ecb30376d47bba2e7d6db8d0cdbad719d188
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
