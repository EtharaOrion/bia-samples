# FORGE-CANARY-BEGIN
# 0: 44f8293b45f9807a774381f7bb9abfdc79292df957e8e8fa63ade87fcf5d063f
# 1: a1861d7c4311fee285f01a9884112f290f38bc322df60798491baf3bd2137c18
# 2: bb468f7d827aee352621e543c634e31e5b39b0a632a4ec428ecc028706f910b8
# 3: 6c609b3f8bf0fad83a3ea38f142d99a39cb8e416bc1290f85cfc03b788763296
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
