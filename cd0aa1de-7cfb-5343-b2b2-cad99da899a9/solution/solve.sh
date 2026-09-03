# FORGE-CANARY-BEGIN
# 0: 5727ebe11fe06c91d60762d12d4037cd57a8ca7c1607c7dbccf346ca7cfce089
# 1: 88fed3d0420047db1dd71b47b86ef76cdc1f99a0ab62fe5b9518f1ac2ac9ee3a
# 2: 8ede583dcd1dde6130aa9b1e1ae4659bd90b725a37b2ac3039a7a210a538b464
# 3: 6f81fb485bb75ec94a59aef6ff26c453500f8a34a87e6c02d3d1f5e7c647e4da
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
