# FORGE-CANARY-BEGIN
# 0: 07db72ec5cfa00b4b3f0d21869cc25f43c526a480629183f4be329953accb4c3
# 1: 6d236d4dc09ce64e5e7a58ffea26c89dd0cafc37bf19a88ddbf1fd4661a4302c
# 2: 1de0fc6ec8dcd83f2b6c9c7bbdd2c5e69ba68ddd07b9b5d614c42a9baf9a18c8
# 3: 2163e14ebfe2c41557fa4dd4f4802b81ab044ff439ae6f024b2df5b861c4a916
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
