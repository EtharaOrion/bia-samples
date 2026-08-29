# FORGE-CANARY-BEGIN
# 0: ba066f71bff1d004a051f1db31f8ae1679f5632b2445ef9de2d827d555113101
# 1: 3711d1026a1fbe7d33583db71ba20e39cda4b3ce34e153c02474a921e31f9160
# 2: 50f6989ffbebd3e0110c7eedc81d6496629257eace787627e694920f11c3b6e1
# 3: a0d28d420db039aa3e8191654e066240a68e278526e77c33a9cc8f004f22376f
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
