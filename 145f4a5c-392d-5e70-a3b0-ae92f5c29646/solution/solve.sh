# FORGE-CANARY-BEGIN
# 0: faa56985c40f754242a2c641b3817b5fa6b6ebe0c796d2d657321b3b5585b453
# 1: 5a264c6dcde54c2bb26d232713cae08dda417daecb6363420251f00ea8bd5f5f
# 2: 50c6ddd2b2b16def7271358cf2a66c99f2a7694b4627a995ca125a7d30a5363f
# 3: 8d6b323a9f67d5516105b1f88813b5052b4d2ee17c8ab81c3de75984b2a89943
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The solution entry point. It drives the bound refinement loop over the reference
# exploration schedule, keeps its own durable ledger under notes/, detects that the
# context summary dropped iterations, reconstructs the dropped findings from the
# ledger, and writes submission.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${HERE}/.." && pwd)"
WORKSPACE="${OER24_WORKSPACE:-${PWD}/workspace}"

mkdir -p "${WORKSPACE}"
python3 "${HERE}/reference.py" \
    --bundle "${BUNDLE}" \
    --workspace "${WORKSPACE}"

echo "submission written to ${WORKSPACE}/submission.json"
