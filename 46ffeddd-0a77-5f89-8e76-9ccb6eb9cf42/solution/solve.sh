# FORGE-CANARY-BEGIN
# 0: f99da548ccbfab0f8c4e562e2fe82bd0d33209542d36d648723afb0cedfcf7a9
# 1: 2ecb4f06916f3e1ff81a332d6ce7c0993f49ea5980d2a15f4dc306559830f09e
# 2: e227d61f7050c8f1b68b45b3237e2086fb46a361043f8f670a7d76b557301202
# 3: a028fb01131fc4fa3ed04640c6a304a8007c4fd8c8c07fd5d931b0562c5fe0ee
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
