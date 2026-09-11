# FORGE-CANARY-BEGIN
# 0: ab1edfd70a312c6844d906125c58ea5fd60be0bac2d2c6423673b7ee830d7314
# 1: e6cae3dda6062574e12c4db3ca285519128e6de580aef0eefb7b512551116a5d
# 2: 3c6a3485bf4a72afc6a132b933e4a666a457c4e396183b205e89a0f73ee5dd9c
# 3: 62778ba74f8da7783b1a99be8e34b3f24b532ae9dbb96d8e1cc1221e45c465f6
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
