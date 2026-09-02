# FORGE-CANARY-BEGIN
# 0: f9504418b4b43f870bc670cb7b28a4fb8ddbfef76f70e0f45a1afa4fd3632f63
# 1: 5fc0d6caf3381b5d7266422024e2b7cb1d7f6e89e1c2f75a9170155f24d58c91
# 2: 4f1a5a4f0c2e7fccc2242563b710c1a873f5edcee40265acc8b305498c7d597f
# 3: 0f2ca675141a47875c31090d9d7100dde37bbb6c8a4c4590af83d0e6de41632e
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
