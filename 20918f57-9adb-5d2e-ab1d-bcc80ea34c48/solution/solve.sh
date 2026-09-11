# FORGE-CANARY-BEGIN
# 0: c445476b754b90506b63cc5f3d718db751e98872e028f27689779f07b8716a72
# 1: 2fc383a0a475a6d06ddc12644777eb6a3f437d58bfd6a4bed263c763e41452f0
# 2: eee4142059c5cade4d190fb7b6add059fcfc3bb36895052cf7d7737e0c4c7442
# 3: 5c2fa8557df9320f4a237826cd740e9659af268eddbdc21fa536f20d0d28ae96
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
