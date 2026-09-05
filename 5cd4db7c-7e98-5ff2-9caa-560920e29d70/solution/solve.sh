# FORGE-CANARY-BEGIN
# 0: 3d764b3b88892040f86c53c38b873dd55c0bbcb66a5f5732848326034152ee08
# 1: 11ed6967bdb9cbddbec477d36dcde299dbf3a7f4239ce89bf1cdb3575ba9a918
# 2: 0c0700340bc9e16c178038fcfd66d40d136d81c5d64b03f4266436ec12c58c42
# 3: fd4431952e5492a32c9b2ea8c826f069c4a0698fceb88dc0c923f932acc3aae2
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
