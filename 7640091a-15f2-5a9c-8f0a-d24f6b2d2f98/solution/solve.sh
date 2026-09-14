# FORGE-CANARY-BEGIN
# 0: d8e9ddb2c016c7ce9b0d0188815609fc3c7df1af1495e28a2dcbbc0ec7082998
# 1: 60c8e64802232bce9591e901f512467f497989298b7faf4c30a6790cf1a7cc9c
# 2: 2dbdc0fda3ac377b5ce7781f7afc323608c6d4e1f83ec225849aad6c211de5ac
# 3: 0deea6162de33936f65bd91c37144dd5c93ce4457065efeb533394c528736ac8
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
