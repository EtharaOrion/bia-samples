# FORGE-CANARY-BEGIN
# 0: 31d2bdf64a3ca80caa7dc162a37b02a03a2e1f4f338835ac89f481f8da151c46
# 1: c55b0ae20886af115aaf65eaac560701259d999b5a60932b3487d7c4402032c7
# 2: c5af7340594983f66ef4249257f7327162b47f50ac5d474c02d47afde4f0dc09
# 3: 5607d004415140b7c8050d5b4c0ace6ec475a65c268a857124f6b992571054b1
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with: python3 solution/recompute.py
#
# The solution entry point. Installs the reference optimizer as the
# submission and drives the bound refinement loop under the recovery
# discipline: reconcile the durable ledger against the handed summary
# BEFORE proposing, at every iteration, and record the reconciliation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(dirname "${HERE}")"
WORKSPACE="${OER_WORKSPACE:-/workspace}"
LEDGER_DIR="${OER_LEDGER_DIR:-/workspace/ledger}"

mkdir -p "${LEDGER_DIR}"
cp "${HERE}/reference.py" "${WORKSPACE}/submission.py"

# The compaction lands at iteration 4 and the summary window is 2.
# Nothing announces it. Detection is the difference between the durable
# ledger and the summary handed in, which is what reference.recover()
# computes; it appends a reconstruction record naming every dropped id.
for ITERATION in 1 2 3 4 5; do
  OER_ITERATION="${ITERATION}" OER_LEDGER_DIR="${LEDGER_DIR}" \
    python3 "${HERE}/reconcile.py"
done

echo "OER-07 reference installed at ${WORKSPACE}/submission.py"
