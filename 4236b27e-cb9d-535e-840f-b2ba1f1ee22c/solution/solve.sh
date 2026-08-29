# FORGE-CANARY-BEGIN
# 0: 3f7b86dc0ce8072f217d97bfebb463bbe8812e22886f18ec592f471ed4af32a1
# 1: 5042d0d47015a72dc89b42f188f2d40f870f148fdaf2efb2e5c73ff30a3cf2ad
# 2: 21fc2ff76a164c7c28ea00dba5a996d7915f90d83c7dbd6812da8f1043e91f85
# 3: 06592622644552cc73fff70d08de87b7472feba8e48d907abe0619549aa40575
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
