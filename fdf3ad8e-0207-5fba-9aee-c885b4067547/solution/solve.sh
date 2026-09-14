# FORGE-CANARY-BEGIN
# 0: 4c0d7c490b0100c86ffff0639cc35fb4ba753f45f8866fc183f1619364dbf644
# 1: 7486508c8570f51102e0961220c4d5462605aadcc44e0df47d65dfc8e58ed1e8
# 2: 67954e0743cc536365fa8c37652b7baba2d1b659b7e7966b16d6d4014b3cd853
# 3: 20da6346734e4642702fe895d98bd4d7635376fb1b81ad5aa9e615a07181b3a6
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
