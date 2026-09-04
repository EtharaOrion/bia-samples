# FORGE-CANARY-BEGIN
# 0: 9627081aed6d4021fdfa5d729ed6146662536def486405d23516c8dcd9cb997f
# 1: e26764a99aaff92e76f980e07f36bf21caa1fb10e80358fac1db212ff2a3eb79
# 2: a5ae65732e047fc27add73a583001817284600496446fe075146d3c250b087ca
# 3: 7344b748cde033ae62cbf0d280312972bd4dc03cb8dc6bd29dbfeef5a5a147de
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
