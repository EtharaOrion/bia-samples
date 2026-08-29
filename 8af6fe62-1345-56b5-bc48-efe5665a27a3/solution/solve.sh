# FORGE-CANARY-BEGIN
# 0: 8627dc07a9b80b600c22b06fee71c0ca947e765218cd483305ca5d5f59d6d8e9
# 1: 84026f547cf6e775e34b8aef49a859a25b07b1bf031866399e454ae689a6ff29
# 2: 9c6bcc852ae3f15f74e65973983292103d4c0294a05a35a400a8eb937f0b1a83
# 3: 114177f2a3006d8626159eb283c220a6e74d12bb48d6a6131763987798a477b8
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
