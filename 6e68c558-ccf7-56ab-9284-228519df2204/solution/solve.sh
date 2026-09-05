# FORGE-CANARY-BEGIN
# 0: 00e3cd80dddd52b471f75b8869a24c0e70ad53564662b0033cbe43999c9e108c
# 1: 18806e45f7c55dc2414bbf68daf3c3277ef255fa04d228f0c6a580f267e1c555
# 2: 29ba609d08ebcde16cde53ecf8072c30587f4d3c87b927fe8672b26d97e15e0c
# 3: d66de14a9f2ff899423eb4f520ce4c11335417f9bf167cc08b1d8b2070a11dbf
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with solution/recompute.py.
#
# The reference entry point. It installs the reference solution as the
# submission and records the claim the reference reaches. It trains nothing
# itself; tests/runner.py re-executes what is installed here.
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION="${OER04_SUBMISSION:-/app/submission.py}"
CLAIM="${OER04_CLAIM:-/app/claim.json}"

mkdir -p "$(dirname "${SUBMISSION}")" "$(dirname "${CLAIM}")"
cp "${SOLUTION_DIR}/reference.py" "${SUBMISSION}"

cat > "${CLAIM}" <<'CLAIM_JSON'
{"claimed_step": 2690}
CLAIM_JSON

# sha256 of solution/reference.py at generation time, bound into the accepting
# fixture so the accepting half proves the checkers accept THIS reference.
# aeb24bba3f74b13036d2d12a98a25dd1a8fd68950e04494ab26fefef7cc0c4e6
echo "reference installed at ${SUBMISSION}, claimed_step 2690"
