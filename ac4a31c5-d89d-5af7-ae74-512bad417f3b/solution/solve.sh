# FORGE-CANARY-BEGIN
# 0: 87ddc555868cf0d195d5de10ef54c7242c03cebb3e0716f73b2b32f06251e0f2
# 1: 83e3e9b3485d32f836109ae98987eef3bbdbce918a0490946fc2ab388d7024b9
# 2: e9fd6b66f7121886ef99d227b414a5745568a60047a4bbbab70c67414a6af8c6
# 3: f8887f5f768aedd8230c36554c98e93d6620834b4b1c2c72567d10e4e0adcb37
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
