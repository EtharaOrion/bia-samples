# FORGE-CANARY-BEGIN
# 0: 390cae5918c7801d73cff1f01c647e682e6c8be55157f48201bb5ac58750a481
# 1: d3b317393d8d5d61ad708140538406a3fb1b56a8f24bb40681a70da4afab5a4b
# 2: 18df0a26c230e77f0e935d8d31819a7fe4e5a7b93311914be6274f9f87bde635
# 3: b496f50fe9b17ed7790fadb668199ac86bd19666d71d67654c1863320557c4d9
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
