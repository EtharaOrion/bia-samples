# FORGE-CANARY-BEGIN
# 0: ef5decf1e8c11ff08709ee70ff8553da093d9e3f3dc0bdca9bc251f95f1c0aa6
# 1: c920654977e8b318451b60417863085a59f9a1e23e86dc91158983dca6cf6845
# 2: 52439753565457dcb7ed54dbcdf55bf44654a28ab80730317a63bb1456aa887c
# 3: ed0d99202bdc948c5ed369e5c6fbccdda48ce0efdabe326580b0b4301b15f7cd
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
