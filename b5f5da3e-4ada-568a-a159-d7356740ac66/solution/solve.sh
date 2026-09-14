# FORGE-CANARY-BEGIN
# 0: e91621c65b7bc043afaa935b2b2f031e1e8385a3b2baa29a5a83af1274fcab55
# 1: 25e949919e1b1d0c90fbc1c2c12471f12c4f0292b1d7b17141edbe4aa430eac0
# 2: f5ff468803f90ae95c452e8c7ca2db6de62ecd16739fd330484bb6ae1920ddfa
# 3: 262267978c5c96773f717af316ec4c7362e32fce31b365387184f30b04bd22d2
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
