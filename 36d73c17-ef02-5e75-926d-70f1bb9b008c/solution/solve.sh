# FORGE-CANARY-BEGIN
# 0: 85a09aaef64002c3101f42bf2da1c334feb2d6eb97d9416fc994f344af94b525
# 1: 81c19191fb27df22e0c544251ea17c82183695a45558b971152326e6ae482e48
# 2: 1689a1a4f0a72a46777269f6aba3dda11200b7bddb87118ab91ac29db64a9196
# 3: 49e494b4f8410e2c8648796876ce38c5aa1171c379c666ac0f85bde167d87eff
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
# d17b2dc483dbe74cd5050a66aa9c0c2721e5e33d86999b61cecdc53297fe2300
echo "reference installed at ${SUBMISSION}, claimed_step 2690"
