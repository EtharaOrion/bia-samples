# FORGE-CANARY-BEGIN
# 0: 498f67430643a39275a52a6c9d070d0eef661d076493d306062bbc93d079d871
# 1: 5234730233da6167498b7690ff0eadeb176d939561893aa75a7e0592d5440b86
# 2: ef59d1deddda8d0631a81e5960d8693e026ac1e51d41125c5633aa9a84147902
# 3: 65fba47c90f099c8e425f97606fedaf6c42e87862726df28de2e19b8161df3d6
# FORGE-CANARY-END
#!/usr/bin/env bash
# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml. Regenerate with: python3 solution/recompute.py
#
# Oracle entry point for slot OER-03.
#
# Installs the reference solution as the submission. It does NOT train here, and
# that is the repair rather than an omission: the verifier RE-RUNS the submission
# out of process (tests/runner.py) and grades the snapshots ITS OWN run produced,
# so a training run in this container is graded by nothing. It used to run the
# whole 3000-step schedule here, which is where most of this slot's wall clock
# went, and then the verifier trained it all over again.
#
# The submission lands on /workspace, which is the ONLY surface this container
# and the verifier container share. It used to be written to /app/submission.py,
# which is inside this image; the verifier looked for that path in ITS image,
# found nothing, and reported `submission-absent` on every run. That is why this
# slot scored a constant zero for the reference and for a wrecked submission
# alike.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUBMISSION="${BIA_SUBMISSION:-/workspace/submission.py}"
mkdir -p "$(dirname "${SUBMISSION}")"
cp "$HERE/reference.py" "${SUBMISSION}"
chmod 0644 "${SUBMISSION}"
echo "reference submission installed at ${SUBMISSION}"

# The reference never decides when it crossed and never reports a crossing.
# The verifier recomputes the graded step from its own evaluations, and it
# measures both ends of the reward scale the same way.
