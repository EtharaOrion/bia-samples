# FORGE-CANARY-BEGIN
# 0: a86f09d8ca1e428c3135d306cc04e705384953afd28230c68c98be7b3aed99a7
# 1: 32ae595312737d53b2fb7ba8f9ade993c736a4db2ea483726baa7ac0387ea16f
# 2: 8adefcf9eb016795f1d620ae96f7ec892f4c8c119b5d56459147b4183d51325c
# 3: f27363c5c65c8664d7f5cc25aa5923be18af0d8ce8ff4d7c6a342bde57b7ae8e
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
