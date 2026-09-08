# FORGE-CANARY-BEGIN
# 0: 04a07d4a4dbe1610a0d0f4f76dd5d8855ce7c9959f5f9da62f7ca3a37badf04e
# 1: ab010c555645a169bbd6c8250824b031ec949421f2af52a3c2c151f4f20ca751
# 2: cded92d02e0fe9028765be5b930d5d9ffacb829830009d660e58080989640e3a
# 3: bc67f7d3f21bbf9e191d034ad3674e502a31dfe61f5b642a2bc7ceb35e1ddfe9
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
