# Agent environment for OER-12. Pinned by digest, never by a floating tag.
# Harbor accepts any image string and does not itself require a digest, so the
# digest requirement is this contract's and this file is where the lane enforces it.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Solver egress is OPEN at this surface, by the bound asymmetry in task.toml.
# The verifier surface is a different image with egress denied. The two surfaces
# differ by design and the divergence is recorded in solution/grounding.yaml under
# solver_egress_divergence.
ENV OER_SLOT=OER-12 \
    OER_FAMILY=F13 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /app

COPY corpus/ /app/corpus/
COPY schema.json /app/schema.json
COPY split.json /app/split.json
COPY frozen_recipe.json /app/frozen_recipe.json
COPY train_frozen.py /app/train_frozen.py

# The graded deliverables. /app/submission.py is the pipeline the agent writes and
# /logs/agent/trajectory.jsonl is the rollout record. Both are declared in task.toml
# artifacts so Harbor snapshots them before teardown.
RUN mkdir -p /logs/agent /logs/verifier

CMD ["/bin/bash"]
