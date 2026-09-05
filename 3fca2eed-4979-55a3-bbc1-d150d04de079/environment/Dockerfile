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
    OER_FINEWEB_DIR=/workspace/data/fineweb10B \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /app

# The GPT-2 BPE encoder the frozen trainer uses. The vocabulary is frozen with the
# model at 50304 entries, so this package is part of the frozen path rather than a
# solver dependency. It is carried unpinned exactly as the upstream record set carries
# it in harness/records/track_3_optimization; the image underneath is digest-pinned.
RUN pip install --no-cache-dir --break-system-packages tiktoken

COPY corpus/ /app/corpus/
COPY schema.json /app/schema.json
COPY split.json /app/split.json
COPY nanogpt_substrate.json /app/nanogpt_substrate.json
COPY frozen_recipe.json /app/frozen_recipe.json
COPY train_frozen.py /app/train_frozen.py

# The FineWeb10B train shards are staged by the harness at OER_FINEWEB_DIR through
# data/cached_fineweb10B.py. The VALIDATION shards are never staged here: the graded
# split is the verifier's and train_frozen.py refuses to start if one is reachable
# from this container.
RUN mkdir -p /workspace/data/fineweb10B

# The graded deliverables. /app/submission.py is the pipeline the agent writes and
# /logs/agent/trajectory.jsonl is the rollout record. Both are declared in task.toml
# artifacts so Harbor snapshots them before teardown.
RUN mkdir -p /logs/agent /logs/verifier /logs/harness/snapshots

CMD ["/bin/bash"]
