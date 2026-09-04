# OER-09 agent environment.
# Pinned by digest, never by a floating tag. A tag without a digest is a lane failure.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The solving agent has open egress by bound design so it can fetch its own packages,
# datasets and models. The upstream deny-list is carried by name in task.toml and is NOT
# a containment proof: mirrors, caches, paraphrased write-ups and released artifacts all
# carry a recipe past any deny-list. The deterministic checkers are the load-bearing control.
ENV OER_SLOT=OER-09
ENV OER_FAMILY=F12
ENV OER_SOLVER_EGRESS=open
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /task

COPY frozen_config.yaml       /task/environment/frozen_config.yaml
COPY nanogpt_substrate.json   /task/environment/nanogpt_substrate.json
COPY curate.py                /task/environment/curate.py
COPY train.py                 /task/environment/train.py
COPY pool_probe.py            /task/environment/pool_probe.py
COPY README.md                /task/environment/README.md
COPY pool                     /task/environment/pool

# The FineWeb10B TRAIN shards are staged here at provisioning by the upstream loader named
# in environment/frozen_config.yaml corpus.loader. No shard matching corpus.split_shards is
# copied or fetched into this image: the held-out validation split is the verifier's, it is
# absent from every agent-visible container, and the graded loss is the verifier's own
# recomputation over its own copy.
RUN mkdir -p /workspace/data/fineweb10B

# The harness owns /logs/harness and writes its own observations there. The agent reads it.
# The agent never writes it, and nothing the agent writes there is ever read as graded state.
# /logs/harness/weights holds the parameter snapshots train.py writes at the points the
# verifier's schedule names; the harness digests and ledgers them and the submission
# selects none of them.
RUN mkdir -p /task/work /logs/harness /logs/harness/weights

CMD ["/bin/bash"]
