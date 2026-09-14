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

COPY frozen_config.yaml /task/environment/frozen_config.yaml
COPY curate.py          /task/environment/curate.py
COPY train.py           /task/environment/train.py
COPY pool_probe.py      /task/environment/pool_probe.py
COPY README.md          /task/environment/README.md
COPY pool               /task/environment/pool

# The harness owns /logs/harness and writes its own observations there. The agent reads it.
# The agent never writes it, and nothing the agent writes there is ever read as graded state.
RUN mkdir -p /task/work /logs/harness

CMD ["/bin/bash"]
