# The agent environment. Pinned by digest, never by a floating tag: a tag can be
# repointed after this bundle is frozen, and a bundle whose base image can move is
# a bundle whose results cannot be reproduced.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /task

# The frozen half of the task and the tools the agent measures with. Copied
# read-only; the agent-visible lower layer grants no write bit anywhere.
COPY frozen /task/environment/frozen
COPY tools /task/environment/tools

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/task/environment/tools

# The solving agent has open egress inside the sandbox. That is bound in task.toml
# under [agent] and is deliberately asymmetric with the verifier, whose egress is
# denied. The deny-list carried alongside it is not a containment proof.
CMD ["/bin/bash"]
