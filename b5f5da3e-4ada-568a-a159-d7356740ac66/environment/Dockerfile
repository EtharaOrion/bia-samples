# Agent environment. Pinned by digest, never by a floating tag.
#
# Harbor accepts any string in [environment].docker_image and does not itself
# require a digest, so a floating tag would validate cleanly and silently. The
# digest requirement is this contract's, and it is enforced here and in task.toml.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# The locked substrate. train.py is the working baseline the agent starts from and
# eval_protocol.md is the verifier-owned evaluation contract, stated so the agent
# optimizes the graded quantity rather than discovering it.
COPY train.py /app/train.py
COPY eval_protocol.md /app/eval_protocol.md
COPY anchors.json /app/anchors.json

# The agent writes here. Nothing under /app that arrived in this image is graded.
RUN mkdir -p /app/submission /logs/agent

# Solver egress is open at this surface. The verifier surface is a different image
# with egress denied; see tests/Dockerfile. The two surfaces differ by design.
CMD ["/bin/bash"]
