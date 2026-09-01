# OER-18 agent environment.
#
# Pinned by digest, never by a floating tag. Harbor accepts any string and does not itself
# require a digest; the digest requirement is this contract's, and this lane enforces it.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The solving agent has open internet egress inside the sandbox so it can fetch its own
# packages, datasets and models. The verifier environment does not; see tests/Dockerfile.
# The deny-list bound in task.toml is not a containment proof and is not treated as one.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C \
    LANG=C

WORKDIR /workspace

# The task substrate. train_stack.py is the frozen model, optimizer and feeder; the verifier
# runs its own copy of the same bytes, whose sha256 is bound in tests/checkers.yaml.
COPY train_stack.py /workspace/train_stack.py
COPY dev_benchmark.json /workspace/dev_benchmark.json

# The submission directory the verifier stages from. The agent writes generator.py here.
RUN mkdir -p /workspace/submission

CMD ["/bin/bash"]
