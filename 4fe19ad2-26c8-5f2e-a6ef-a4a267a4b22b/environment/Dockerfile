# Agent environment for the sustained-refinement slot.
#
# Pinned by @sha256:, never by a floating tag. Harbor accepts any string and
# does not itself require a digest, so the digest requirement is this contract's
# and it is enforced here and in tests/Dockerfile with the same digest.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The solving agent has OPEN egress in this sandbox, so it installs what it
# wants. Nothing below is a containment control; the deterministic checkers are.
ENV PYTHONDONTWRITEBYTECODE=1 \
    BIA_ROLE=agent \
    BIA_SPLIT=dev

WORKDIR /workspace
COPY . /workspace/environment/

# The graded split seed is absent from this image by construction. It is not
# filtered out after the fact; it was never copied in.
CMD ["/bin/bash"]
