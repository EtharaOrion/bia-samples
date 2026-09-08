# The agent-facing task environment.
#
# Pinned by digest, never by a floating tag. Harbor accepts any string and does not itself
# require a digest, so the digest requirement is this contract's and this file enforces it.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The fence substrate is a deterministic build artifact over an integer recurrence. It needs no
# accelerator runtime and no third-party package: everything it uses is in the standard library.
# That is deliberate, because a reproducible grading path cannot rest on a wheel resolving the
# same way twice. Every dependency is therefore baked at build time and nothing is fetched at
# graded time. The base image is Debian 12 with Python 3.12 and enforces PEP 668, so a pip
# install here would need --break-system-packages; none is needed and none is performed.

WORKDIR /workspace

COPY fencelib.py /opt/fence/fencelib.py
COPY build_fence.py /opt/fence/build_fence.py
COPY fence_probe.py /opt/fence/fence_probe.py

# THE FENCE STATE IS ESTABLISHED HERE, AT BUILD TIME. The number of crossings the producer chain
# presents and the set of kinds the registry admits are both properties of these bytes. Neither
# is written down by the generator and neither appears anywhere in the agent-visible bundle.
RUN python3 /opt/fence/build_fence.py --root /opt/fence/state \
    && chmod -R a-w /opt/fence/state

ENV PYTHONPATH=/opt/fence
ENV PYTHONDONTWRITEBYTECODE=1
ENV OER26_FENCE_STATE=/opt/fence/state

RUN mkdir -p /workspace/notes /workspace/logs

CMD ["/bin/bash"]
