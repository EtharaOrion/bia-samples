# The agent-facing environment for slot OER-23.
#
# Pinned by digest, never by a floating tag. Harbor accepts any string and does not itself
# require a digest, so the digest requirement is this contract's and this lane enforces it.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The solving agent has OPEN internet egress inside the sandbox so it can fetch its own
# packages, datasets and models. The verifier surface is a different image and keeps egress
# denied. The two surfaces differ by design; see solution/grounding.yaml solver_egress_divergence.
ENV OER23_SLOT=OER-23 \
    OER23_FAMILY=F11 \
    OER23_SUBMISSION_DIR=/workspace/submission \
    PYTHONDONTWRITEBYTECODE=1 \
    LC_ALL=C

WORKDIR /workspace

# The frozen substrate. The request trace, the hardware envelope and the p99 objective are
# frozen task bytes: they are readable, they are not modifiable, and a checker refuses a
# session whose surface digest has moved. serve_sim.py is the serving harness; it owns the
# tick clock and emits the timing telemetry the verifier grades on.
COPY trace.json hardware.json objective.json serve_sim.py /workspace/environment/

RUN mkdir -p /workspace/submission

CMD ["/bin/bash"]
