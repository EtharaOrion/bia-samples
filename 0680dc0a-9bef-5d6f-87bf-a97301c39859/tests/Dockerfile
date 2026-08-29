# Isolated verifier environment for OER-16.
#
# Same pinned digest as the agent environment, and pinned for the same reason: a
# floating tag validates cleanly and silently resolves to different bytes.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The verifier's network policy is DENIED. This differs from the agent surface
# by design: the agent has open egress so it can fetch packages, datasets and
# models; the verifier fetches nothing, ever. The asymmetry is bound in
# task.toml and the divergence is recorded in solution/grounding.yaml.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER16_VERIFIER_EGRESS=denied \
    OER16_SLOT=OER-16

WORKDIR /verifier

# The grading tree. It is never mounted on the agent surface: Harbor assembles a
# non-oracle agent surface from task.toml, instruction.md and environment/ only,
# so tests/ and solution/ are absent from it by construction rather than removed
# by a filter that could be misconfigured.
COPY . /verifier/tests/

# The reward carrier. /logs/verifier/reward.txt takes the bare float and
# /logs/verifier/score.json takes the reason and metric block.
RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
