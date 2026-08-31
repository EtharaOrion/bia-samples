# The isolated verifier environment for slot OER-23.
#
# Same digest pin as the agent environment, different network posture. The verifier keeps the
# bound network policy: EGRESS DENIED. The solving agent's surface has open egress. That
# asymmetry is deliberate and bound in task.toml, and recorded in solution/grounding.yaml
# under solver_egress_divergence rather than left silent.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV OER23_SLOT=OER-23 \
    OER23_REWARD_ROOT=/logs/verifier \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    LC_ALL=C

WORKDIR /verifier

# The verifier carries its OWN copy of the serving harness and its OWN copy of the frozen
# substrate under fixtures/. It never grades on bytes reached through the agent surface; it
# hashes that surface and compares, which is a different act from trusting it.
COPY . /verifier/tests/

# No package is fetched here. The whole grading path is standard library plus PyYAML from the
# base image, so the verifier needs no network and is given none.
RUN chmod +x /verifier/tests/test.sh && mkdir -p /logs/verifier

ENTRYPOINT ["/verifier/tests/test.sh"]
