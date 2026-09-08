# Isolated verifier environment. Same digest pin as the agent environment, and
# the opposite network policy: the verifier denies egress while the solver's
# environment allows it. That asymmetry is deliberate and bound in task.toml.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C \
    LANG=C \
    TZ=UTC

WORKDIR /verifier

COPY . /verifier/tests

RUN mkdir -p /logs/verifier

# Egress denied. Grading reads only the telemetry this environment's own
# process produced, so nothing here has a reason to reach the network, and a
# grading path that could reach it is a grading path that could be fed.
ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
