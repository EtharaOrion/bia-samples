# Isolated verifier environment for slot OER-21.
# Same digest pin as the agent environment, and egress is denied at this surface.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /verifier

COPY . /verifier/tests

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/verifier/tests

# The bound reward carrier. /logs/verifier/reward.txt takes the bare float and
# /logs/verifier/score.json takes the reason and the metric block.
VOLUME ["/logs/verifier"]

ENTRYPOINT ["bash", "/verifier/tests/test.sh"]
