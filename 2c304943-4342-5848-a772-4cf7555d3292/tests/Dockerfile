# Isolated verifier environment for slot OER-01.
#
# Separate from the agent image on purpose. The reward rests on runs the verifier
# re-executes itself, so this environment must be accelerator-capable and must not
# be reachable from the agent's. Egress is denied here; see [verifier] in
# task.toml. The digest is the same pin the agent image carries, so the two
# environments differ in what they mount and what they may reach, never in which
# bytes they were built from.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

WORKDIR /verifier

COPY . /verifier/tests/

RUN mkdir -p /logs/verifier /workspace/submission

# The entry point owns the whole reward contract, because Harbor ships no result
# parser. Every exit path writes both bound reward artifacts from one EXIT trap.
ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
