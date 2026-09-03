# The isolated verifier environment. Same pinned digest as the agent environment so
# the two surfaces differ only where they are declared to differ.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /verifier

COPY . /verifier/tests

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

# Verifier egress is denied, bound in task.toml under [verifier]. Nothing on the
# graded path reads the network, a clock or a random source, so the grading result
# is a function of the submission's corpus and of the frozen bytes alone.
ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
