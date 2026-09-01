# OER-18 verifier environment. Isolated from the agent environment on purpose.
#
# Same digest pin as environment/Dockerfile, because the graded stack must be the stack the
# agent saw. Pinned by digest, never by a floating tag.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Verifier egress is DENIED. This is the asymmetry task.toml binds: the solving agent has
# open egress inside its sandbox, and the verifier has none. Nothing on the grading path
# resolves a name, opens a socket, or fetches a byte. Every graded number is computed here
# from the corpus the submission emitted and the frozen bytes in this tree.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    LC_ALL=C \
    LANG=C \
    FORGE_VERIFIER_EGRESS=denied

WORKDIR /verifier

# The grading tree. The held-out benchmark lives here and only here; it is never assembled
# onto the agent surface, which carries task.toml, instruction.md and environment/ alone.
COPY . /verifier/tests

# The reward carriers Harbor mounts.
RUN mkdir -p /logs/verifier

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
