# Verifier environment for slot OER-03. Isolated from the agent environment.
#
# Same pinned digest as environment/Dockerfile, and pinned for the same reason: a
# scanned image that differs from the executed image makes the measurement
# unverifiable. The two environments differ in exactly two respects, and both are
# deliberate.
#
#   1. EGRESS IS DENIED HERE. The agent surface carries open egress so the
#      solving model can fetch its own packages, datasets and models. The
#      measurement the reward rests on must not be reachable over a network, so
#      this surface has none. Bound in task.toml under [verifier] egress.
#
#   2. THE HELD-OUT SPLIT LIVES HERE AND ONLY HERE. The graded loss is computed
#      by this environment on a split the agent container never receives, from
#      the parameter snapshots the pinned harness wrote. That is what makes the
#      graded readout unshapeable from inside the run.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV BIA_SLOT=OER-03 \
    BIA_ROLE=verifier \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    http_proxy="" \
    https_proxy="" \
    no_proxy="*"

WORKDIR /verifier

COPY test.sh checkers.py checkers.yaml grade.py runner.py rubrics.jsonl test_output.py golden.json /verifier/

# No package install. The pinned image already carries PyYAML, and nothing on the
# graded path imports pytest: test.sh runs the compiled surface advisorily behind
# `||` and the reward comes from grade.py. Installing here would fail the build
# under PEP 668 on this externally-managed base, and forcing it through would
# replace the pinned image's own PyYAML, making the scanned image differ from the
# executed one.

# The reward directory Harbor mounts. Created here so an aborted verifier can
# still write its attributed zero rather than fail on a missing directory, which
# would turn a graded failure back into silence.
RUN mkdir -p /logs/verifier

RUN chmod +x /verifier/test.sh

ENTRYPOINT ["/verifier/test.sh"]
