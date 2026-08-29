# Isolated verifier environment for OER-07.
#
# Same digest as the agent environment, and a different posture: egress is
# denied here. The solving agent had open egress for packages, datasets and
# models; the grader has none. The asymmetry is deliberate and bound in
# task.toml under [agent] and [verifier].
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    OER_SLOT=OER-07 \
    OER_TELEMETRY=/logs/verifier

# No package install. This base image is externally managed under PEP 668, so a
# bare pip install exits non-zero and no verifier ever runs. Nothing on the graded
# path imports pytest: test.sh runs tests/test_output.py behind `|| true` and the
# reward is written by tests/grade.py.

WORKDIR /verifier
COPY tests/ /verifier/tests/

# The reward carrier the live seed/forge/verifier.py binds, and its companion
# reason document. tests/test.sh writes both from one EXIT trap.
RUN mkdir -p /logs/verifier

# Recorded for the operator, and enforced by the runtime rather than by this
# line. A LABEL is a declaration and never a control.
LABEL org.bia.slot="OER-07" \
      org.bia.verifier_egress="denied" \
      org.bia.reward_carrier="/logs/verifier/reward.txt"

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
