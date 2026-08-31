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

# pytest carries tests/test_output.py, which is generated from
# solution/grounding.yaml by solution/recompute.py and never hand-edited.
RUN python3 -m pip install --no-cache-dir --disable-pip-version-check --break-system-packages "pytest==8.3.3"

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
