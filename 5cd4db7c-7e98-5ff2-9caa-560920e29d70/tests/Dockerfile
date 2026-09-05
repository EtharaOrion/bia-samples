# The isolated verifier environment.
#
# Same digest pin as the task environment, and pinned by digest for the same reason: a
# floating tag would let the grading interpreter move underneath a frozen bundle.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# EGRESS IS DENIED on this surface. The solving agent's surface has open egress; these two
# surfaces differ by design and the asymmetry is bound in task.toml. A verifier that can
# reach the network is a verifier whose verdict can be fetched.
ENV OER24_VERIFIER_EGRESS=denied
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /task

COPY . /task/tests/

# The verifier re-runs the frozen trace under the submitted configuration on the
# harness-owned simulator and computes both graded readouts from its own telemetry. It reads
# no clock; the simulator's clock is virtual and arithmetic.
ENTRYPOINT ["/bin/bash", "/task/tests/test.sh"]
