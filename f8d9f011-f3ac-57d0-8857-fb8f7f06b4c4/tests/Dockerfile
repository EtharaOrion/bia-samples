# The isolated verifier environment. Same digest pin as the agent surface, so the
# grading process and the substrate cannot drift apart across a re-pull.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OER22_VERIFIER_EGRESS=denied

WORKDIR /opt/task

COPY task.toml /opt/task/task.toml
COPY instruction.md /opt/task/instruction.md
COPY environment/ /opt/task/environment/
COPY tests/ /opt/task/tests/
COPY solution/ /opt/task/solution/

# Verifier egress is DENIED. That is bound in task.toml under [verifier] and is not
# established by this file: a Dockerfile states an intent and a runtime enforces it.
# Nothing in the grading path opens a socket, so a denied egress removes a capability
# the verifier never exercised rather than closing a hole it depended on.

ENTRYPOINT ["/usr/bin/env", "bash", "/opt/task/tests/test.sh"]
