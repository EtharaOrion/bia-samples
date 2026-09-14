# Agent environment for slot OER-14.
#
# The image is pinned BY DIGEST and never by a floating tag: a tag is a mutable pointer, so
# a bundle pinned to one is not frozen and a rebuild months later grades a different stack.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The SOLVING agent has OPEN egress inside the sandbox so it can fetch its own packages,
# datasets and models. The VERIFIER does not: tests/Dockerfile builds a separate environment
# whose egress is denied. The asymmetry is deliberate and is bound in task.toml under
# [agent] solver_egress and [verifier] egress. The upstream deny-list is applied by the
# runtime, not by this file, and it is not a containment proof.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-14 \
    SUBMISSION_PATH=/workspace/submission.json

WORKDIR /workspace

# The frozen substrate and the frozen corpus. Nothing under solution/ or tests/ is copied
# here: the oracle and the held-out checkers are left behind by CONSTRUCTION rather than by
# a filter that could be misconfigured into letting them through.
COPY environment/substrate.py /workspace/environment/substrate.py
COPY environment/corpus /workspace/environment/corpus
COPY instruction.md /workspace/instruction.md
COPY task.toml /workspace/task.toml

# The lower layer is read-only. The agent writes its submission, and nothing else.
RUN chmod -R a-w /workspace/environment && chmod a-w /workspace/instruction.md /workspace/task.toml

CMD ["/bin/bash"]
