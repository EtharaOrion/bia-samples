# Agent environment. Pinned by digest, never by a floating tag: Harbor accepts
# any string for docker_image, so a tag would validate cleanly and silently
# while the bytes underneath it moved.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C \
    LANG=C \
    TZ=UTC

WORKDIR /app

# The agent surface carries the protocol description and a starting submission.
# It does not carry the graded surface: tests/ and solution/ are excluded from
# the non-oracle agent surface by construction, not by a filter.
COPY session_client.py /app/session_client.py
COPY submission_template.py /app/submission.py

# Solver egress is OPEN here so the agent can fetch its own packages. The
# verifier environment denies egress. The asymmetry is bound in task.toml and
# recorded in solution/grounding.yaml under solver_egress_divergence.

CMD ["/bin/bash"]
