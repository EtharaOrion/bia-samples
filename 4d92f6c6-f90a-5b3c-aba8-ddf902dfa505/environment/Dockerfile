# Agent environment for OER-16.
#
# Pinned by @sha256: and never by a floating tag. Harbor accepts any string for
# an image reference and does not itself require a digest, so a tag would
# validate cleanly while silently resolving to different bytes on a later pull.
# The digest requirement is this contract's, and this line is where it is kept.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The solving agent has OPEN egress inside the sandbox and may fetch its own
# packages, datasets and models. Nothing is pre-fetched here on the agent's
# behalf, so what the agent chose to install stays visible in its trajectory.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER16_SLOT=OER-16

WORKDIR /app

# The frozen half of the task. Copied read-only; the verifier fingerprints these
# at the start of the run and again at the end.
COPY frozen/ /app/frozen/

# The free half: the interface the submission implements and a complete working
# baseline that is valid today.
COPY tokenizer_interface.py /app/tokenizer_interface.py
COPY baseline_tokenizer.py /app/baseline_tokenizer.py
COPY submission_protocol.md /app/submission_protocol.md

# The graded deliverable. The agent writes it; nothing here writes it for them.
# /app/submission.py is snapshotted as an artifact by the delivery manifest.

CMD ["/bin/bash"]
