# Agent environment for slot OER-01.
#
# Pinned by digest, never by a floating tag. A tag resolves to different bytes on
# different days, so a run reproduced from a tag is a run against an image nobody
# recorded.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# The agent surface and nothing else. tests/ and solution/ are not copied, and
# they are absent from this file rather than filtered out of it, so no
# misconfiguration can let the held-out checkers or the reference through.
COPY train_baseline.py recipe_contract.md frozen_axes.json published_records.json fingerprint_tool.py graded_readout.json /app/environment/

RUN mkdir -p /workspace/submission /logs/agent

# Open egress at the agent surface is bound in task.toml under [agent]. The
# verifier image is a separate build with egress denied. Nothing in this file is
# the record-displacement control; that control is the deterministic fingerprint
# screen the verifier runs before any seed executes.
CMD ["/bin/bash"]
