# Agent environment for OER-07.
#
# Pinned by digest, never by a floating tag. A tag can be re-pointed under a
# frozen bundle and would silently change what graded, which is exactly the
# silent-change failure this batch exists to make loud.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OER_SLOT=OER-07 \
    OER_LEDGER_DIR=/workspace/ledger

# The agent-visible lower layer. Harbor assembles task.toml, instruction.md and
# environment/ for a non-oracle agent and nothing else; solution/ and tests/ are
# absent from that surface by construction rather than by a filter.
WORKDIR /workspace
COPY environment/ /workspace/environment/

# The durable, agent-writable recovery area. It is a separate path from the
# agent-visible lower layer on purpose: it must be writable without any bit of
# the lower layer becoming writable.
RUN mkdir -p /workspace/ledger && chmod 0775 /workspace/ledger

# The solving agent has open egress here. The verifier image does not.
LABEL org.bia.slot="OER-07" \
      org.bia.family="F15" \
      org.bia.archetype="AR10-memory-degradation" \
      org.bia.solver_egress="open"

CMD ["/bin/bash"]
