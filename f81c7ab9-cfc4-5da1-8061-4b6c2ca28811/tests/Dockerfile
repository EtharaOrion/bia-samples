# --- FORGE Wave One repair -------------------------------------------------
# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
#
# This file previously carried a bare `COPY . <dest>`, which only assembles a
# working image when the build context is this tests/ directory.  Under that
# context the image holds the verifier and nothing else, so the grader's
# BUNDLE_DIR resolves to a directory with no environment/ beside it.  The first
# read of environment/<anything> raises, the EXIT trap writes reward 0.0, and
# the slot returns a constant zero for every possible submission - including
# the reference.  That is not a hard task; it is a dead one.
#
# Naming what is copied fixes it and keeps solution/ out of the verifier image.
# ---------------------------------------------------------------------------
# Isolated verifier environment. Same digest pin as the agent environment, and
# the opposite network policy: the verifier denies egress while the solver's
# environment allows it. That asymmetry is deliberate and bound in task.toml.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C \
    LANG=C \
    TZ=UTC

WORKDIR /verifier

COPY tests/ /verifier/tests/

COPY environment/ /verifier/environment/
RUN mkdir -p /logs/verifier /workspace

# Egress denied. Grading reads only the telemetry this environment's own
# process produced, so nothing here has a reason to reach the network, and a
# grading path that could reach it is a grading path that could be fed.
ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
