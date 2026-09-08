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
# The isolated verifier environment.
#
# Same digest pin as the task environment, and pinned by digest for the same reason: a
# floating tag would let the grading interpreter move underneath a frozen bundle.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# EGRESS IS DENIED at graded time. The solving agent's surface has open egress; these two
# surfaces differ by design and the asymmetry is bound in task.toml. A verifier that can
# reach the network is a verifier whose verdict can be fetched.
ENV OER27_VERIFIER_EGRESS=denied
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

# The GRADED path imports nothing outside the standard library. The only dependency baked
# here is the runner for the compiled per-checker suite, pinned by version so the suite that
# proves both halves of every checker runs inside this image and not only on an author's
# host. This base is Debian 12 with Python 3.12 and enforces PEP 668, so the install carries
# --break-system-packages. It happens at BUILD time; at graded time egress is denied.
RUN pip install --no-cache-dir --break-system-packages "pytest==9.1.1"

WORKDIR /task

COPY tests/ /task/tests/

COPY environment/ /task/environment/
# The verifier rebuilds the near-duplicate index itself, from the bundle's frozen corpus, in
# an isolated subprocess, and recomputes the threshold, the duplicate grouping and the whole
# near-duplicate pair set from that index. It reads no clock.

RUN mkdir -p /logs/verifier /logs/agent
ENTRYPOINT ["/bin/bash", "/task/tests/test.sh"]
