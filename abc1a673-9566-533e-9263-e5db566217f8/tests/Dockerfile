# The isolated verifier environment.
#
# Same digest pin as the task environment, and pinned by digest for the same reason: a floating
# tag would let the grading interpreter move underneath a frozen bundle.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# EGRESS IS DENIED on this surface. The solving agent's surface has open egress; these two
# surfaces differ by design and the asymmetry is bound in task.toml. A verifier that can reach
# the network is a verifier whose verdict can be fetched.
ENV OER26_VERIFIER_EGRESS=denied
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /task

# The one third-party dependency on this surface, baked at build time and pinned to an exact
# version, so nothing is fetched at graded time and the compiled per-checker suite actually runs
# instead of being skipped. The base is Debian 12 with Python 3.12 and enforces PEP 668, which is
# why --break-system-packages is required here.
RUN pip install --no-cache-dir --break-system-packages "pytest==8.3.5"

# The verifier builds the SAME fence state from the SAME generator. The build is deterministic
# over an integer recurrence, so both images carry byte-identical state and the verifier's own
# traversal is over state it established rather than state a submission handed it.
COPY environment/fencelib.py /opt/fence/fencelib.py
COPY environment/build_fence.py /opt/fence/build_fence.py
COPY environment/fence_probe.py /opt/fence/fence_probe.py
RUN python3 /opt/fence/build_fence.py --root /opt/fence/state \
    && chmod -R a-w /opt/fence/state

COPY tests/ /task/tests/

ENV OER26_FENCE_STATE=/opt/fence/state
ENV OER26_FENCE_PROBE=/opt/fence/fence_probe.py

ENTRYPOINT ["/bin/bash", "/task/tests/test.sh"]
