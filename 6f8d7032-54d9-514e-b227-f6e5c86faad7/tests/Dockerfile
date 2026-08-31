# The isolated VERIFIER PHASE environment.
#
# Same digest pin as the task environment, and pinned by digest for the same reason: a floating
# tag would let the grading interpreter move underneath a frozen bundle.
#
# The build context is the BUNDLE ROOT, because this image needs two of the bundle's trees. It
# needs tests/ for the grading path and it needs environment/ for the frozen stream, the frozen
# parameters and the resume driver, since the graded charge has to be counted by this phase over
# the bundle's own bytes rather than over anything the agent phase supplied. solution/ is never
# copied in.
FROM python@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7

# EGRESS IS DENIED on this surface. The solving agent's surface has its own policy; these two
# surfaces differ by design and the asymmetry is bound in task.toml. A verifier that can reach
# the network is a verifier whose verdict can be fetched.
ENV OER28_VERIFIER_EGRESS=denied
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /task

# Baked at build time, which is the only time this image is allowed to reach a network. The
# graded run has egress denied and installs nothing.
RUN pip install --no-cache-dir --disable-pip-version-check pytest==8.3.4

COPY tests/ /task/tests/
COPY environment/ /task/environment/

# The verifier resumes the operator from the carry the agent phase handed over, in an isolated
# subprocess, and counts the reads on its own Tape. It reads no clock; the cost model is an
# instrumented read counter and is arithmetic.
ENTRYPOINT ["/bin/bash", "/task/tests/test.sh"]
