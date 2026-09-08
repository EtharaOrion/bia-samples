# The isolated verifier environment.
#
# Build context is the BUNDLE ROOT, not this directory:
#
#     docker build -f tests/Dockerfile -t bia-oer29-verifier .
#
# The whole graded surface is baked in: the frozen warehouse, the harness executor, the checker
# module, the bound constants and the frozen fixtures. Nothing the verifier grades against is
# mounted at run time, so a run that mounts nothing but a workspace still reaches a verdict.
#
# Same digest pin as the task environment, and pinned by digest for the same reason twice over.
# A floating tag would let the grading interpreter move underneath a frozen bundle, and on this
# slot it would also move the database engine, whose own step count is the graded cost. The two
# surfaces have to agree on that engine or the cost model is not a cost model.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# EGRESS IS DENIED on this surface. The solving agent's surface has provisioning egress; these
# two surfaces differ by design and the asymmetry is bound in task.toml. A verifier that can
# reach the network is a verifier whose verdict can be fetched.
ENV OER29_VERIFIER_EGRESS=denied
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

# The single dependency the graded path takes, baked at build time so no resolution happens at
# graded time. This base is Debian 12 with PEP 668 in force, so the install needs
# --break-system-packages; the version is pinned because an unpinned runner is a runner that can
# change a verdict. The grading path itself, tests/grade.py, imports only the standard library:
# pytest drives the compiled per-checker suite and nothing else.
RUN pip install --no-cache-dir --break-system-packages 'pytest==8.3.5'

WORKDIR /task

COPY environment /task/environment
COPY tests /task/tests
COPY solution/fixtures /task/solution/fixtures
COPY solution/warehouse_spec.json /task/solution/warehouse_spec.json

# The verifier re-runs the submitted plan and the frozen baseline plan on the harness-owned
# executor and computes the graded speedup from its own step counts. It reads no clock.
ENTRYPOINT ["/bin/bash", "/task/tests/test.sh"]
