# The agent-facing task environment.
#
# Build context is the BUNDLE ROOT, not this directory:
#
#     docker build -f environment/Dockerfile -t bia-oer29-env .
#
# The context reaches outside environment/ on purpose. The warehouse is built here, in a first
# stage, from a spec that lives on the private side of the bundle, and only the finished
# database crosses into the image the agent receives. The builder and the spec are discarded
# with the first stage, so the quarantine is a property of the built state rather than a
# constant sitting in a file the agent can read.
#
# Pinned by digest, never by a floating tag. The database engine version has to be identical on
# this surface and on the verifier surface, because the graded cost is that engine's own step
# count; a floating tag would let the engine move underneath a frozen bundle and move the cost
# with it. Both Dockerfiles carry the same digest for exactly that reason.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407 AS warehouse-build

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /build
COPY solution/warehouse_build.py /build/warehouse_build.py
COPY solution/warehouse_spec.json /build/warehouse_spec.json

# The whole warehouse, composed and written here. Deterministic by construction: one fixed
# integer recurrence, one fixed insert order, the page geometry pinned before the first write,
# no clock, no locale, no network and no module named random.
RUN python3 /build/warehouse_build.py --spec /build/warehouse_spec.json --out /build/warehouse.db

FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Everything the substrate needs is in the standard library, so no package is installed here
# and no wheel has to resolve the same way twice for the grading path to hold. This base is
# Debian 12 with PEP 668 in force; had anything needed installing it would have needed
# --break-system-packages, and nothing does.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0
ENV OER29_WAREHOUSE=/task/environment/warehouse.db

WORKDIR /workspace

COPY --from=warehouse-build /build/warehouse.db /task/environment/warehouse.db
COPY environment/planrun.py /task/environment/planrun.py
COPY environment/probe.py /task/environment/probe.py

# The built database is read-only to the agent. The exclusion rule is evaluated against these
# bytes and nowhere else, so the bytes are not writable.
RUN chmod 0444 /task/environment/warehouse.db

ENV PYTHONPATH=/task/environment

CMD ["/bin/bash"]
