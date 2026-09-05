# The agent-facing task environment.
#
# Pinned by digest, never by a floating tag. Harbor accepts any string and does not itself
# require a digest, so the digest requirement is this contract's and this file enforces it.
#
# TWO STAGES, ON PURPOSE. The first stage RUNS the minter and materialises the sealed
# provenance store. The second stage copies the STORE and leaves the minter and the ledger
# source behind. The realised attestation order and every atom digest therefore exist in the
# shipped image only as built state, reachable through the read-only handle by walking the
# seal chain, and not as a script the agent can re-run to be told the answer.
FROM python@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7 AS minter

WORKDIR /build
COPY mint_store.py /build/mint_store.py
COPY ledger_source.json /build/ledger_source.json
COPY instances.json /build/instances.json

# The store is established HERE, at image build time, by arithmetic over the frozen source.
# No network, no clock and no random source is consulted, so the same source produces a
# byte-identical store on any host.
RUN python3 /build/mint_store.py \
        --source /build/ledger_source.json \
        --instances /build/instances.json \
        --out /build/store

FROM python@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7

# The one baked dependency. It is installed at BUILD time and the graded surface has no
# egress, so nothing is resolved while a run is being graded.
RUN pip install --no-cache-dir "pytest==8.3.5"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0
ENV OER30_STORE=/task/state/store
ENV OER30_INSTANCES=/task/environment/instances.json

WORKDIR /workspace

COPY --from=minter /build/store /task/state/store
COPY store_handle.py /task/environment/store_handle.py
COPY instances.json /task/environment/instances.json

# Read-only in spirit and in mode: the store is built state and a run that rewrites it is
# rewriting the thing it is supposed to be reading.
RUN chmod -R a-w /task/state/store && mkdir -p /workspace

CMD ["/bin/bash"]
