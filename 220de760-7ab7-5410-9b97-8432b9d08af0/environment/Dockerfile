# OER-08 agent environment.
# Pinned by digest, never by a floating tag: a tag is a mutable pointer and a
# bundle pinned to one is not frozen.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /workspace

# The agent surface Harbor assembles carries task.toml, instruction.md and this
# directory. solution/ and tests/ are absent by construction, not by a filter.
COPY nanogpt_substrate.json /workspace/environment/nanogpt_substrate.json
COPY corpus_spec.json /workspace/environment/corpus_spec.json
COPY graded_readout.json /workspace/environment/graded_readout.json
COPY mixture.yaml /workspace/environment/mixture.yaml
COPY plan_schema.md /workspace/environment/plan_schema.md
COPY harness.py /workspace/environment/harness.py

# FineWeb10B TRAIN shards are staged here by the upstream loader named in
# corpus_spec.json. No validation shard is staged into this image: the graded
# split lives verifier-side at tests/heldout_spec.json and is absent from every
# agent-reachable path by construction.
ENV OER08_DATA=/workspace/data/fineweb10B

# Solver egress is open at this surface, by task.toml [agent].solver_egress.
# The verifier surface denies egress; the two surfaces differ by design.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

CMD ["/bin/bash"]
