# OER-11 agent environment.
#
# Pinned by digest, never by a floating tag. Harbor accepts any string and does not
# itself require a digest, so the digest requirement is this contract's and this lane
# enforces it.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The solving agent has OPEN egress inside the sandbox so it can fetch its own
# packages, datasets and models. The verifier environment in tests/Dockerfile is
# egress-denied. The two surfaces differ by design; see task.toml [agent] and
# [verifier], and solution/grounding.yaml solver_egress_divergence.
ENV OER11_SOLVER_EGRESS=open

WORKDIR /task

COPY corpus /task/corpus
COPY corpus_api.py /task/corpus_api.py
COPY train.py /task/train.py
COPY pipeline.py /task/pipeline.py
COPY README.md /task/README.md

# The corpus service state lives under a harness-owned root the agent reads through
# corpus_api.py and never writes. The harness mounts it; this line only declares where
# it lands, and a declaration is not an enforcement.
ENV OER11_CORPUS_ROOT=/task/corpus
ENV OER11_HARNESS_LOGS=/logs/harness

CMD ["python3", "-c", "import corpus_api; print(corpus_api.snapshot_version())"]
