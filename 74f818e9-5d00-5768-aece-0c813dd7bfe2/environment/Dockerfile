# OER-18 agent environment.
#
# Pinned by digest, never by a floating tag. Harbor accepts any string and does not itself
# require a digest; the digest requirement is this contract's, and this lane enforces it.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The solving agent has open internet egress inside the sandbox so it can fetch its own
# packages, datasets and models. The verifier environment does not; see tests/Dockerfile.
# The deny-list bound in task.toml is not a containment proof and is not treated as one.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C \
    LANG=C \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /workspace

RUN pip install --no-cache-dir --break-system-packages tiktoken==0.7.0

# The GPT-2 BPE is baked in at build time. The scored run has no egress, and the verifier
# has none at all, so an encoder that resolved itself over the network at grading time would
# be a graded path that fails closed on a name lookup.
RUN mkdir -p /opt/tiktoken && python3 -c "import tiktoken; tiktoken.get_encoding('gpt2')"

# The task substrate. train_stack.py is the frozen nanoGPT decoder, optimizer and feeder;
# nanogpt_substrate.json is the canonical operating point it reads its architecture from.
# The verifier runs its own copy of the same bytes, whose sha256 is bound in
# tests/checkers.yaml.
COPY train_stack.py /workspace/train_stack.py
COPY nanogpt_substrate.json /workspace/nanogpt_substrate.json
COPY dev_benchmark.json /workspace/dev_benchmark.json

# environment/corpus/ is deliberately NOT copied. It carries a redaction notice standing in
# for a held-out evaluation payload an earlier pass wrote onto this surface. The held-out
# split lives only under tests/ and is never assembled into this image.

# The submission directory the verifier stages from. The agent writes generator.py here.
RUN mkdir -p /workspace/submission

CMD ["/bin/bash"]
