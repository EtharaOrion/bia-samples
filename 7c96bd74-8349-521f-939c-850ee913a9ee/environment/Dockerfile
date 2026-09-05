# The agent environment. Pinned by digest exactly as the verifier environment is; a
# floating tag would let the substrate the agent solves against move under a frozen
# bundle.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Solver egress is OPEN inside the sandbox so the agent can fetch its own packages,
# datasets and models. The deny-list bound in task.toml is not a containment proof:
# mirrors, caches, paraphrased write-ups and model weights all carry a recipe past any
# deny-list. The deterministic near-duplicate screen in tests/checkers.py is the
# load-bearing control, not the network policy.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    LC_ALL=C \
    LANG=C

# The GPT-2 BPE the FineWeb10B shards were built with, cached at build time so the local
# mirror tokenizes the same way the verifier does.
RUN python3 -m pip install --no-cache-dir --break-system-packages "tiktoken==0.9.0" \
 && python3 -c "import tiktoken; tiktoken.get_encoding('gpt2')"

WORKDIR /workspace
COPY trainer.py /workspace/trainer.py
COPY dev_examples.jsonl /workspace/dev_examples.jsonl
COPY GENERATOR_CONTRACT.md /workspace/GENERATOR_CONTRACT.md
COPY nanogpt_substrate.json /workspace/nanogpt_substrate.json
RUN mkdir -p /workspace/submission

# The FineWeb10B TRAIN shards, so the local mirror builds the same mixed batch the graded
# run builds. This mirrors data/cached_fineweb10B.py from the record set with one
# deliberate divergence: the upstream loader also fetches fineweb_val_000000.bin and this
# build does not. Two chunks is 200M tokens, enough for the short local passes the mirror
# is for, and the VALIDATION shards are absent by construction.
RUN python3 -m pip install --no-cache-dir --break-system-packages "huggingface_hub==0.27.1" \
 && mkdir -p /workspace/data/fineweb10B \
 && python3 -c "\
from huggingface_hub import hf_hub_download;\
[hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename='fineweb_train_%06d.bin' % i, repo_type='dataset', local_dir='/workspace/data/fineweb10B') for i in range(1, 3)]"

# The held-out benchmark is NOT here and is not reachable from here. It lives under
# tests/ only, and Harbor assembles the non-oracle agent surface from task.toml,
# instruction.md and environment/ alone.
CMD ["/bin/bash"]
