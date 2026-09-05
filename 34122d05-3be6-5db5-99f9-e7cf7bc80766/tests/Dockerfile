# The isolated verifier environment. Pinned by digest, never by a floating tag: a tag
# is a mutable pointer and a bundle pinned to one is a bundle whose grading environment
# can change without a byte of this tree moving.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The verifier surface keeps the bound network policy: egress denied. task.toml binds
# [verifier] egress = "denied" while [agent] binds solver_egress = "open". The two
# surfaces differ by design and the divergence is recorded in solution/grounding.yaml.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    LC_ALL=C \
    LANG=C

# PyYAML reads tests/checkers.yaml. tiktoken carries the GPT-2 BPE the FineWeb10B shards
# were built with, and the encoding is pre-cached at BUILD time because the graded run
# denies egress and a run-time fetch would turn a grade into a network dependency.
RUN python3 -m pip install --no-cache-dir --break-system-packages "pyyaml==6.0.2" "tiktoken==0.9.0" \
 && python3 -c "import tiktoken; tiktoken.get_encoding('gpt2')"

# The FineWeb10B TRAIN shards, staged at build time. This mirrors data/cached_fineweb10B.py
# from the record set, which the substrate declaration names as the loader, with ONE
# deliberate divergence: the upstream loader also fetches fineweb_val_000000.bin and this
# build does not. The validation shards are absent from this image on purpose, because the
# split that scores a submission is tests/heldout/benchmark.jsonl, which arrives with the
# grading tree below and is digested on both sides of every run. Fifteen chunks is 1.5B
# tokens, above the 1278 million FineWeb tokens the frozen 3250-step budget draws at
# 393216 tokens per step.
RUN python3 -m pip install --no-cache-dir --break-system-packages "huggingface_hub==0.27.1" \
 && mkdir -p /verifier/data/fineweb10B \
 && python3 -c "\
from huggingface_hub import hf_hub_download;\
[hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename='fineweb_train_%06d.bin' % i, repo_type='dataset', local_dir='/verifier/data/fineweb10B') for i in range(1, 16)]"

WORKDIR /verifier
COPY . /verifier/tests

# The grading tree is copied in whole and the submission is never copied in beside it.
# tests/runner.py copies the submission alone into a fresh temporary directory at run
# time, so the generation process never shares a working directory with the held-out
# split, the FineWeb shards or the parameter snapshots.
ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
