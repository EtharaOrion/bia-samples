# Agent environment for slot OER-14.
#
# The image is pinned BY DIGEST and never by a floating tag: a tag is a mutable pointer, so
# a bundle pinned to one is not frozen and a rebuild months later grades a different stack.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The SOLVING agent has OPEN egress inside the sandbox so it can fetch its own packages,
# datasets and models. The VERIFIER does not: tests/Dockerfile builds a separate environment
# whose egress is denied. The asymmetry is deliberate and is bound in task.toml under
# [agent] solver_egress and [verifier] egress. The upstream deny-list is applied by the
# runtime, not by this file, and it is not a containment proof.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-14 \
    SUBMISSION_PATH=/workspace/submission.json

WORKDIR /workspace

# tiktoken carries the GPT-2 BPE decoder the corpus loader and the substrate both use to turn
# FineWeb shard token ids back into the underlying bytes. It is installed at BUILD time,
# while provisioning egress is open, because the scored run has no network.
ENV TIKTOKEN_CACHE_DIR=/workspace/.tiktoken
RUN pip install --no-cache-dir --break-system-packages tiktoken huggingface_hub \
 && python3 -c "import tiktoken; tiktoken.get_encoding('gpt2')" \
 && chmod -R a+rX /workspace/.tiktoken

# The frozen substrate, the replicated canonical operating point, and the corpus declaration.
# Nothing under solution/ or tests/ is copied here: the oracle and the held-out checkers are
# left behind by CONSTRUCTION rather than by a filter that could be misconfigured into
# letting them through.
COPY environment/substrate.py /workspace/environment/substrate.py
COPY environment/nanogpt_substrate.json /workspace/environment/nanogpt_substrate.json
COPY environment/corpus /workspace/environment/corpus
COPY instruction.md /workspace/instruction.md
COPY task.toml /workspace/task.toml

# The FineWeb10B TRAINING shards, staged by the canonical loader named in the declaration at
# corpus.loader. Only the train glob is staged here. THE VALIDATION SHARD IS NEVER STAGED ON
# THIS SURFACE: the slice that scores a submission is materialised in tests/Dockerfile alone,
# from a token offset and a token count that live in the verifier-only control table. That is
# an absence by construction, not a filter, so there is nothing here to misconfigure.
RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('/workspace/data/fineweb10B', exist_ok=True); \
hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename='fineweb_train_000001.bin', \
repo_type='dataset', local_dir='/workspace/data/fineweb10B')"

# The lower layer is read-only. The agent writes its submission, and nothing else.
RUN chmod -R a-w /workspace/environment /workspace/data \
 && chmod a-w /workspace/instruction.md /workspace/task.toml

CMD ["/bin/bash"]
