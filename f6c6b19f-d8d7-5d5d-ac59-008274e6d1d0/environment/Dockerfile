# OER-13 agent environment. The surface the solving agent works on.
#
# The image is pinned by digest, never by a floating tag. A tag is a moving
# pointer, so a bundle pinned to one is not the same bundle tomorrow, and the
# frozen axes this task grades would silently move underneath the grader.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Solver egress is OPEN inside the sandbox by design, so the agent can fetch its
# own packages, datasets and models. The deny-list bound in task.toml under
# [agent].solver_egress_deny is a friction, not a containment proof: mirrors,
# caches, paraphrased write-ups and model weights all carry a recipe past any
# deny-list. The load-bearing control is the deterministic checker in tests/,
# which recomputes the graded quantity from bytes the verifier owns.
ENV OER13_SOLVER_EGRESS=open
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0
ENV LC_ALL=C
ENV LANG=C

WORKDIR /workspace

# The frozen encoder, installed AND warmed at build time. The token space is a
# frozen axis: the verifier's held-out
# shards are encoded with the GPT-2 byte pair encoder, so a training stream in
# any other id space is not comparable to the split it is scored against. It is
# installed at build time because the verifier that measures the delivered
# baseline runs with egress denied and cannot fetch it later.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN pip install --no-cache-dir "tiktoken==0.13.0" \
 && mkdir -p /opt/tiktoken_cache \
 && python -c "import tiktoken; tiktoken.get_encoding('gpt2')" \
 && chmod -R a+rX /opt/tiktoken_cache

# The whole agent-visible substrate. task.toml and instruction.md are mounted by
# Harbor alongside it; tests/ and solution/ are absent from this surface by
# construction and no COPY here can reach them.
#
# What is NOT here, and cannot be reached from here: the held-out FineWeb
# validation shards the graded loss is computed on. environment/eval/ carries a
# notice and the fold declaration and none of the split's bytes.
COPY . /workspace/environment

# The three files the agent is licensed to rewrite. Everything else under
# environment/ is frozen and digested against the delivered bytes by the
# verifier: nanogpt_substrate.json, pipeline/train.py, pipeline/protocol.json,
# eval/* and corpus/*. The substrate declaration leads that list because it is
# what pins the architecture the graded parameters must instantiate.
LABEL oer13.writable="environment/pipeline/parse.py,environment/pipeline/tokenize.py,environment/pipeline/compose.json"
LABEL oer13.frozen="environment/nanogpt_substrate.json,environment/pipeline/train.py,environment/pipeline/protocol.json,environment/eval/*,environment/corpus/*"
LABEL oer13.substrate="environment/nanogpt_substrate.json"
LABEL oer13.architecture="nanoGPT, vocab_size 50304, 12 layers, model_dim 768, head_dim 128, 6 heads, seq_len 1024"
LABEL oer13.accelerators="1"
LABEL oer13.compute_envelope="one H100, single accelerator, no multi-GPU scaling"

CMD ["/bin/bash"]
