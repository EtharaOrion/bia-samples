# OER-09 agent environment.
# Pinned by digest, never by a floating tag. A tag without a digest is a lane failure.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The solving agent has open egress by bound design so it can fetch its own packages,
# datasets and models. The upstream deny-list is carried by name in task.toml and is NOT
# a containment proof: mirrors, caches, paraphrased write-ups and released artifacts all
# carry a recipe past any deny-list. The deterministic checkers are the load-bearing control.
ENV OER_SLOT=OER-09
ENV OER_FAMILY=F12
ENV OER_SOLVER_EGRESS=open
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /task

COPY frozen_config.yaml       /task/environment/frozen_config.yaml
COPY nanogpt_substrate.json   /task/environment/nanogpt_substrate.json
COPY curate.py                /task/environment/curate.py
COPY observe.py               /task/environment/observe.py
COPY train.py                 /task/environment/train.py
COPY pool_probe.py            /task/environment/pool_probe.py
COPY README.md                /task/environment/README.md
COPY pool                     /task/environment/pool

# The FineWeb10B TRAIN shards are staged here at provisioning by the upstream loader named
# in environment/frozen_config.yaml corpus.loader. No shard matching corpus.split_shards is
# copied or fetched into this image: the held-out validation split is the verifier's, it is
# absent from every agent-visible container, and the graded loss is the verifier's own
# recomputation over its own copy.
#
# Staged at BUILD time: provisioning egress is public, run-time egress is not.
# frozen_config.yaml budget is 419430400 tokens fed once (fed_once: true) and
# train.py's stream() never wraps, so the shards must physically hold that many
# admitted tokens. The reference chain admits 1959 of 4096 register rows, so
# roughly 47.8 percent of documents survive; ten 100M-token shards yield about
# 478M admitted tokens against the 419.4M budget. Fewer than nine would exhaust
# the corpus and exit 3, "the admitted corpus was exhausted".
#
# hf_hub_download per train file, NOT the loader named in corpus.loader: that
# loader also pulls fineweb_val_000000.bin, which is corpus.split_shard and is
# declared split_present_in_environment: false.
RUN mkdir -p /workspace/data/fineweb10B \
 && pip install --no-cache-dir --break-system-packages huggingface_hub \
 && python3 -c "from huggingface_hub import hf_hub_download; [hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename='fineweb_train_%06d.bin' % i, repo_type='dataset', local_dir='/workspace/data/fineweb10B') for i in range(1, 11)]" \
 && rm -f /workspace/data/fineweb10B/fineweb_val_*.bin \
 && ls -l /workspace/data/fineweb10B/

# The harness owns /logs/harness and writes its own observations there. The agent reads it.
# The agent never writes it, and nothing the agent writes there is ever read as graded state.
# /logs/harness/weights holds the parameter snapshots train.py writes at the points the
# verifier's schedule names; the harness digests and ledgers them and the submission
# selects none of them.
RUN mkdir -p /task/work /logs/harness /logs/harness/weights

CMD ["/bin/bash"]
