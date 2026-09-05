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

# Staged at BUILD time: provisioning egress is public, run-time egress is not.
# Shards 1-5 are corpus.shards, the 640-slice pool; shard 6 is dev_probe.shard.
# Six is the whole set corpus_spec.json names and nothing beyond it is fetched.
#
# hf_hub_download per train file, NOT the vendored cached_fineweb10B.py: that
# loader also pulls fineweb_val_000000.bin, and the graded split lives verifier-
# side at tests/heldout_spec.json and must be absent from every agent path.
RUN pip install --no-cache-dir --break-system-packages huggingface_hub \
 && python3 -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('/workspace/data/fineweb10B', exist_ok=True); [hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename='fineweb_train_%06d.bin' % i, repo_type='dataset', local_dir='/workspace/data/fineweb10B') for i in range(1, 7)]" \
 && rm -f /workspace/data/fineweb10B/fineweb_val_*.bin \
 && ls -l /workspace/data/fineweb10B/

# Solver egress is open at this surface, by task.toml [agent].solver_egress.
# The verifier surface denies egress; the two surfaces differ by design.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

CMD ["/bin/bash"]
