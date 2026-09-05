# Isolated VERIFIER environment for slot OER-14.
#
# Pinned by digest, same digest as the agent environment, because a verifier grading on a
# different stack than the one the agent ran on is measuring two things at once.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /verifier

# Set BEFORE the warm below, so the encoder file lands in the baked cache and not in a
# default location the offline run-time environment would then miss.
ENV TIKTOKEN_CACHE_DIR=/verifier/.tiktoken

# BUILD-TIME provisioning. The GPT-2 BPE decoder and the FineWeb VALIDATION shard are both
# fetched here, while the build still has egress, and are baked into the image. Nothing on
# the grading path fetches anything, which is what makes the denied run-time egress below
# cost the verifier nothing.
RUN pip install --no-cache-dir tiktoken huggingface_hub \
 && python3 -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('/verifier/data/fineweb10B', exist_ok=True); \
hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename='fineweb_val_000000.bin', \
repo_type='dataset', local_dir='/verifier/data/fineweb10B'); \
hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename='fineweb_train_000001.bin', \
repo_type='dataset', local_dir='/verifier/data/fineweb10B')" \
 && python3 -c "import tiktoken; tiktoken.get_encoding('gpt2')"

# THE HELD-OUT SPLIT LIVES IN THIS IMAGE AND IN NO OTHER. The validation shard staged above
# is the only copy in this bundle's delivery closure, and the slice inside it that scores a
# submission is fixed by the shard name, the token offset and the token count in
# tests/controls.json, which this image carries and the agent image does not. The agent image
# stages the train glob alone, so the absence on that surface is a matter of what was never
# copied rather than of a filter that could be misconfigured.

# EGRESS IS DENIED on this surface at run time. It is bound in task.toml under
# [verifier] egress. The agent surface is open by design and the two surfaces differ on
# purpose, and this removes the only path by which a graded number could arrive from outside
# this image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 \
    HTTP_PROXY=http://127.0.0.1:9 \
    HTTPS_PROXY=http://127.0.0.1:9 \
    NO_PROXY="" \
    OER_SLOT=OER-14

# The verifier owns the substrate, the canonical declaration, the corpus declaration and the
# shards, so every number on the graded path is produced inside this image, by training the
# frozen decoder on bytes this image carries.
COPY environment/substrate.py /verifier/environment/substrate.py
COPY environment/nanogpt_substrate.json /verifier/environment/nanogpt_substrate.json
COPY environment/corpus /verifier/environment/corpus
COPY tests /verifier/tests

# The reward contract root. tests/test.sh writes the bare float to
# /logs/verifier/reward.txt and the reason and metric block to /logs/verifier/score.json.
RUN mkdir -p /logs/verifier

# The submission is MOUNTED at grade time; it is never baked in and never imported. See
# tests/runner.py, which copies it alone into a fresh temporary directory and launches it as
# a new session leader whose whole process group is killed in a finally block.
VOLUME ["/workspace"]

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
