# Isolated verifier environment for OER-16.
#
# Same pinned digest as the agent environment, and pinned for the same reason: a
# floating tag validates cleanly and silently resolves to different bytes.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The verifier's network policy is DENIED at run time. This differs from the
# agent surface by design: the agent has open egress so it can fetch packages,
# datasets and models; the verifier fetches nothing once the run starts. The
# asymmetry is bound in task.toml and the divergence is recorded in
# solution/grounding.yaml.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER16_VERIFIER_EGRESS=denied \
    OER16_SLOT=OER-16 \
    OER16_VERIFIER_DATA=/verifier/data/fineweb10B \
    OER16_TRAIN_DATA=/verifier/data/fineweb10B

WORKDIR /verifier

# The GPT-2 BPE vocabulary the FineWeb shards were written with. The harness
# decodes shards back to UTF-8 text with it before re-tokenizing that text with
# the submitted vocabulary.
RUN pip install --no-cache-dir --break-system-packages tiktoken numpy huggingface_hub

# The grading tree. It is never mounted on the agent surface: Harbor assembles a
# non-oracle agent surface from task.toml, instruction.md and environment/ only,
# so tests/ and solution/ are absent from it by construction rather than removed
# by a filter that could be misconfigured.
COPY . /verifier/tests/

# The corpus. Build egress is open; run-time egress is denied, so both halves
# are staged here at build. This image is the ONLY place the validation shards
# exist: they are stripped from the agent image, which is what makes the graded
# figure something a submission cannot precompute.
RUN mkdir -p /verifier/data \
 && python3 -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/data/cached_fineweb10B.py', '/verifier/data/cached_fineweb10B.py')" \
 && cd /verifier/data && python3 cached_fineweb10B.py 20 \
 && ls /verifier/data/fineweb10B/fineweb_val_000000.bin

# The reward carrier. /logs/verifier/reward.txt takes the bare float and
# /logs/verifier/score.json takes the reason and metric block.
RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
