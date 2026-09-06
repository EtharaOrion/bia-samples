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

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/. tests/grade.py resolves its
# own bundle root as the parent of the directory it was laid down in and then
# reads a tree BESIDE tests/: grade.py:168 sets frozen_dir to
# ${BUNDLE}/environment/frozen and hands it to tests/harness.py::load_frozen,
# which subscripts five declarations out of it, and to tests/runner.py, which
# copies the whole directory in beside the submission. A tests-only context
# builds an image that carries none of those bytes, and the first read dies on
# FileNotFoundError for /verifier/environment/frozen/model_config.json inside a
# container that had already started, at which point test.sh writes reward 0.0
# with reason grading-chain-failed and nothing is ever graded. Whatever the
# entry point reads has to be in the image that runs it.
#
# The grading tree is never mounted on the agent surface: Harbor assembles a
# non-oracle agent surface from task.toml, instruction.md and environment/ only,
# so tests/ and solution/ are absent from it by construction rather than removed
# by a filter that could be misconfigured. Copying the whole tests/ directory
# from the bundle root reproduces exactly the image content the delivered
# tests-only `COPY . /verifier/tests/` produced, so nothing this surface carried
# is dropped.
COPY tests/ /verifier/tests/

# The frozen half of the task, staged EXACTLY the way environment/Dockerfile
# stages it for the agent at lines 23 and 29: the frozen directory first, then
# the canonical operating point laid down INSIDE it. environment/Dockerfile is
# the producer of the runtime layout, and it puts nanogpt_substrate.json at
# /app/frozen/nanogpt_substrate.json even though the bundle carries the file one
# level up beside frozen/. tests/harness.py::load_frozen:77 subscripts
# `base / "nanogpt_substrate.json"` off the frozen directory it is handed, so it
# is reading the layout environment/Dockerfile produces and the verifier image
# is the surface that was failing to produce it. Both COPY lines are needed and
# the order between them is load bearing: the second must follow the first, or
# the directory copy overwrites it.
#
# The verifier reads its OWN copy at grading time, so editing the agent's copy
# produces a refused run with reason frozen-axis-mutated rather than a different
# grade. tests/ reads nothing else under environment/, so nothing else is
# carried here.
COPY environment/frozen/ /verifier/environment/frozen/
COPY environment/nanogpt_substrate.json /verifier/environment/frozen/nanogpt_substrate.json

# solution/ is deliberately NOT copied. tests/ reads no file under it: every
# checker fixture is embedded in tests/test_output.py rather than opened from
# solution/fixtures/, and tests/grade.py imports only checkers, harness and
# runner. Baking solution/solve.sh in would additionally put the reference
# solver inside the image the verifier runs, which is the reference's own work
# and not a submission.

# The corpus. Build egress is open; run-time egress is denied, so both halves
# are staged here at build. This image is the ONLY place the validation shards
# exist: they are stripped from the agent image, which is what makes the graded
# figure something a submission cannot precompute.
RUN mkdir -p /verifier/data \
 && python3 -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/data/cached_fineweb10B.py', '/verifier/data/cached_fineweb10B.py')" \
 && cd /verifier/data && python3 cached_fineweb10B.py 20 \
 && ls /verifier/data/fineweb10B/fineweb_val_000000.bin

# The GPT-2 BPE encoding tables themselves, resolved at BUILD time into a cache
# directory this image owns. `pip install tiktoken` installs the library and
# none of its data: tests/harness.py:201 calls tiktoken.get_encoding("gpt2"),
# which fetches vocab.bpe and encoder.json from openaipublic.blob.core.windows.net
# on first use and caches them under $TIKTOKEN_CACHE_DIR. Run-time egress is
# DENIED at this surface by the policy the ENV block above declares, so that
# fetch cannot succeed in a graded run: it raises ConnectionError out of
# harness._decoder, grade.py dies and test.sh writes reward 0.0 with reason
# grading-chain-failed, having graded nothing. The vocabulary is a fixed input
# to the measurement rather than something the run discovers, so it is staged
# here beside the corpus it decodes, exactly as the corpus is.
ENV TIKTOKEN_CACHE_DIR=/verifier/tiktoken
RUN mkdir -p /verifier/tiktoken \
 && python3 -c "import tiktoken; enc = tiktoken.get_encoding('gpt2'); assert enc.n_vocab == 50257, enc.n_vocab" \
 && test -n "$(ls -A /verifier/tiktoken)"

# The reward carrier. /logs/verifier/reward.txt takes the bare float and
# /logs/verifier/score.json takes the reason and metric block.
RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
