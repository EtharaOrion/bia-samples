# Agent container spec for BIA S02.
#
# Harbor reads this as the MAIN container. This task declares no
# [verifier.environment], so environment_mode is shared and the verifier runs in
# this same image, which it must: a separate verifier container could not reach
# the agent's /telemetry, and /telemetry is the record grade.py grades.
#
# The base is pinned by digest and never by a floating tag, because the digest is
# what binds this task to the runtime that produced its recorded results.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Setup-only egress. Everything the graded run needs is staged here, at build
# time, so the scored run needs no network at all.
RUN pip install --no-cache-dir numpy pyyaml pytest

WORKDIR /workspace

# The frozen surface. The agent reads all of it and may edit none of it: the
# verifier compares the schedule and the architecture against digests it pins
# independently, and a run whose telemetry reports an edited module scores zero.
COPY frozen_schedule.py /workspace/environment/frozen_schedule.py
COPY runner /workspace/environment/runner
COPY starter /workspace/environment/starter
COPY fixtures /workspace/environment/fixtures

RUN mkdir -p /workspace/submission /telemetry /logs/verifier /data/fineweb

# The real training data. FineWeb-10B GPT-2 tokens, the same public shards the
# upstream track-3 benchmark trains on, staged at build time and verified by
# sha256 so the graded run reads bytes whose identity is fixed in this file
# rather than whatever a later fetch happens to return. Setup-only egress is what
# makes this the right place for it.
ARG FINEWEB_BASE=https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main
RUN set -eux; \
    for shard in fineweb_train_000001.bin fineweb_val_000000.bin; do \
      curl -fsSL -o "/data/fineweb/$shard" "$FINEWEB_BASE/$shard?download=true"; \
    done; \
    cd /data/fineweb; \
    echo "771fa4a99b9fe0946ffb6e848b4ba5c6a9b0fe87860ebf03bc2c1c7e45f8178e  fineweb_train_000001.bin" > SHA256SUMS; \
    echo "5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640  fineweb_val_000000.bin" >> SHA256SUMS; \
    sha256sum -c SHA256SUMS

ENV BIA_TELEMETRY_DIR=/telemetry \
    BIA_DATA_DIR=/data/fineweb \
    PYTHONUNBUFFERED=1

# The chain key is deliberately absent from this image and must stay absent. It
# is injected per campaign into the verifier environment. A literal here would
# put the key that signs the telemetry into the solving agent's own shell.
