# --- FORGE Wave One repair -------------------------------------------------
# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
#
# This file previously carried a bare `COPY . <dest>`, which only assembles a
# working image when the build context is this tests/ directory.  Under that
# context the image holds the verifier and nothing else, so the grader's
# BUNDLE_DIR resolves to a directory with no environment/ beside it.  The first
# read of environment/<anything> raises, the EXIT trap writes reward 0.0, and
# the slot returns a constant zero for every possible submission - including
# the reference.  That is not a hard task; it is a dead one.
#
# Naming what is copied fixes it and keeps solution/ out of the verifier image.
# ---------------------------------------------------------------------------
# OER-18 verifier environment. Isolated from the agent environment on purpose.
#
# Same digest pin as environment/Dockerfile, because the graded stack must be the stack the
# agent saw. Pinned by digest, never by a floating tag.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Verifier egress is DENIED. This is the asymmetry task.toml binds: the solving agent has
# open egress inside its sandbox, and the verifier has none. Nothing on the grading path
# resolves a name, opens a socket, or fetches a byte. Every graded number is computed here
# from the corpus the submission emitted and the frozen bytes in this tree.
#
# CUBLAS_WORKSPACE_CONFIG is not a tuning knob here. tests/grade.py grades a submission
# partly by retraining the same prefix and comparing parameter digests, so the run has to be
# bit reproducible, and without this setting cuBLAS has no deterministic reduction available
# and an honest submission would fail the harness-recompute gate.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    LC_ALL=C \
    LANG=C \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    FORGE_VERIFIER_EGRESS=denied

WORKDIR /verifier

RUN pip install --no-cache-dir --break-system-packages tiktoken==0.7.0

# The GPT-2 BPE is baked in at build time, because the graded path has no egress to fetch it
# with. A verifier that had to resolve its own tokenizer at grading time would fail closed.
RUN mkdir -p /opt/tiktoken && python3 -c "import tiktoken; tiktoken.get_encoding('gpt2')"

# The grading tree. The held-out benchmark lives here and only here: tests/benchmark.json
# pins the window schedule and tests/corpus/held_out_split.json carries the FineWeb
# validation slice it cuts them from. Neither is ever assembled onto the agent surface,
# which carries task.toml, instruction.md and environment/ alone.
COPY tests/ /verifier/tests/
COPY environment/ /verifier/environment/
# The reward carriers Harbor mounts.
RUN mkdir -p /logs/verifier

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
