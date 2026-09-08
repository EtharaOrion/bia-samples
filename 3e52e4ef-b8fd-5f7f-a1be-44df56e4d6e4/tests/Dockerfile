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
# OER-13 verifier environment. Isolated from the agent environment on purpose.
#
# Same pinned digest as environment/Dockerfile, because the graded quantity is
# recomputed here and a different interpreter build is a different answer. Pinned
# by digest, never by a floating tag.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The asymmetry that matters: the solving agent has OPEN egress, the verifier has
# NONE. It is declared here and in task.toml [verifier].egress, so it is a bound
# byte rather than an accident of how a runner happened to be invoked.
ENV OER13_VERIFIER_EGRESS=denied
ENV http_proxy=""
ENV https_proxy=""
ENV no_proxy="*"

# Determinism of the grading process itself. The checkers read no clock, no
# locale and no random source; these are belt on top of that. The forward pass
# that produces the graded scalar is run under inference mode with no dropout
# and a fixed contiguous evaluation order, so it carries no random source either.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0
ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C
ENV LANG=C

WORKDIR /verifier

# The frozen encoder, installed AND warmed at build time because run-time egress
# is denied and a cold encoder would reach for the network.
# The verifier needs it to execute the delivered baseline and the reference
# composition, both of which it runs itself.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN pip install --no-cache-dir --break-system-packages "tiktoken==0.13.0" \
 && mkdir -p /opt/tiktoken_cache \
 && python -c "import tiktoken; tiktoken.get_encoding('gpt2')" \
 && chmod -R a+rX /opt/tiktoken_cache

# The build context is the bundle root, so tests/, solution/ and environment/
# all land here: tests/grade.py imports the private reference from solution/ to
# measure the in-run floor composition, and tests/harness.py reads the pristine
# environment/ tree both to measure the in-run baseline composition and to read
# the frozen substrate declaration the architecture is pinned by. All of those
# are verifier-side reads of bundle bytes.
#
# The SUBMISSION never travels into this image. It is copied alone into a fresh
# temporary directory by tests/runner.py and launched out of process as a new
# session leader under a small environment allowlist, and tests/grade.py never
# imports it.
# The destination depth is load bearing. `COPY tests/ /verifier/` lands test.sh
# at /verifier/test.sh, one level ABOVE the ENTRYPOINT path at the foot of this
# file, so the container starts and dies immediately on
# `/bin/bash: /verifier/tests/test.sh: No such file or directory` with rc 127.
# Nothing runs, so the EXIT trap in tests/test.sh -- which is the only thing that
# ever writes a reward -- is never even registered and the slot emits no reward
# carrier at all rather than an attributed zero. The tree also has to land under
# tests/ for tests/test.sh:19 and tests/grade.py:32, which both resolve BUNDLE as
# the PARENT of the directory they sit in and then read environment/ and
# solution/ beside it.
COPY tests/ /verifier/tests/
COPY environment/ /verifier/environment/

# solution/ is deliberately NOT copied.
#
# It used to be, because tests/grade.py imported solution/reference.py to materialise
# the FLOOR of the reward scale. That made the floor and the oracle the same
# composition, so the reference arm of a differentiation gate was the high anchor being
# graded against itself -- measured, once the decoder initialisation was seeded, as a
# graded loss equal to the floor to seven decimal places and a reward of 0.9999995.
#
# The floor now comes from tests/private/reference_composition.py, which arrives with
# the COPY above and exists only in this image. Keeping solution/ out is what makes
# that structural rather than a convention: the oracle's bytes are not in the process
# that grades submissions, so no import can reach them.
# The held-out split, STAGED AT BUILD TIME into the exact root tests/bound.json
# pins. It was previously only DECLARED, as `VOLUME ["/verifier/holdout/fineweb10B"]`
# with nothing ever put there. A VOLUME line stages no bytes; it makes Docker
# create an EMPTY anonymous volume at that path at every container start, so the
# root always resolved and always resolved to nothing. tests/harness.py:340 calls
# nanogpt.holdout_shards on it, gets an empty list, and the first evaluation dies
# in tests/frozen/nanogpt.py:427 with "the held-out split carries no full
# evaluation window" -- inside harness.build, before the submission and the
# reference floor are ever compared. Build egress is open and run-time egress is
# denied, so the shards are fetched here exactly as the agent image fetches its
# own train glob.
#
# This is still the verifier's own split and no weaker for being baked in: the
# root is resolved from tests/bound.json alone, environment/Dockerfile stages no
# validation shard, and nothing on the agent surface reaches this path. Baking it
# in removes the shadowing, not the ownership.
RUN pip install --no-cache-dir --break-system-packages huggingface_hub \
 && python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename='fineweb_val_000000.bin', \
repo_type='dataset', local_dir='/verifier/holdout/fineweb10B')" \
 && test -s /verifier/holdout/fineweb10B/fineweb_val_000000.bin
LABEL oer13.holdout_root="/verifier/holdout/fineweb10B"
LABEL oer13.holdout_glob="fineweb_val_*.bin"

# Harbor mounts the reward root. The entry point terminates by writing the bare
# float to reward.txt with the reason and metric block in score.json, and an EXIT
# trap guarantees a reward on every exit path including failure.
#
# The root is CREATED below and deliberately NOT declared as a VOLUME. A
# `VOLUME ["/logs/verifier"]` makes Docker create an anonymous volume at that exact
# path whenever the runtime mounts the PARENT /logs rather than that subdirectory,
# and the anonymous volume is layered ON TOP of the parent bind mount. The EXIT
# trap in tests/test.sh:36 then writes reward.txt into the anonymous volume, the
# write succeeds, the container exits -- and the host side of /logs holds an empty
# verifier/ directory. The guarantee that every exit path writes a reward held; the
# reward simply never left the container, and the run read as an infrastructure
# fault instead of as the score the grader computed.
LABEL oer13.reward_path="/logs/verifier/reward.txt"
LABEL oer13.score_document_path="/logs/verifier/score.json"
LABEL oer13.egress="denied"


RUN mkdir -p /logs/verifier /logs/agent
ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
