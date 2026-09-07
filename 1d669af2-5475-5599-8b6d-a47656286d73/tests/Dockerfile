# The isolated verifier environment. Same digest pin as the agent surface, so the
# grading process and the substrate cannot drift apart across a re-pull.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OER22_VERIFIER_EGRESS=denied

WORKDIR /opt/task

COPY task.toml /opt/task/task.toml
COPY instruction.md /opt/task/instruction.md
COPY environment/ /opt/task/environment/
COPY tests/ /opt/task/tests/
COPY solution/ /opt/task/solution/

# The pinned upstream loader, placed as the parent of this image's corpus root so the
# loader's own destination rule and the --root below resolve to one directory, exactly
# as they do on the agent surface. environment/bootstrap.py names this file, refuses to
# reimplement it and refuses to run without it, so it is a build input on both surfaces.
COPY environment/data/cached_fineweb10B.py /verifier/data/cached_fineweb10B.py

# The held-out split. This image, and ONLY this image, stages the FineWeb10B
# VALIDATION shards. environment/Dockerfile refuses to stage them, so the two surfaces
# differ by construction rather than by convention. The paths below are the ones
# tests/bound.json pins, and the grading path resolves the split from that file alone.
#
# Everything the graded path reads is staged HERE, at build time, and baked in. Nothing
# on the grading path fetches anything, which is what makes the denied run-time egress
# below cost the verifier nothing.
RUN python3 /opt/task/environment/bootstrap.py corpus \
      --root /verifier/data/fineweb10B \
      --split train \
      --shards 8 \
      --loader /verifier/data/cached_fineweb10B.py \
 && rm -rf /verifier/data/fineweb10B/.cache /root/.cache/huggingface

# THE VALIDATION SPLIT, STAGED ON THIS SURFACE AND ON NO OTHER. The delivered line here
# was `COPY --from=fineweb-holdout`, naming a build stage no FROM in this file ever
# declared, so BuildKit read `fineweb-holdout` as an external image, tried to pull
# docker.io/library/fineweb-holdout:latest and was denied. Nothing was ever going to
# resolve that name: no such image exists and no stage produced it.
#
# What replaces it fetches the real shard from the upstream dataset the pinned loader
# names, `kjj0/fineweb10B-gpt2`, which is the same source the train shards above come
# from, and asserts its content digest before the build may continue. The digest below
# is the sha256 of the project's own real source shard at
# harness/shared/data/fineweb10B/fineweb_val_000000.bin, so a build that lands anything
# other than those exact 200,001,024 bytes fails here instead of grading on them. The
# loader reaches the validation split only when handed --val, and nothing on the agent
# surface can hand it that.
RUN python3 /verifier/data/cached_fineweb10B.py --val \
 && python3 -c "\
import hashlib,sys; \
p='/verifier/data/fineweb10B/fineweb_val_000000.bin'; \
h=hashlib.sha256(open(p,'rb').read()).hexdigest(); \
expected='5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640'; \
sys.exit(0) if h==expected else sys.exit('held-out shard digest '+h+' is not the pinned '+expected)" \
 && rm -rf /verifier/data/fineweb10B/.cache /root/.cache/huggingface

# The verifier's own pristine checkpoint is built by the identical bootstrap invocation
# under the identical pinned seed and step count as the agent surface, and the verifier
# quantizes THAT copy under the submitted allocation and never loads a parameter file a
# submission wrote. It is built by tests/test.sh on entry rather than into a layer here,
# because an image build has no accelerator: BuildKit runs its steps outside the
# daemon's default runtime, so the delivered `--device cuda` step died on `Found no
# NVIDIA driver on your system`, and the same training on this host's CPU measures 564s
# per optimizer step against a 1800s build bound. Building it in test.sh keeps it
# verifier-owned and unreachable from the agent surface, which is the property that
# matters, and moves only where the accelerator is held.
RUN mkdir -p /verifier/checkpoint

# The bound reward carrier root, so tests/test.sh writes into a directory that exists
# even when nothing is mounted over it.
RUN mkdir -p /logs/verifier

# Verifier egress is DENIED. That is bound in task.toml under [verifier] and is not
# established by this file: a Dockerfile states an intent and a runtime enforces it.
# Nothing in the grading path opens a socket, so a denied egress removes a capability
# the verifier never exercised rather than closing a hole it depended on.

ENTRYPOINT ["/usr/bin/env", "bash", "/opt/task/tests/test.sh"]
