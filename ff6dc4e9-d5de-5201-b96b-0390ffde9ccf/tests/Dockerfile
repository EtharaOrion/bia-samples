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

# The held-out split. This image, and ONLY this image, stages the FineWeb10B
# VALIDATION shards. environment/Dockerfile refuses to stage them, so the two surfaces
# differ by construction rather than by convention. The paths below are the ones
# tests/bound.json pins, and the grading path resolves the split from that file alone.
RUN python3 /opt/task/environment/bootstrap.py corpus \
      --root /verifier/data/fineweb10B \
      --split train \
      --shards 8
COPY --from=fineweb-holdout /data/fineweb10B/fineweb_val_*.bin /verifier/data/fineweb10B/

# The verifier's own pristine checkpoint, built by the identical bootstrap invocation
# under the identical pinned seed and step count as the agent image. The verifier
# quantizes THIS copy under the submitted allocation and never loads a parameter file
# a submission wrote.
RUN python3 /opt/task/environment/bootstrap.py checkpoint \
      --declaration /opt/task/environment/nanogpt_substrate.json \
      --shards /verifier/data/fineweb10B \
      --steps 3250 \
      --seed 1337 \
      --out /verifier/checkpoint/nanogpt.pt

# Verifier egress is DENIED. That is bound in task.toml under [verifier] and is not
# established by this file: a Dockerfile states an intent and a runtime enforces it.
# Nothing in the grading path opens a socket, so a denied egress removes a capability
# the verifier never exercised rather than closing a hole it depended on.

ENTRYPOINT ["/usr/bin/env", "bash", "/opt/task/tests/test.sh"]
