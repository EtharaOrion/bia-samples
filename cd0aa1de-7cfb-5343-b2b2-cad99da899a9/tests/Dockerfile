# Isolated VERIFIER environment for slot OER-14.
#
# Pinned by digest, same digest as the agent environment, because a verifier grading on a
# different stack than the one the agent ran on is measuring two things at once.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# EGRESS IS DENIED on this surface. It is bound in task.toml under [verifier] egress. The
# agent surface is open by design and the two surfaces differ on purpose. Nothing in the
# grading path fetches anything, so a denied egress costs the verifier nothing and removes
# the only path by which a graded number could arrive from outside this image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    HTTP_PROXY=http://127.0.0.1:9 \
    HTTPS_PROXY=http://127.0.0.1:9 \
    NO_PROXY="" \
    OER_SLOT=OER-14

WORKDIR /verifier

# The verifier owns the substrate and the corpus, so every number on the graded path is
# produced inside this image from bytes this image carries.
COPY environment/substrate.py /verifier/environment/substrate.py
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
