# Agent environment for bia slot S06, gradient estimator under injected noise.
#
# Pinned by digest, never by floating tag. This is the linux/amd64 digest of
# pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime as recorded in the verified reference
# bundle this task's shape was drawn from.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BIA_BUNDLE=/workspace \
    BIA_FIXTURES=/workspace/environment/fixtures \
    BIA_LOG_DIR=/logs/verifier \
    BIA_SUBMISSION=/workspace/submission/estimator.py

# Setup only egress. Nothing below runs during scored work.
RUN python3 -m pip install --no-cache-dir "numpy>=1.26,<3" pytest \
 && python3 -c "import numpy, torch; print(numpy.__version__, torch.__version__)"

WORKDIR /workspace

# The frozen substrate travels with the image. The injected gradient noise process is a
# frozen fixture with a fixed seed and it ships here so the same noise realization is
# reproducible across attempts and across graders. An unfrozen noise process would make
# steps to target incomparable between two runs and the reward unmeasurable.
COPY harness /workspace/environment/harness
COPY fixtures /workspace/environment/fixtures
COPY starter /workspace/environment/starter

RUN mkdir -p /workspace/submission /logs/verifier \
 && cp /workspace/environment/starter/estimator.py /workspace/submission/estimator.py

# Fail the build rather than ship a substrate that does not match its bound digests.
RUN python3 - <<'PY'
import hashlib
pairs = {
    "/workspace/environment/fixtures/noise_process.py":
        "78477bbfa8b6a818dee25338484fb4a078735c1a11faa74f9175b5f8707e5c88",
    "/workspace/environment/fixtures/corpus.bin":
        "23abc203c5716cea52623e903f24b41a94081e863fa3be44c10151568d199a73",
}
for path, want in pairs.items():
    with open(path, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    assert got == want, (path, got, want)
print("fixture digests verified")
PY

CMD ["/bin/bash"]
