# Agent environment. Derived from harness/Dockerfile in the pinned harness submodule.
# The published record corpus at harness/records/ is deliberately NOT copied here:
# it holds complete runnable recipes for every accepted result and is the primary
# contamination carrier for this task family.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12.7
ENV PATH=/usr/local/bin:$PATH

RUN apt update && apt install -y --no-install-recommends build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

# git is deliberately absent. Sanitization HISTORY ABSENCE and REMOTE ABSENCE reject
# version-control history, remotes and credential helpers anywhere in the closure.

RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && ./configure --enable-optimizations && \
    make -j$(nproc) && make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

# Torch is pinned to 2.5.1+cu124, which is what record 46 ran on every seed and
# therefore what the oracle must reach 2690 steps under. The CUDA base is 12.4 to
# match. SOAPH's 2.6.0+cu124 is the timing anchor only; where the anchors disagree
# the pass/fail anchor governs, because a timing anchor cannot decide correctness.
# Ubuntu is 22.04 rather than 24.04 because no CUDA 12.4 image ships on 24.04.
COPY requirements.txt /tmp/requirements.txt
RUN python3.12 -m pip install --no-cache-dir -r /tmp/requirements.txt

# The agent-visible surface. The verifier tree is NOT copied here: a submission
# that could import the grading code could read the anchors it is scored against.
WORKDIR /app
COPY frozen_gpt.py /app/frozen_gpt.py
COPY bia_data.py /app/bia_data.py
COPY bia_loader.py /app/bia_loader.py
COPY train_baseline.py /app/train_baseline.py
COPY README.md /app/README.md
COPY shape.json /env/shape.json
ENV BIA_SHAPE=/env/shape.json
ENV BIA_STEP_GRID=/env/step_grid.json
ENV BIA_TRAIN_SHARDS=/data/train_*.bin
ENV BIA_CHECKPOINT_DIR=/submission/checkpoints
RUN mkdir -p /submission/checkpoints /env
CMD ["bash"]
