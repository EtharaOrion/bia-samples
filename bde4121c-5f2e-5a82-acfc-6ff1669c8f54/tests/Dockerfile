# Verifier environment. Separate from the agent environment because grading
# re-executes the submitted recipe rather than parsing an agent-written log,
# so the verifier needs its own GPU-capable image whose measurement the
# submission cannot influence beyond the submitted script itself.
#
# CUDA 12.4 to match torch 2.5.1+cu124, which is what record 46 ran and
# therefore what the oracle must reach 2690 steps under.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12.7
ENV PATH=/usr/local/bin:$PATH

# Egress is denied for this environment in task.toml. Nothing here fetches at
# run time; every dependency is baked at build time so a graded run cannot
# reach the network at all.
RUN apt update && apt install -y --no-install-recommends build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

# git is absent by construction. Sanitization HISTORY ABSENCE and REMOTE ABSENCE
# reject version-control history, remotes and credential helpers anywhere in the
# agent-facing closure, and the verifier image is held to the same standard so a
# graded run cannot reach a repository through the grader either.

RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && ./configure --enable-optimizations && \
    make -j$(nproc) && make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

COPY requirements-verifier.txt /tmp/requirements-verifier.txt
RUN python3.12 -m pip install --no-cache-dir -r /tmp/requirements-verifier.txt

# Determinism. These reduce version and topology drift. They do not make
# training bitwise reproducible: the measured same-seed spread on this workload
# is 0.00076, which is what remains after every pin is honoured. Grading
# therefore requires verdict stability and never score equality.
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTHONHASHSEED=0
ENV NVIDIA_TF32_OVERRIDE=0

WORKDIR /task
