# Base pinned by digest, not by tag. Harbor accepts any string for docker_image and
# does not itself require a digest, so a floating tag would validate cleanly and
# silently; invariant 21 makes that BLOCK:INVALID_TASK.
#
# Deliberately NOT inherited from harness/Dockerfile: its final layer runs
# `pip install --pre torch --index-url .../nightly/cu126 --upgrade`, which resolves a
# different binary on every build day and contradicts its own torch==2.10 pin. A
# graded environment that changes under you is not an environment.
FROM nvidia/cuda@sha256:4981f502afcad93d102cf9c4b5adc0ad45d30fb89a1da959c4f4c7bbb12c5c54

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv ca-certificates git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Exact pins. No nightly, no floating index.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
      "torch==2.10" "numpy" "tqdm"

WORKDIR /app
COPY bia_loader.py /app/bia_loader.py
CMD ["bash"]
