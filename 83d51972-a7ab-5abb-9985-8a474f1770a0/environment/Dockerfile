# The agent-facing task environment.
#
# Pinned by digest, never by a floating tag. Harbor accepts any string and does not itself
# require a digest, so the digest requirement is this contract's and this file enforces it.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The near-duplicate substrate is deterministic arithmetic over a frozen corpus. It needs no
# accelerator runtime and no third-party package: everything it uses is in the standard
# library. That is deliberate, because a reproducible grading path cannot rest on a wheel
# resolving the same way twice. Nothing is installed here, so nothing here needs
# --break-system-packages, and the graded path never depends on network egress.
#
# The solving agent has open internet egress inside the sandbox and may install whatever it
# wants for its own search. Nothing it installs reaches the graded path.

WORKDIR /workspace

COPY nd_index.py /task/environment/nd_index.py
COPY build_index.py /task/environment/build_index.py
COPY index_query.py /task/environment/index_query.py
COPY corpus.jsonl /task/environment/corpus.jsonl
COPY index_spec.json /task/environment/index_spec.json

ENV PYTHONPATH=/task/environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV OER27_INDEX=/task/index

# THE INDEX IS BUILT HERE, AT IMAGE BUILD TIME.
#
# Everything the index establishes comes into existence on this line and on no earlier one.
# Neither the adjudication threshold nor the collision structure is written into any file
# this bundle ships; both are computed from the frozen corpus by the builder above and land
# in /task/index. The statement the agent reads names neither.
RUN python3 /task/environment/build_index.py \
        --corpus /task/environment/corpus.jsonl \
        --spec /task/environment/index_spec.json \
        --out /task/index \
 && python3 -c "import json,sys; d=json.load(open('/task/index/calibration.json')); sys.exit(0 if d['adjudication_threshold']>0 else 1)"

RUN mkdir -p /workspace/notes /workspace/logs

CMD ["/bin/bash"]
