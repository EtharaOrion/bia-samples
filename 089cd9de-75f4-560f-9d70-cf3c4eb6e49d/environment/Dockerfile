# The agent surface. Pinned by digest, never by a floating tag: a tag is a mutable
# pointer, and a bundle whose image can move under it is not frozen.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Solver egress is OPEN at this surface so the agent can fetch its own packages,
# datasets and models. The verifier surface in tests/Dockerfile is DENIED. The two
# surfaces differ by design and the divergence is recorded in
# solution/grounding.yaml under solver_egress_divergence.
ENV OER15_SOLVER_EGRESS=open
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /workspace

COPY harness.py /workspace/environment/harness.py
COPY manifest.json /workspace/environment/manifest.json
COPY default_tokenizer.py /workspace/environment/default_tokenizer.py
COPY build_corpus.py /workspace/environment/build_corpus.py
COPY corpus /workspace/environment/corpus

# The submission directory is the only place the agent writes that the graded path
# reads. Everything under /workspace/environment is a frozen axis.
RUN mkdir -p /workspace/submission

CMD ["/bin/bash"]
