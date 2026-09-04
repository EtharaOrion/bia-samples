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

# tiktoken carries the GPT-2 byte-pair codec the harness uses to decode the
# FineWeb10B shards back to the bytes a tokenizer construction works over.
# huggingface_hub pulls the shards themselves. Both are pinned.
RUN pip install --no-cache-dir tiktoken==0.7.0 huggingface_hub==0.24.6

COPY harness.py /workspace/environment/harness.py
COPY manifest.json /workspace/environment/manifest.json
COPY nanogpt_substrate.json /workspace/environment/nanogpt_substrate.json
COPY default_tokenizer.py /workspace/environment/default_tokenizer.py
COPY build_corpus.py /workspace/environment/build_corpus.py

# The FineWeb10B TRAINING shards, staged at build time into the path the frozen
# substrate names. The VALIDATION shards are not staged here and never are: the
# held-out slice that produces the grade is the verifier's, and this surface is
# the solver's. environment/corpus/ is not copied at all, because the generated
# stand-in corpus it used to carry is retired.
RUN python3 /workspace/environment/build_corpus.py --shards 2

# The submission directory is the only place the agent writes that the graded path
# reads. Everything under /workspace/environment is a frozen axis.
RUN mkdir -p /workspace/submission

CMD ["/bin/bash"]
