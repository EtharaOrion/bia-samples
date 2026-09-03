# Agent environment for slot OER-21.
# Pinned by digest, never by a floating tag.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /work

# The task substrate the agent works on. The handed quantization toolkit, the frozen
# model, the frozen evaluation corpus, the unquantized reference readings, and the
# published scheme schema that names every free axis.
COPY model /work/environment/model
COPY corpus /work/environment/corpus
COPY quantize.py /work/environment/quantize.py
COPY scheme_schema.json /work/environment/scheme_schema.json

ENV PYTHONDONTWRITEBYTECODE=1
ENV OUT_DIR=/work/out

CMD ["/bin/bash"]
