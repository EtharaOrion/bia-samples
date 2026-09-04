# The isolated verifier environment. Same pinned digest as the agent environment so
# the two surfaces differ only where they are declared to differ.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /verifier

# The tokenizer vocabulary is fetched once here, at build time, and warmed into the
# image. Verifier egress is denied at run time, so a tokenizer that resolved its
# vocabulary from the network during grading would fail closed rather than grade.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN pip install --no-cache-dir tiktoken \
    && mkdir -p /opt/tiktoken \
    && python -c "import tiktoken; tiktoken.get_encoding('gpt2')" \
    && chmod -R a+rX /opt/tiktoken

# The whole verifier surface, including the held-out FineWeb validation split and
# the verifier's own copy of the substrate declaration. Neither is ever assembled
# onto the agent surface.
COPY . /verifier/tests

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

# Verifier egress is denied, bound in task.toml under [verifier]. Nothing on the
# graded path reads the network or a clock, and the one random source is the fixed
# seed the frozen trainer sets, so the grading result is a function of the
# submission's corpus and of the frozen bytes alone up to accelerator accumulation
# order, which the sustain tolerance is bound wider than.
ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
