# The agent environment. Pinned by digest, never by a floating tag: a tag can be
# repointed after this bundle is frozen, and a bundle whose base image can move is
# a bundle whose results cannot be reproduced.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /task

# The tokenizer the FineWeb corpus is built with. Its vocabulary is fetched once at
# build time and warmed into the image, because the verifier that runs the mirror
# of this path has egress denied and a tokenizer reaching for the network at
# grading time would fail closed rather than grade.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN pip install --no-cache-dir tiktoken \
    && mkdir -p /opt/tiktoken \
    && python -c "import tiktoken; tiktoken.get_encoding('gpt2')" \
    && chmod -R a+rX /opt/tiktoken

# The frozen half of the task and the tools the agent measures with. Copied
# read-only; the agent-visible lower layer grants no write bit anywhere. The
# substrate declaration travels beside them because trainer.py reads the
# architecture out of it rather than carrying it as a literal.
COPY frozen /task/environment/frozen
COPY tools /task/environment/tools
COPY nanogpt_substrate.json /task/environment/nanogpt_substrate.json

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/task/environment/tools

# The solving agent has open egress inside the sandbox. That is bound in task.toml
# under [agent] and is deliberately asymmetric with the verifier, whose egress is
# denied. The deny-list carried alongside it is not a containment proof.
CMD ["/bin/bash"]
