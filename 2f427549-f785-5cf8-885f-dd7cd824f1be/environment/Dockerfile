# The agent environment. Pinned by digest exactly as the verifier environment is; a
# floating tag would let the substrate the agent solves against move under a frozen
# bundle.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Solver egress is OPEN inside the sandbox so the agent can fetch its own packages,
# datasets and models. The deny-list bound in task.toml is not a containment proof:
# mirrors, caches, paraphrased write-ups and model weights all carry a recipe past any
# deny-list. The deterministic near-duplicate screen in tests/checkers.py is the
# load-bearing control, not the network policy.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    LC_ALL=C \
    LANG=C

WORKDIR /workspace
COPY trainer.py /workspace/trainer.py
COPY dev_examples.jsonl /workspace/dev_examples.jsonl
COPY GENERATOR_CONTRACT.md /workspace/GENERATOR_CONTRACT.md
RUN mkdir -p /workspace/submission

# The held-out benchmark is NOT here and is not reachable from here. It lives under
# tests/ only, and Harbor assembles the non-oracle agent surface from task.toml,
# instruction.md and environment/ alone.
CMD ["/bin/bash"]
