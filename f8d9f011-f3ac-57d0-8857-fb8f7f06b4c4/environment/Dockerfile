# The agent-facing environment. Pinned by digest, never by a floating tag: a tag can
# be re-pointed after this bundle is frozen, which would move the substrate under a
# result that was already recorded against it.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

WORKDIR /workspace

COPY task.toml /workspace/task.toml
COPY instruction.md /workspace/instruction.md
COPY environment/ /workspace/environment/

# The solving agent has OPEN egress inside the sandbox so it can fetch its own
# packages, datasets and models. The verifier surface denies egress. The two surfaces
# differ by design and the difference is bound in task.toml, not left to accident.
# The deny-list carried there is not a containment proof: mirrors, caches and
# paraphrased write-ups carry a recipe past any deny-list, which is exactly why the
# deterministic checker chain is the load-bearing control.

CMD ["/bin/bash"]
