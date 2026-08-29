# Agent environment for slot OER-03.
#
# Pinned by digest, never by a floating tag. Harbor accepts any string and does
# not itself require a digest, so this requirement is the batch contract's and is
# enforced here. A tag without a digest means the scanned image and the executed
# image can differ, which makes every reproduction claim about this bundle
# unverifiable.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The solving surface carries OPEN egress, bound in task.toml under
# [agent] solver_egress = "open". The deny-list narrows a convenience path and is
# explicitly NOT a containment proof: mirrors, caches, paraphrased write-ups and
# model weights all carry a recipe past any deny-list. The deterministic checker
# in tests/ is the load-bearing control.
ENV BIA_SLOT=OER-03 \
    BIA_FAMILY=F1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /app

COPY bia_harness.py bia_loader.py shape.json train_locked.py README.md /app/env/

# The frozen training shard. The held-out validation split is deliberately NOT
# copied into this image; it exists only in the verifier image, which is why
# there is nothing here to tune the graded readout against.
VOLUME ["/data/fineweb/train"]

RUN python3 -c "import json, pathlib; json.loads(pathlib.Path('/app/env/shape.json').read_text())"

CMD ["/bin/bash"]
