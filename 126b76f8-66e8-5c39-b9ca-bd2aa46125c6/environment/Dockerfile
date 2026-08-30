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

# gpt.py, muon.py, bia_corpus.py and operating_point.json are NEW and they are
# the repair. environment/bia_loader.py imported `gpt` and the two training
# scripts import `muon`, both under the comment "shipped in the pinned image".
# Measured against the pinned digest below:
#
#   python3 -c "import gpt"   -> ModuleNotFoundError: No module named 'gpt'
#   python3 -c "import muon"  -> ModuleNotFoundError: No module named 'muon'
#
# The comment was false, so no submission could reach one optimizer step. The
# modules now ship in the bundle and are COPYed here, which is what makes them
# genuinely available in the image this bundle pins. No base image was changed
# and no package was installed to obtain them, so the pinned digest does not move.
COPY bia_harness.py bia_loader.py shape.json train_locked.py README.md \
     gpt.py muon.py bia_corpus.py operating_point.json /app/env/

# The frozen training shard. The pinned image carries no FineWeb byte and this
# bundle ships none, so the shard the loader reads is materialised here from the
# bound seeds in operating_point.json, deterministically and without a network.
# It is baked BEFORE the VOLUME line so the anonymous volume Docker creates at
# that path is seeded from it; environment/bia_corpus.py regenerates the same
# bytes on demand where it is not.
RUN python3 /app/env/bia_corpus.py

# The held-out validation split is deliberately NOT created in this image; it
# exists only in the verifier image, which is why there is nothing here to tune
# the graded readout against.
VOLUME ["/data/fineweb/train"]

RUN python3 -c "import json, pathlib; json.loads(pathlib.Path('/app/env/shape.json').read_text())"

CMD ["/bin/bash"]
