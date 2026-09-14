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

# gpt.py and muon.py are NEW and they are the repair. environment/bia_loader.py
# imports `gpt` and both training entry points import `muon`, each under the
# comment "shipped in the pinned image". Measured against the pinned digest above:
#
#   python3 -c "import gpt"   -> ModuleNotFoundError: No module named 'gpt'
#   python3 -c "import muon"  -> ModuleNotFoundError: No module named 'muon'
#
# The comment was false, so no submission could reach one optimizer step. The two
# modules now ship in the bundle and are COPYed here, which is what makes them
# genuinely available in the image this bundle pins. No base image was changed and
# no package was installed to obtain them, so the pinned digest does not move.
COPY bia_harness.py bia_loader.py shape.json train_locked.py README.md \
     gpt.py muon.py bia_corpus.py operating_point.json /app/env/

# Staged at BUILD time because task.toml binds [environment] network_mode =
# "public" for provisioning and [agent] network_mode = "none" at run time.
#
# ONLY the train shard is fetched. The vendored cached_fineweb10B.py also pulls
# fineweb_val_000000.bin, and the held-out split must never reach an agent-facing
# image, so hf_hub_download is called per train file instead. One shard is the
# minimum: bia_loader.frozen_batches() cycles its file list forever.
# The frozen TRAIN shard. NOT fetched at build time any more.
#
# The former line pip-installed huggingface_hub and pulled fineweb_train_000001.bin
# into /data/fineweb/train during the build. That is a network fetch of a
# hundred-odd megabytes on every build of every image, it makes the build depend
# on an external host staying up, and it bakes a 100 MB layer into an image whose
# only use for it is to read it back.
#
# environment/bia_loader.py:train_shards() now searches an ordered list of roots
# and takes the first that carries shards matching the bound glob
# `fineweb_train_*.bin`. /data/fineweb/train stays FIRST in that list, so an image
# that does stage the shard at build time is still preferred and nothing about
# the frozen data axis changed. When it is absent the loader falls through to
# /workspace/data/fineweb10B, which is where the upstream directory is mounted.
#
# The glob is `fineweb_train_*.bin` and not `*.bin`, and that matters here: the
# upstream directory carries fineweb_val_000000.bin beside the train shards, and
# that is the shard tests/corpus/held_out_split.json is sliced out of. The old
# `*.bin` glob pointed at that directory would have trained the graded model on
# the graded split.
RUN mkdir -p /data/fineweb/train

# The VOLUME declaration that used to sit here is gone with the build-time fetch
# it protected. It existed so a later layer could not discard the staged shard;
# with nothing staged it only produced a fresh empty anonymous volume on every
# run, which is an orphan rather than a guarantee. The held-out validation split
# is still deliberately NOT copied into this image and exists only in the
# verifier image, which is why there is nothing here to tune the graded readout
# against.

RUN python3 -c "import sys, json; sys.path.insert(0, '/app/env'); import bia_loader, bia_corpus; print(json.dumps(bia_loader.executed_point(), sort_keys=True)); print(json.dumps(bia_corpus.bound_axes(), sort_keys=True))"

CMD ["/bin/bash"]
