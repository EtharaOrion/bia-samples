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
     gpt.py muon.py /app/env/

# Staged at BUILD time because task.toml binds [environment] network_mode =
# "public" for provisioning and [agent] network_mode = "none" at run time.
#
# ONLY the train shard is fetched. The vendored cached_fineweb10B.py also pulls
# fineweb_val_000000.bin, and the held-out split must never reach an agent-facing
# image, so hf_hub_download is called per train file instead. One shard is the
# minimum: bia_loader.frozen_batches() cycles its file list forever.
RUN pip install --no-cache-dir --break-system-packages huggingface_hub \
 && python3 -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('/data/fineweb/train', exist_ok=True); hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename='fineweb_train_000001.bin', repo_type='dataset', local_dir='/data/fineweb/train')" \
 && rm -f /data/fineweb/train/fineweb_val_*.bin \
 && ls -l /data/fineweb/train/fineweb_train_000001.bin

# Declared AFTER the shard is in place: a VOLUME discards writes made by later
# layers, so this ordering is what keeps the shard in the image. The held-out
# validation split is deliberately NOT copied into this image; it exists only in
# the verifier image, which is why there is nothing here to tune the graded
# readout against.
VOLUME ["/data/fineweb/train"]

RUN python3 -c "import json, pathlib; json.loads(pathlib.Path('/app/env/shape.json').read_text())"

CMD ["/bin/bash"]
