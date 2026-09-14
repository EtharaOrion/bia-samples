# Verifier environment for slot OER-03. Isolated from the agent environment.
#
# Same pinned digest as environment/Dockerfile, and pinned for the same reason: a
# scanned image that differs from the executed image makes the measurement
# unverifiable. The two environments differ in exactly two respects, and both are
# deliberate.
#
#   1. EGRESS IS DENIED HERE. The agent surface carries open egress so the
#      solving model can fetch its own packages, datasets and models. The
#      measurement the reward rests on must not be reachable over a network, so
#      this surface has none. Bound in task.toml under [verifier] egress.
#
#   2. THE HELD-OUT SPLIT LIVES HERE AND ONLY HERE. The graded loss is computed
#      by this environment on a split the agent container never receives, from
#      the parameter snapshots the pinned harness wrote. That is what makes the
#      graded readout unshapeable from inside the run.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV BIA_SLOT=OER-03 \
    BIA_ROLE=verifier \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    http_proxy="" \
    https_proxy="" \
    no_proxy="*"

WORKDIR /verifier

# evaluator.py and corpus/held_out_split.json are NEW and they are the repair.
# tests/grade.py:_held_out_loss has always done `import evaluator`, and measured
# against the pinned digest above:
#
#   python3 -c "import evaluator"  ->  ModuleNotFoundError: No module named 'evaluator'
#
# _held_out_loss catches that ImportError and returns None, evaluate_ledger drops
# every point that returns None, and the evaluation series therefore came out
# empty on every run whatever the run did, which is the
# `run-produced-no-verifier-state` zero. No submission could be graded on its
# merits. The module now ships in the bundle and is COPYed here, which is what
# makes it genuinely importable in the image this bundle pins. No base image was
# changed and no package was installed to obtain it, so the pinned digest does not
# move.
COPY test.sh checkers.py checkers.yaml grade.py runner.py rubrics.jsonl test_output.py golden.json \
     evaluator.py /verifier/

# The held-out split: 1048576 GPT-2 BPE token ids copied without transformation
# out of the upstream FineWeb10B shard fineweb_val_000000.bin at token offset
# 800000, carried with the slice digest evaluator.py verifies before it evaluates
# anything. COPYed into this image and into no other: environment/Dockerfile
# copies environment/bia_harness.py, bia_loader.py, shape.json, train_locked.py,
# README.md, gpt.py and muon.py and does not copy environment/corpus/, so the
# agent image receives no validation split.
COPY corpus/held_out_split.json /verifier/corpus/held_out_split.json

# The verifier RE-RUNS the submission out of process (tests/runner.py), and both
# the submission and evaluator.py import the pinned harness, the loader, the
# architecture and the optimizer from /app/env. task.toml binds [verifier]
# environment_mode = "isolated" and Harbor mounts only /app/submission.py into
# this image, so this image carries its own byte-identical copy of those runtime
# modules. They are listed by name rather than as a directory so the projection is
# exact.
COPY agent_env/bia_harness.py agent_env/bia_loader.py agent_env/shape.json \
     agent_env/gpt.py agent_env/muon.py /app/env/

RUN pip install --no-cache-dir --break-system-packages pytest==8.3.4 PyYAML==6.0.2

# The frozen TRAIN shard, staged at BUILD time for the same reason and by the
# same call environment/Dockerfile stages it: task.toml binds [environment]
# network_mode = "public" for provisioning while the verifier keeps egress denied
# at RUN time, and bia_loader.frozen_batches() reads /data/fineweb/train. Only the
# train shard is fetched here. fineweb_val_000000.bin is never downloaded into any
# image; the held-out slice above is the only validation data in this bundle.
RUN pip install --no-cache-dir --break-system-packages huggingface_hub \
 && python3 -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('/data/fineweb/train', exist_ok=True); hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename='fineweb_train_000001.bin', repo_type='dataset', local_dir='/data/fineweb/train')" \
 && rm -f /data/fineweb/train/fineweb_val_*.bin \
 && ls -l /data/fineweb/train/fineweb_train_000001.bin

# The build-time proof that the repair holds in the image rather than only in the
# tree. This imports the module grade.py imports, decodes the split, verifies its
# frozen slice digest and reports the shape of the evaluation. It computes no
# loss, so it needs no accelerator and no state file, and it fails the build if
# the module grade.py needs is absent or the split is not the split it declares.
RUN python3 -c "import sys; sys.path.insert(0, '/verifier'); import evaluator; import json; print(json.dumps(evaluator.describe(), sort_keys=True))"

# The reward directory Harbor mounts. Created here so an aborted verifier can
# still write its attributed zero rather than fail on a missing directory, which
# would turn a graded failure back into silence.
RUN mkdir -p /logs/verifier

RUN chmod +x /verifier/test.sh

ENTRYPOINT ["/verifier/test.sh"]
