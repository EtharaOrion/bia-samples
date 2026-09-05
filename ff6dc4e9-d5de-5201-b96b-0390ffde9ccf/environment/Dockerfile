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

# The corpus. The pinned loader stages the FineWeb10B TRAIN shards only, at the paths
# environment/nanogpt_substrate.json declares. The VALIDATION shards are deliberately
# absent from this image: the number that scores a submission is computed by the
# verifier on a held-out slice of that validation split, from a container this one
# cannot reach. A build that staged fineweb_val_*.bin here would put the graded
# surface on the solver surface, which is the one thing this stage exists to prevent.
RUN python3 /workspace/environment/bootstrap.py corpus \
      --root /workspace/data/fineweb10B \
      --split train \
      --shards 8

# The model. A real parameter snapshot of the frozen 12-layer 768-dim decoder is built
# here, once, at image build, from the declaration and the pinned train shards under a
# pinned seed. It lands at /workspace/checkpoint/nanogpt.pt and it is the artifact this
# task quantizes. The verifier holds its own pristine copy of the identical checkpoint
# and grades from that copy, never from this one.
RUN python3 /workspace/environment/bootstrap.py checkpoint \
      --declaration /workspace/environment/nanogpt_substrate.json \
      --shards /workspace/data/fineweb10B \
      --steps 3250 \
      --seed 1337 \
      --out /workspace/checkpoint/nanogpt.pt

# The two content digests the graded path depends on are established HERE, in built
# state, and on no agent-visible byte of this bundle: the checkpoint's own sha256 and
# the sha256 over the raw staged bytes of the calibration slice in force. A run reads
# them back out of this state through environment/model.py and
# environment/calibration_probe.py. A run that never read them cannot reproduce them.

# The solving agent has OPEN egress inside the sandbox so it can fetch its own
# packages. The verifier surface denies egress. The two surfaces differ by design and
# the difference is bound in task.toml, not left to accident. The deny-list carried
# there is not a containment proof: mirrors, caches and paraphrased write-ups carry a
# recipe past any deny-list, which is exactly why the deterministic checker chain is
# the load-bearing control.

CMD ["/bin/bash"]
