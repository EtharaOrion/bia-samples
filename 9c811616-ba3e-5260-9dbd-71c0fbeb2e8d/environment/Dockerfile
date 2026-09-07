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

# The pinned upstream loader, at the path environment/nanogpt_substrate.json names it
# under corpus.loader and environment/bootstrap.py defaults to. bootstrap.py refuses to
# reimplement it and refuses to run without it, so it is a build input and it is placed
# here rather than left to be discovered. Its own directory is the parent of
# corpus.container_train_path, which is what makes the loader's destination rule and the
# --root below resolve to one directory.
#
# IT IS PLACED FROM THE PAYLOAD THE LINE ABOVE ALREADY DELIVERED, not by a second COPY
# of the same byte, and that is load bearing. This file is the only statement this bundle
# makes about where its environment/ payload lands in the image, and that destination is
# read back by folding the destinations of every environment/-sourced COPY to their
# common ancestor. A COPY of environment/data/... to /workspace/data puts /workspace/data
# into that set beside /workspace/environment, the ancestor collapses to /workspace, and
# the bundle root a run resolves is then one directory ABOVE the payload, where
# environment/substrate.json is not what solution/solve.sh reads. So no environment/-
# sourced COPY here lands outside /workspace/environment, and the loader reaches its own
# runtime path by an in-image copy. The byte at /workspace/data/cached_fineweb10B.py is
# unchanged: same file, same mode, from the tree the COPY above installed.
RUN mkdir -p /workspace/data \
 && cp -p /workspace/environment/data/cached_fineweb10B.py \
          /workspace/data/cached_fineweb10B.py

# The corpus. The pinned loader stages the FineWeb10B TRAIN shards only, at the paths
# environment/nanogpt_substrate.json declares. The VALIDATION shards are deliberately
# absent from this image: the number that scores a submission is computed by the
# verifier on a held-out slice of that validation split, from a container this one
# cannot reach. A build that staged fineweb_val_*.bin here would put the graded
# surface on the solver surface, which is the one thing this stage exists to prevent.
# bootstrap.py stage_corpus passes the loader a shard COUNT and can express no flag, and
# the loader fetches the validation shard only when handed --val, so this stage has no
# route to that split rather than a filter over it.
RUN python3 /workspace/environment/bootstrap.py corpus \
      --root /workspace/data/fineweb10B \
      --split train \
      --shards 8 \
 && rm -rf /workspace/data/fineweb10B/.cache /root/.cache/huggingface

# The held-out split is absent from this image and this line proves it rather than
# asserting it. The build fails if any fineweb_val_*.bin reached this filesystem.
RUN test -z "$(find / -name 'fineweb_val_*.bin' -print -quit)" \
    || (echo "the validation split reached the agent surface" && false)

# The model. A real parameter snapshot of the frozen 12-layer 768-dim decoder, built
# from the declaration and the pinned train shards under a pinned seed, landing at
# /workspace/checkpoint/nanogpt.pt. It is the artifact this task quantizes. The verifier
# holds its own copy built by the identical invocation and grades from that copy, never
# from this one.
#
# IT IS BUILT ON ENTRY TO THIS SURFACE RATHER THAN INTO A LAYER, and the reason is that
# an image build has no accelerator. See environment/provision.sh, which carries the
# invocation, the measurement behind that statement and the step count.
#
# BASH_ENV is how the provisioning reaches every route into this surface. Bash sources
# it before running a non-interactive script, so an entry point that is handed
# solution/solve.sh provisions before solve.sh's first line; /root/.bashrc covers an
# interactive session. Both are one file test once the checkpoint exists.
COPY environment/provision.sh /workspace/environment/provision.sh
RUN chmod +x /workspace/environment/provision.sh \
 && printf '. /workspace/environment/provision.sh\n' >> /root/.bashrc
ENV BASH_ENV=/workspace/environment/provision.sh

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
