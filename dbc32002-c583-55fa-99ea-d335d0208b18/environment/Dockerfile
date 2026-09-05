# The agent surface. Pinned by digest, never by a floating tag: a tag is a moving
# target and a bundle pinned to one is a bundle whose environment nobody froze.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Egress at THIS surface is open by design: the solving agent fetches its own
# packages, datasets and models. The verifier surface is a different image with a
# different policy, and the asymmetry is bound in task.toml rather than left to
# whatever the runtime happens to allow.
ENV OER20_SLOT=OER-20 \
    OER20_FAMILY=F9 \
    OER20_CHECKPOINT=/checkpoint/nanogpt_fp32.pt \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /workspace

COPY nanogpt_substrate.json /workspace/environment/nanogpt_substrate.json
COPY model_stats.json /workspace/environment/model_stats.json
COPY eval_corpus.json /workspace/environment/eval_corpus.json
COPY reference_unquantized.json /workspace/environment/reference_unquantized.json
COPY nanogpt_model.py /workspace/environment/nanogpt_model.py
COPY train_nanogpt.py /workspace/environment/train_nanogpt.py
COPY quantize_lab.py /workspace/environment/quantize_lab.py

# The TRAIN shards, and only the train shards. The held-out validation shards are
# never staged into this image: they are the surface the grade is computed on and
# they live with the verifier. `data/cached_fineweb10B.py` is the upstream
# loader named in the substrate declaration, and it is invoked with the train
# glob alone.
RUN mkdir -p /workspace/data/fineweb10B

# The frozen checkpoint. It is produced ONCE by the pinned harness with
#     python3 /workspace/environment/train_nanogpt.py --out /checkpoint/nanogpt_fp32.pt
# and mounted read-only here and on the verifier surface, so the agent and the
# grader hold the same parameters by identity rather than by re-derivation. It is
# not trained inside this image at run time: the model is frozen, and only the
# per-tensor bit allocation and the quantization scheme are free. The verifier
# compares the digest of the mount against its own copy of the checkpoint
# manifest, so a checkpoint swapped underneath this surface is a refused run and
# not a different grade.
VOLUME ["/checkpoint"]

# The frozen substrate is read-only on the agent surface. This is a convenience
# for the agent, not the proof: the read-only guarantee is established by the way
# Harbor assembles the lower layer for a non-oracle agent and is measured in
# seed/tasks/OER-20/adequacy.py. A mount option describes an enforcement a kernel
# would apply if a mount happened, and a claim resting on a mount nobody performed
# is a claim about nothing.
RUN chmod -R a-w /workspace/environment

CMD ["/bin/bash"]
