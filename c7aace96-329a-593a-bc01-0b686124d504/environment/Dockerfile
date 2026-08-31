# The agent surface. Pinned by digest, never by a floating tag: a tag is a moving
# target and a bundle pinned to one is a bundle whose environment nobody froze.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Egress at THIS surface is open by design: the solving agent fetches its own
# packages, datasets and models. The verifier surface is a different image with a
# different policy, and the asymmetry is bound in task.toml rather than left to
# whatever the runtime happens to allow.
ENV OER20_SLOT=OER-20 \
    OER20_FAMILY=F9 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /workspace

COPY model_stats.json /workspace/environment/model_stats.json
COPY eval_corpus.json /workspace/environment/eval_corpus.json
COPY reference_unquantized.json /workspace/environment/reference_unquantized.json
COPY quantize_lab.py /workspace/environment/quantize_lab.py

# The frozen substrate is read-only on the agent surface. This is a convenience
# for the agent, not the proof: the read-only guarantee is established by the way
# Harbor assembles the lower layer for a non-oracle agent and is measured in
# seed/tasks/OER-20/adequacy.py. A mount option describes an enforcement a kernel
# would apply if a mount happened, and a claim resting on a mount nobody performed
# is a claim about nothing.
RUN chmod -R a-w /workspace/environment

CMD ["/bin/bash"]
