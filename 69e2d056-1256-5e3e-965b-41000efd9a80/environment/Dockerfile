# Agent environment for slot OER-21.
# Pinned by digest, never by a floating tag.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /work

# The task substrate the agent works on. The canonical nanoGPT operating point, the
# architecture module, the checkpoint manifest and its parameter shard table, the
# calibration manifest, the handed quantization toolkit, and the published scheme schema
# that names every free axis.
COPY nanogpt_substrate.json /work/environment/nanogpt_substrate.json
COPY model /work/environment/model
COPY corpus /work/environment/corpus
COPY quantize.py /work/environment/quantize.py
COPY scheme_schema.json /work/environment/scheme_schema.json

# What is deliberately NOT copied. The held-out FineWeb slices the grade is computed on
# are named only in tests/anchors.json and are mounted only into the verifier. No COPY
# above reaches them, nothing under /work names them, and tests/runner.py launches the
# submission under a four-name environment allowlist that cannot name the verifier mount.
# The number that scores a submission is therefore computed on parameters and tokens this
# container has no path to.

# Provisioning. The frozen checkpoint and the agent-visible calibration slice are staged
# here at build time at the container paths environment/model/checkpoint.json and
# environment/corpus/eval_corpus.json name, out of the build context this bundle carries.
# The base image is pinned by digest and carries no fetch client, and this build is invoked
# with no build argument, so provisioning reads bytes rather than a URL. The verifier stages
# its OWN copy of the checkpoint separately and grades from that one, so a change to the
# copy in this container moves nothing about the grade.
#     /work/environment/model/checkpoint.pt      the frozen parameter snapshot
#     /work/environment/corpus/calibration.bin   the calibration split, train shard
COPY provision.sh /work/environment/provision.sh
RUN bash /work/environment/provision.sh

ENV PYTHONDONTWRITEBYTECODE=1
ENV OUT_DIR=/work/out
ENV OER21_ENVIRONMENT=/work/environment

CMD ["/bin/bash"]
