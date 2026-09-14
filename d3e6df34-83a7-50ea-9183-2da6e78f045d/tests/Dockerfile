# Isolated verifier environment for OER-05.
#
# Pinned by digest, exactly as [environment] pins it. A floating tag on the
# grading surface would let the bytes that graded move after the bundle froze.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /verifier

# The grading tree, and nothing else. The agent's working tree is not copied
# here and is never on this image's import path.
COPY . /verifier/tests

# Egress is DENIED on this surface. The solving agent's sandbox has open egress;
# the verifier's does not, and the two surfaces differ by design. The flags
# below are the local half of that posture: no index, no wheel fetch, no
# implicit download at grade time.
ENV PIP_NO_INDEX=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV NO_PROXY="*"
ENV PYTHONHASHSEED=0
ENV PYTHONDONTWRITEBYTECODE=1

# The verifier never trains an agent-authored rule inside the grading process.
# It replays telemetry it produced itself and re-evaluates on the frozen
# held-out split, which is why this surface needs no accelerator and why
# grading can never silently depend on one being present.
ENV CUDA_VISIBLE_DEVICES=""

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
