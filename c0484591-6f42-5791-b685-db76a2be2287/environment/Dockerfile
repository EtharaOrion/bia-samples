# Agent environment for OER-05.
#
# Pinned by digest, never by a floating tag. A floating tag can be re-pointed
# after this bundle is frozen, which would silently change the bytes that
# graded while every hash over this file stayed still.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /workspace

# The frozen substrate. train_track3.py and frozen_contract.json are the locked
# halves; update_rule.py is the one file the agent edits.
COPY train_track3.py /workspace/train_track3.py
COPY frozen_contract.json /workspace/frozen_contract.json
COPY update_rule.py /workspace/update_rule.py

# The solving agent has open egress inside the sandbox, so no offline flag is
# set here. The verifier image is a different image with egress denied; see
# tests/Dockerfile.
ENV PYTHONHASHSEED=0
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python3", "/workspace/train_track3.py"]
