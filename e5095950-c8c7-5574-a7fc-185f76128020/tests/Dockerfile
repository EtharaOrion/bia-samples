# The isolated verifier environment.
#
# Same digest pin as the task environment, and pinned by digest for the same reason: a
# floating tag would let the grading interpreter move underneath a frozen bundle.
FROM python@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7

# EGRESS IS DENIED on this surface. The verifier is run with no network at all, and this file
# bakes its one dependency at BUILD time so nothing is resolved while a run is being graded.
# A verifier that can reach the network is a verifier whose verdict can be fetched.
RUN pip install --no-cache-dir "pytest==8.3.5"

ENV OER30_VERIFIER_EGRESS=denied
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /task

COPY . /task/tests/

# The verifier re-mints the sealed provenance store from the BUNDLE's frozen ledger source and
# frozen instance family, walks the seal chain itself to recover the realised attestation
# order, folds every atom digest itself, and solves the graded instance exactly. It never
# reads the store the agent's own image carried and it never accepts a digest or an order the
# submission asserted.
ENTRYPOINT ["/bin/bash", "/task/tests/test.sh"]
