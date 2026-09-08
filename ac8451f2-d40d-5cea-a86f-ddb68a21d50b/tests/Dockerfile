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

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
#
# A bare `COPY . /task/tests/` under a bundle-root context lands the whole bundle one level
# too deep: the entry point resolves /task/tests/test.sh while the file arrives at
# /task/tests/tests/test.sh, the container dies with "No such file or directory", and no
# reward carrier is ever written because the EXIT trap in test.sh never runs. That same
# bare COPY also ships solution/ into the verifier image, putting the private reference on
# the grading surface. Both are fixed by naming the two directories the verifier reads.
COPY tests/ /task/tests/

# grade.py resolves BUNDLE_DIR as TESTS_DIR/.. and reads environment/instances.json,
# environment/ledger_source.json and environment/mint_store.py from it, so the frozen
# environment has to be present verifier-side or the re-mint raises before the submission
# is ever opened. solution/ is deliberately absent from this image.
COPY environment/ /task/environment/

# The bound reward carrier is /logs/verifier/reward.txt. The EXIT trap writes through that
# path on the abort branch too, so the directory must exist even with nothing mounted over it.
RUN mkdir -p /logs/verifier /logs/agent

# The verifier re-mints the sealed provenance store from the BUNDLE's frozen ledger source and
# frozen instance family, walks the seal chain itself to recover the realised attestation
# order, folds every atom digest itself, and solves the graded instance exactly. It never
# reads the store the agent's own image carried and it never accepts a digest or an order the
# submission asserted.
ENTRYPOINT ["/bin/bash", "/task/tests/test.sh"]
