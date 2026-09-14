# OER-08 verifier environment. Isolated from the agent environment on purpose.
#
# Same digest pin as environment/Dockerfile. A floating tag would leave the
# grading surface free to move underneath a frozen bundle.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /verifier

# The verifier keeps the bound network policy: egress denied. The solving agent
# gets open egress inside its own sandbox. The asymmetry is deliberate and is
# bound in task.toml, not left to the runtime to decide.
ENV OER08_VERIFIER_EGRESS=denied
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

COPY test.sh /verifier/tests/test.sh
COPY runner.py /verifier/tests/runner.py
COPY checkers.py /verifier/tests/checkers.py
COPY checkers.yaml /verifier/tests/checkers.yaml
COPY grade.py /verifier/tests/grade.py
COPY rubrics.jsonl /verifier/tests/rubrics.jsonl
COPY test_output.py /verifier/tests/test_output.py
COPY heldout_spec.json /verifier/tests/heldout_spec.json
COPY budget_spec.json /verifier/tests/budget_spec.json

RUN chmod +x /verifier/tests/test.sh

# The reward carrier the live verifier binds, and the companion score document.
VOLUME ["/logs/verifier"]

ENTRYPOINT ["/verifier/tests/test.sh"]
