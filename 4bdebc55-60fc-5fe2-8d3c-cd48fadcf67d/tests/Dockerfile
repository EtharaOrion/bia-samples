# OER-09 verifier environment. Isolated from the agent environment on purpose.
# Pinned by digest, never by a floating tag.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The verifier keeps the bound network policy: egress denied. The solving agent's surface has
# open egress by design and the two surfaces differ deliberately. This asymmetry is a bound
# byte in task.toml, never an accident.
ENV OER_SLOT=OER-09
ENV OER_VERIFIER_EGRESS=denied
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0
ENV NO_PROXY=*
ENV no_proxy=*

WORKDIR /verifier

COPY test.sh        /verifier/tests/test.sh
COPY runner.py      /verifier/tests/runner.py
COPY grade.py       /verifier/tests/grade.py
COPY checkers.py    /verifier/tests/checkers.py
COPY checkers.yaml  /verifier/tests/checkers.yaml
COPY test_output.py /verifier/tests/test_output.py
COPY rubrics.jsonl  /verifier/tests/rubrics.jsonl

RUN mkdir -p /logs/verifier /logs/harness && chmod +x /verifier/tests/test.sh

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
