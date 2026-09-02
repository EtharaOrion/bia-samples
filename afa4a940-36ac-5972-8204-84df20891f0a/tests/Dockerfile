# Isolated verifier environment for slot OER-10.
#
# Pinned by digest, never by a floating tag, and the same digest the agent environment
# pins so the two surfaces differ by policy rather than by accident of build.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The verifier surface is EGRESS DENIED. The agent surface is egress open. The asymmetry is
# deliberate and is bound in task.toml under [agent] and [verifier]; it is recorded in
# solution/grounding.yaml under solver_egress_divergence and it is never an accident.
ENV OER10_TELEMETRY=/logs/verifier/telemetry.json \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /verifier

COPY test.sh /verifier/tests/test.sh
COPY runner.py /verifier/tests/runner.py
COPY checkers.py /verifier/tests/checkers.py
COPY checkers.yaml /verifier/tests/checkers.yaml
COPY grade.py /verifier/tests/grade.py
COPY test_output.py /verifier/tests/test_output.py
COPY fixtures.json /verifier/tests/fixtures.json
COPY rubrics.jsonl /verifier/tests/rubrics.jsonl

RUN mkdir -p /logs/verifier && chmod +x /verifier/tests/test.sh

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
