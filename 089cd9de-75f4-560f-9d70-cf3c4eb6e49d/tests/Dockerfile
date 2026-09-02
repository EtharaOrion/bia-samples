# The verifier environment. Isolated from the agent surface and pinned by the same
# digest, so the grading run and the solving run cannot diverge on the interpreter.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Verifier egress is DENIED. The solving agent's surface is open; this one is not.
# The asymmetry is bound in task.toml and recorded in solution/grounding.yaml.
ENV OER15_VERIFIER_EGRESS=denied
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /verifier

COPY test.sh /verifier/tests/test.sh
COPY runner.py /verifier/tests/runner.py
COPY grade.py /verifier/tests/grade.py
COPY checkers.py /verifier/tests/checkers.py
COPY checkers.yaml /verifier/tests/checkers.yaml
COPY test_output.py /verifier/tests/test_output.py
COPY rubrics.jsonl /verifier/tests/rubrics.jsonl

# The reward carrier the runtime mounts. tests/test.sh terminates by writing it.
VOLUME ["/logs/verifier"]

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
