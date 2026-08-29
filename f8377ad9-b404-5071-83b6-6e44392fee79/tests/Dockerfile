# The verifier surface. Isolated from the agent surface and pinned to the same
# digest, so the two differ by policy and by nothing else.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Egress DENIED at this surface. The solving agent has open egress; the verifier
# does not. That asymmetry is bound in task.toml and repeated here so the image
# and the contract cannot drift apart silently.
ENV OER20_SLOT=OER-20 \
    OER20_VERIFIER_EGRESS=denied \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /verifier

COPY test.sh /verifier/tests/test.sh
COPY runner.py /verifier/tests/runner.py
COPY grade.py /verifier/tests/grade.py
COPY checkers.py /verifier/tests/checkers.py
COPY checkers.yaml /verifier/tests/checkers.yaml
COPY emit_reward.py /verifier/tests/emit_reward.py
COPY test_output.py /verifier/tests/test_output.py
COPY rubrics.jsonl /verifier/tests/rubrics.jsonl

RUN chmod +x /verifier/tests/test.sh

# The bound reward carrier and its companion score document.
VOLUME ["/logs/verifier"]

ENTRYPOINT ["/verifier/tests/test.sh"]
