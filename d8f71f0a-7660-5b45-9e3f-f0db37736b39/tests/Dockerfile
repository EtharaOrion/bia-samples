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
COPY evaluate.py /verifier/tests/evaluate.py
COPY checkers.py /verifier/tests/checkers.py
COPY checkers.yaml /verifier/tests/checkers.yaml
COPY grade.py /verifier/tests/grade.py
COPY reference_floor.py /verifier/tests/reference_floor.py
COPY test_output.py /verifier/tests/test_output.py
COPY fixtures.json /verifier/tests/fixtures.json
COPY rubrics.jsonl /verifier/tests/rubrics.jsonl

# The held-out split pin. It lives on THIS surface and only this surface: the agent image
# copies a declaration that carries no payload, and this is the byte that says which shard,
# which offset and which digest the graded evaluation resolves against.
COPY held_out_split.json /verifier/tests/held_out_split.json

# The validation shards the pin resolves to are staged here at provisioning time and are
# never mounted into the agent image.
RUN mkdir -p /logs/verifier /verifier/data/fineweb10B && chmod +x /verifier/tests/test.sh

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
