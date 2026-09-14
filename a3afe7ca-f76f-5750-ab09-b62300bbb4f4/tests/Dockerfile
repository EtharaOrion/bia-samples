# OER-11 ISOLATED VERIFIER environment.
#
# Pinned by digest, never by a floating tag, and it is the same digest the agent
# environment pins so the two surfaces differ only where they are declared to differ.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# EGRESS DENIED. The solving agent's sandbox has OPEN egress; this one does not. The
# asymmetry is a bound byte in task.toml ([agent].solver_egress = "open",
# [verifier].egress = "denied") and the divergence is recorded in
# solution/grounding.yaml under solver_egress_divergence. A declaration in a Dockerfile
# is not an enforcement, and this file makes no enforcement claim.
ENV OER11_VERIFIER_EGRESS=denied
ENV OER11_HARNESS_LOGS=/logs/harness

WORKDIR /verifier

COPY test.sh /verifier/test.sh
COPY emit_reward.py /verifier/emit_reward.py
COPY checkers.py /verifier/checkers.py
COPY checkers.yaml /verifier/checkers.yaml
COPY grade.py /verifier/grade.py
COPY runner.py /verifier/runner.py
COPY rubrics.jsonl /verifier/rubrics.jsonl
COPY test_output.py /verifier/test_output.py

RUN chmod +x /verifier/test.sh

# /logs/verifier is where the bound reward carrier lands. The trap in test.sh writes
# /logs/verifier/reward.txt and /logs/verifier/score.json on every exit path.
VOLUME ["/logs"]

ENTRYPOINT ["/verifier/test.sh"]
