# The isolated verifier environment. Same digest pin, different surface.
#
# It is separate from the agent environment because the reward rests on losses this
# image produced by re-executing the submitted script. A verifier sharing the agent's
# container would let the submission reach the measurement, and the measurement is
# the whole control.
#
# Egress is DENIED here. The agent surface has open egress so it can fetch its own
# packages, datasets and models. The two surfaces differ by design, the asymmetry is
# bound in task.toml under [agent].solver_egress and [verifier].egress, and it is
# recorded in solution/grounding.yaml rather than left to be inferred.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /verifier

COPY runner.py /verifier/runner.py
COPY held_out_eval.py /verifier/held_out_eval.py
COPY checkers.py /verifier/checkers.py
COPY checkers.yaml /verifier/checkers.yaml
COPY grade.py /verifier/grade.py
COPY test_output.py /verifier/test_output.py
COPY rubrics.jsonl /verifier/rubrics.jsonl
COPY test.sh /verifier/test.sh

RUN chmod +x /verifier/test.sh && mkdir -p /logs/verifier

# The frozen validation shard and the frozen architecture module are mounted read-only
# by the harness at run time. They are never fetched here, because a verifier that
# reaches the network to obtain its own held-out split has no held-out split.
VOLUME ["/data/fineweb10B"]

ENTRYPOINT ["/verifier/test.sh"]
