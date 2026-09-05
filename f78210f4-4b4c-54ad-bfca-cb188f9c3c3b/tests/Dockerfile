# The verifier surface. Isolated from the agent surface and pinned to the same
# digest, so the two differ by policy and by nothing else.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Egress DENIED at this surface. The solving agent has open egress; the verifier
# does not. That asymmetry is bound in task.toml and repeated here so the image
# and the contract cannot drift apart silently.
ENV OER20_SLOT=OER-20 \
    OER20_VERIFIER_EGRESS=denied \
    OER20_CHECKPOINT=/checkpoint/nanogpt_fp32.pt \
    OER20_HELDOUT=/verifier/heldout \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /verifier

COPY test.sh /verifier/tests/test.sh
COPY runner.py /verifier/tests/runner.py
COPY decoder.py /verifier/tests/decoder.py
COPY grade.py /verifier/tests/grade.py
COPY checkers.py /verifier/tests/checkers.py
COPY checkers.yaml /verifier/tests/checkers.yaml
COPY calibration.py /verifier/tests/calibration.py
COPY emit_reward.py /verifier/tests/emit_reward.py
COPY test_output.py /verifier/tests/test_output.py
COPY rubrics.jsonl /verifier/tests/rubrics.jsonl

# The verifier's OWN copy of the frozen architecture declaration. tests/decoder.py
# reads this path and never the copy inside an agent-visible container, so editing
# environment/nanogpt_substrate.json moves a digest the checkers grade and moves
# nothing the model is built from.
COPY nanogpt_substrate.json /verifier/nanogpt_substrate.json

RUN chmod +x /verifier/tests/test.sh

# The held-out FineWeb validation slice. Mounted here and NOWHERE else. It is the
# only corpus the graded perplexity is computed over, the evaluation order is
# derived from the digest of its manifest, and no agent-visible container carries
# it or names its points. If this mount is absent the harness produces zero
# evaluation points and the metric is undefined, which is the intended behaviour
# and not a degraded one.
VOLUME ["/verifier/heldout"]

# The frozen checkpoint, the same parameters the agent surface was handed.
VOLUME ["/checkpoint"]

# The bound reward carrier and its companion score document.
VOLUME ["/logs/verifier"]

ENTRYPOINT ["/verifier/tests/test.sh"]
