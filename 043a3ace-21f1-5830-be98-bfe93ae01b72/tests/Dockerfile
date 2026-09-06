# OER-09 verifier environment. Isolated from the agent environment on purpose.
# Pinned by digest, never by a floating tag.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The verifier keeps the bound network policy: egress denied. The solving agent's surface has
# open egress by design and the two surfaces differ deliberately. This asymmetry is a bound
# byte in task.toml, never an accident.
ENV OER_SLOT=OER-09
ENV OER_VERIFIER_EGRESS=denied
ENV OER09_HARNESS_LOGS=/logs/harness
ENV OER09_HELDOUT=/verifier/heldout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0
ENV NO_PROXY=*
ENV no_proxy=*

WORKDIR /verifier

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/. tests/test.sh resolves its own bundle
# root as the parent of the directory it was laid down in, and three of the files it drives
# read from trees BESIDE tests/ rather than from inside it: tests/test_output.py opens
# ${BUNDLE}/solution/fixtures.json, tests/grade.py imports ${BUNDLE}/solution/reference.py
# for the reference identity it stamps into the score document, and tests/harness.py reads
# ${BUNDLE}/environment for the frozen declarations and the source register it recomputes
# the graded state from. A tests-only context builds an image that carries none of them, and
# the first of those reads dies on FileNotFoundError inside a container that had already
# started, at which point the exit trap writes reward 0.0 with reason
# verifier-aborted-before-grading and nothing is ever graded. Whatever the entry point reads
# has to be in the image that runs it.
COPY tests/test.sh        /verifier/tests/test.sh
COPY tests/runner.py      /verifier/tests/runner.py
COPY tests/harness.py     /verifier/tests/harness.py
COPY tests/grade.py       /verifier/tests/grade.py
COPY tests/checkers.py    /verifier/tests/checkers.py
COPY tests/checkers.yaml  /verifier/tests/checkers.yaml
COPY tests/expected_pool.py /verifier/tests/expected_pool.py
COPY tests/test_output.py /verifier/tests/test_output.py
COPY tests/rubrics.jsonl  /verifier/tests/rubrics.jsonl

# The frozen declarations and the source register. tests/harness.py recomputes the graded
# state over these bytes: the source pool digest and token total, the reference arm's own
# curation, the held-out stratum count, and the register schema the feed resolved. They are
# already agent-reachable, so carrying them here leaks nothing; carrying them here is what
# lets the verifier measure instead of trusting a number the run reported about itself.
COPY environment/ /verifier/environment/

# The two private solution files tests/ genuinely reads, and no others.
#
# fixtures.json is the compiled both-halves fixture set tests/test_output.py drives every
# checker over. reference.py is imported by tests/grade.py for REFERENCE_ID alone and by
# tests/harness.py for the reference filter chain the harness's own arm applies.
#
# solution/solve.sh is deliberately NOT copied. It is the reference solver, and baking it
# into the image would make it the default ${SUBMISSION} for any run that does not set
# OER09_SUBMISSION, so a submission that delivered nothing would be graded on the reference's
# work. tests/runner.py already grades an absent entry point as a recorded reason rather than
# as a fault, which is the behaviour that case is supposed to have.
COPY solution/fixtures.json /verifier/solution/fixtures.json
COPY solution/reference.py  /verifier/solution/reference.py

RUN mkdir -p /logs/verifier /logs/harness && chmod +x /verifier/tests/test.sh

# The held-out FineWeb validation split. Mounted here and NOWHERE else. It is the verifier's
# and environment/frozen_config.yaml declares it absent from the agent's container, so it
# cannot be baked into these bytes: it is 200001024 bytes of shard that lives outside the
# bundle. tests/harness.py resolves it from this mount and computes the graded readout over
# it. When the mount is absent the harness publishes the loss as null with that reason
# attached, which is the intended behaviour and not a degraded one.
VOLUME ["/verifier/heldout"]

# The bound reward carrier and its companion score document land under /logs/verifier, and
# /logs/harness is the harness log root tests/harness.py publishes the graded state into.
VOLUME ["/logs"]

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
