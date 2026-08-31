# OER-13 verifier environment. Isolated from the agent environment on purpose.
#
# Same pinned digest as environment/Dockerfile, because the graded quantity is
# recomputed here and a different interpreter build is a different answer. Pinned
# by digest, never by a floating tag.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The asymmetry that matters: the solving agent has OPEN egress, the verifier has
# NONE. It is declared here and in task.toml [verifier].egress, so it is a bound
# byte rather than an accident of how a runner happened to be invoked.
ENV OER13_VERIFIER_EGRESS=denied
ENV http_proxy=""
ENV https_proxy=""
ENV no_proxy="*"

# Determinism of the grading process itself. The checkers read no clock, no
# locale and no random source; these are belt on top of that.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0
ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C
ENV LANG=C

WORKDIR /verifier

# The build context is the bundle root, so tests/, solution/ and environment/
# all land here: tests/grade.py imports the private reference from solution/ to
# measure the in-run floor composition, and tests/harness.py reads the pristine
# environment/ tree to measure the in-run baseline composition. Both are
# verifier-side reads of bundle bytes.
#
# The SUBMISSION never travels into this image. It is copied alone into a fresh
# temporary directory by tests/runner.py and launched out of process as a new
# session leader under a small environment allowlist, and tests/grade.py never
# imports it.
COPY . /verifier

# Harbor mounts the reward root. The entry point terminates by writing the bare
# float to reward.txt with the reason and metric block in score.json, and an EXIT
# trap guarantees a reward on every exit path including failure.
VOLUME ["/logs/verifier"]
LABEL oer13.reward_path="/logs/verifier/reward.txt"
LABEL oer13.score_document_path="/logs/verifier/score.json"
LABEL oer13.egress="denied"

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
