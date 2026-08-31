# The isolated verifier environment. Same pinned digest as the agent environment so
# the two surfaces differ only where they are declared to differ.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

WORKDIR /verifier

COPY . /verifier/tests

# MEASURED CORRECTION, wave-11. tests/grade.py:145 resolves the submission's tool
# directory as BUNDLE/"environment"/"tools", and BUNDLE is /verifier in this image
# because the line above copies tests/ alone. /verifier/environment/tools therefore
# did not exist, tests/runner.py put a non-existent directory on the submission's
# PYTHONPATH, and every submission died on
#   ModuleNotFoundError: No module named 'corpus_diversity'
# before writing a line of corpus. Measured in the delivered verifier image: the
# bundle's OWN reference generator graded 0.0 with reason generator-collapsed-silently
# at agent_metric 0.25, which is the score of an empty corpus. The verifier was
# refusing every submission, including the reference, for a reason that describes the
# submission rather than the missing tool.
#
# The tool is materialised from tests/diversity.py, which is byte-identical to
# environment/tools/corpus_diversity.py: both are
# e880b1296de8c0800981b1e13354bf448bb0c206ab3eef11c3ad1e310fc35e07, and
# tests/checkers.yaml pins that same value as sha256 and mirror_sha256. Copying the
# verifier's own already-present copy keeps the agent surface out of the verifier
# image and adds no byte the bundle did not already ship.
RUN mkdir -p /verifier/environment/tools \
 && cp /verifier/tests/diversity.py /verifier/environment/tools/corpus_diversity.py \
 && python3 -c "import hashlib,sys; p='/verifier/environment/tools/corpus_diversity.py'; d=hashlib.sha256(open(p,'rb').read()).hexdigest(); pin='e880b1296de8c0800981b1e13354bf448bb0c206ab3eef11c3ad1e310fc35e07'; sys.exit('diversity tool digest ' + d + ' does not match the tests/checkers.yaml pin ' + pin) if d != pin else sys.stdout.write('corpus_diversity pinned at ' + d + '\n')"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

# Verifier egress is denied, bound in task.toml under [verifier]. Nothing on the
# graded path reads the network, a clock or a random source, so the grading result
# is a function of the submission's corpus and of the frozen bytes alone.
ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
