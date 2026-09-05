# The agent-facing task environment. This is the image the AGENT PHASE runs in.
#
# Pinned by digest, never by a floating tag. Harbor accepts any string and does not itself
# require a digest, so the digest requirement is this contract's and this file enforces it.
FROM python@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7

# The substrate is exact integer arithmetic over a frozen stream with an instrumented read
# counter. It needs no accelerator runtime and no third-party package: everything it uses is in
# the standard library. That is deliberate, because a reproducible grading path cannot rest on a
# wheel resolving the same way twice, and it is why nothing is installed here.

# The two shared paths of the two-phase contract, named here so that the workspace carrier is
# derivable from the bundle's own bytes. OER28_WORKSPACE is where the agent works and where the
# handoff lands. OER28_HARNESS_LOGS is where the agent-phase driver writes the harness journal
# that the verifier phase reads back.
ENV OER28_WORKSPACE=/workspace
ENV OER28_HARNESS_LOGS=/logs/harness
ENV OER28_SUBMISSION=/app/submission.py
ENV PYTHONPATH=/task/environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /workspace

COPY opstream.py /task/environment/opstream.py
COPY phase_a.py /task/environment/phase_a.py
COPY phase_b.py /task/environment/phase_b.py
COPY stream.json /task/environment/stream.json
COPY params.json /task/environment/params.json

# /app is where the graded kernel goes, /workspace is where the handoff goes, and
# /logs/harness is where the agent-phase driver writes the harness journal. The three are
# separate carriers on purpose, and all are created here so a phase that runs finds them.
RUN mkdir -p /app /workspace /logs/harness /logs/verifier

CMD ["/bin/bash"]
