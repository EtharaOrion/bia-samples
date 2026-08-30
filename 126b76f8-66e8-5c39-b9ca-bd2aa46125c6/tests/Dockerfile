# Verifier environment for slot OER-03. Isolated from the agent environment.
#
# Same pinned digest as environment/Dockerfile, and pinned for the same reason: a
# scanned image that differs from the executed image makes the measurement
# unverifiable. The two environments differ in exactly two respects, and both are
# deliberate.
#
#   1. EGRESS IS DENIED HERE. The agent surface carries open egress so the
#      solving model can fetch its own packages, datasets and models. The
#      measurement the reward rests on must not be reachable over a network, so
#      this surface has none. Bound in task.toml under [verifier] egress.
#
#   2. THE HELD-OUT SPLIT LIVES HERE AND ONLY HERE. The graded loss is computed
#      by this environment on a split the agent container never receives, from
#      the parameter snapshots the pinned harness wrote. That is what makes the
#      graded readout unshapeable from inside the run.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV BIA_SLOT=OER-03 \
    BIA_ROLE=verifier \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    http_proxy="" \
    https_proxy="" \
    no_proxy="*"

WORKDIR /verifier

COPY test.sh checkers.py checkers.yaml grade.py runner.py rubrics.jsonl test_output.py golden.json \
     evaluator.py holdout.json /verifier/

# evaluator.py is NEW and it is half the repair. tests/grade.py:_held_out_loss
# has always done `import evaluator`, and measured against the pinned digest
# above, `python3 -c "import evaluator"` raises ModuleNotFoundError. _held_out_loss
# swallows that ImportError and returns None, which drops every point, empties the
# evaluation series and produces the `run-produced-no-verifier-state` zero the
# oracle drive recorded. holdout.json carries the stream seed that defines the
# graded split and is COPYed into this image and into nothing else.

# The verifier RE-RUNS the submission out of process (tests/runner.py), and the
# submission imports the pinned harness, the loader, the architecture and the
# optimizer from /app/env. Harbor mounts /app/submission.py into this image and
# does not mount /app/env, so this image carries its own byte-identical copy of
# the runtime modules. They are listed by name rather than as a directory so the
# projection is exact.
COPY agent_env/bia_harness.py agent_env/bia_loader.py agent_env/shape.json \
     agent_env/gpt.py agent_env/muon.py agent_env/bia_corpus.py \
     agent_env/operating_point.json /app/env/

# THE PEP 668 FIX, and the reason it is this one. The delivered predecessor ran
# `RUN pip install --no-cache-dir pytest==8.3.4 PyYAML==6.0.2` here and the build
# failed at that line: the base carries /usr/lib/python3.12/EXTERNALLY-MANAGED
# and its python3 IS the system interpreter, with no /opt/conda anywhere. The
# three candidate fixes were measured rather than argued:
#
#   --break-system-packages  installs INTO the system interpreter, uninstalling
#                            the pinned image's PyYAML 6.0.3 for 6.0.2 and
#                            pulling unpinned transitive dependencies, so the
#                            executed image stops being the scanned image.
#   a venv                   `python3 -m venv` cannot bootstrap here: ensurepip
#                            is absent from this base.
#   a different base         would move the digest task.toml pins, which is a
#                            re-sign trigger and is not the smallest fix.
#   --target                 installs into a private prefix, touches no system
#                            site-packages, keeps PyYAML 6.0.3 and torch 2.13.0
#                            exactly as the pinned image ships them, and pins the
#                            version. This is the one used.
#
# Deleting the line instead, as the migration predecessor did, made the build
# succeed and left the image with no pytest at all, so `python3 -m pytest
# test_output.py` in test.sh failed silently behind its `||` and the compiled
# rubric surface FORGE item 10f requires was inert in the pinned image. This
# restores it. THE PINNED BASE DIGEST IS UNCHANGED.
RUN pip install --no-cache-dir --target=/opt/bia-verifier-site pytest==8.3.4

ENV PYTHONPATH=/opt/bia-verifier-site

# The held-out split, and the training shard the re-run needs. Both are
# materialised from bound seeds, deterministically and without a network, so the
# verifier surface keeps `egress = "denied"` at run time.
RUN python3 /app/env/bia_corpus.py && \
    python3 -c "import sys; sys.path.insert(0, '/verifier'); import evaluator; print(evaluator.describe())"

# The reward directory Harbor mounts. Created here so an aborted verifier can
# still write its attributed zero rather than fail on a missing directory, which
# would turn a graded failure back into silence.
RUN mkdir -p /logs/verifier

RUN chmod +x /verifier/test.sh

ENTRYPOINT ["/verifier/test.sh"]
