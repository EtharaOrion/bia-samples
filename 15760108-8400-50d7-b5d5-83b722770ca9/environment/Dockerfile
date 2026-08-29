# Agent container spec for bia S09 multi-objective-frontier.
#
# Harbor reads this as the MAIN container. task.toml declares no
# [verifier.environment], so environment_mode is "shared" and the verifier runs
# in this same image. A separate verifier container could not reach the agent's
# telemetry directory, which tests/grade.py must read.
#
# On a normal run this file is never read: harbor pulls [environment].docker_image
# by digest. It exists for --force-build, where harbor builds this instead.
# Starting FROM the identical digest is what makes both paths converge. An
# independent base would let that flag silently swap the graded runtime.
#
# The base carries torch, numpy, PyYAML and pytest, which is everything the
# harness, the verifier and the recomputation instrument import. Nothing else
# is installed, because a layer that exists only on the --force-build path and
# is absent on the pull path is a difference between two runtimes that are
# supposed to be the same runtime.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The frozen fixtures and the frozen runner. The agent may read every byte of
# these: the contract is public and the frozen surface is enforced by the
# verifier reading live state, never by hiding the rules.
COPY frontier_spec.json       /workspace/harness/frontier_spec.json
COPY frontier_spec_smoke.json /workspace/harness/frontier_spec_smoke.json
COPY harness/engine.py        /workspace/harness/engine.py
COPY harness/run_frontier.py  /workspace/harness/run_frontier.py

RUN mkdir -p /workspace/submission /telemetry /logs/verifier

ENV BIA_S09_BUNDLE=/workspace \
    BIA_S09_SPEC=/workspace/harness/frontier_spec.json \
    BIA_S09_SUBMISSION=/workspace/submission/frontier.py \
    BIA_S09_TELEMETRY_DIR=/telemetry \
    BIA_S09_LOGS=/logs/verifier \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace
