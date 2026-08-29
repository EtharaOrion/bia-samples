# Agent container spec for BIA slot S04.
#
# Harbor reads this as the MAIN container rather than the verifier's. A verifier
# gets its own only when task.toml declares [verifier.environment], which this
# task omits, so the verifier runs in this same image. A separate verifier
# container could not reach the agent's /telemetry, which grade.py must read.
#
# On a normal run this file is never read: harbor pulls [environment].docker_image
# by digest. It exists for --force-build, where harbor builds this instead.
# Starting FROM the identical digest is what makes both paths converge. An
# independent base would let that flag silently swap the graded runtime.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The frozen substrate and the only writer of the run record. The corpus is built
# at run time from the image's own Python standard library source, so nothing is
# downloaded during scored work and the bundle carries no multi-megabyte fixture.
COPY frozen /workspace/environment/frozen
COPY runner /workspace/environment/runner

ENV BIA_TELEMETRY_DIR=/telemetry \
    BIA_FROZEN_DIR=/workspace/environment/frozen \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /telemetry /workspace/submission /logs/verifier
WORKDIR /workspace
