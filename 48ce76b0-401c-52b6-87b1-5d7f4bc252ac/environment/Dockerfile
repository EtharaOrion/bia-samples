# Agent container spec for the S03 data-order task.
#
# Harbor reads this as the MAIN container. This task declares no [verifier.environment], so
# environment_mode is shared and the verifier runs in this same image. That is required rather
# than incidental: the grader must read /telemetry, which the agent's run wrote, and a separate
# verifier container could not reach it.
#
# On a normal run this file is never read, because Harbor pulls [environment].docker_image by
# digest. It exists for --force-build. Starting FROM the identical digest is what makes both
# paths converge; an independent base would let that flag silently swap the graded runtime.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Only pure-python and numeric dependencies the runner and the verifier need. Nothing here
# changes the frozen substrate, which is digested from environment/runner/profiles.py at run
# time and compared by the VALUE checker rather than trusted.
RUN pip install --no-cache-dir "numpy>=2.0,<3" "pyyaml>=6.0,<7" "pytest>=8.0,<9"

# The runner is the only writer of the harness telemetry. It is installed outside /workspace so
# an agent editing its own workspace cannot silently replace it, and [verifier].env pins
# BIA_RUNNER to this path.
COPY runner/ /opt/bia/runner/

RUN mkdir -p /telemetry /logs/verifier /workspace/submission /workspace/.bia_cache
ENV BIA_TELEMETRY=/telemetry/run_record.jsonl \
    BIA_EGRESS_AUDIT=/telemetry/egress_audit.jsonl \
    BIA_ORDER_MODULE=/workspace/submission/order.py \
    BIA_RUNNER=/opt/bia/runner/run_s03.py \
    BIA_CACHE_DIR=/workspace/.bia_cache \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace
