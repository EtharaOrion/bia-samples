# Verifier environment for OER-28.
#
# NOTE: this image declares NO VOLUME. A `VOLUME ["/logs/verifier"]` line would
# create an anonymous volume that shadows the /logs bind mount and silently
# discard the reward carriers.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-28 \
    OER28_SUBMISSION=/workspace/submission/plan.json \
    OER28_LOGDIR=/logs/verifier \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /verifier

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
COPY tests/ /verifier/tests/

# solution/ is deliberately NOT copied.

# THE IDENTITY ASSERTION. Five files must be byte-identical across the two surfaces
# or this image does not build.
COPY environment/harness.py            /verifier/mirror/harness.py
COPY environment/kernel_plan.py        /verifier/mirror/kernel_plan.py
COPY environment/model/nanogpt.py      /verifier/mirror/model/nanogpt.py
COPY environment/frozen/task_spec.json /verifier/mirror/frozen/task_spec.json
COPY environment/default_plan.json     /verifier/mirror/default_plan.json
RUN set -eu; \
    for f in harness.py kernel_plan.py model/nanogpt.py frozen/task_spec.json default_plan.json; do \
        cmp "/verifier/tests/$f" "/verifier/mirror/$f" \
          || { echo "DRIFT: tests/$f differs from environment/$f"; exit 1; }; \
    done; \
    echo "agent/verifier surface identity OK"; \
    rm -rf /verifier/mirror

# Build-time conformance, CPU only. The held-out slice must be long enough for the
# equivalence batches, both anchors must satisfy the SAME public schema a submission
# does, and they must be different plans -- two identical plans would leave no span.
RUN python3 - <<'PY'
import sys
sys.path.insert(0, "/verifier/tests")
import harness, kernel_plan
spec = harness.load_spec()
holdout = harness.load_shard("/verifier/tests/holdout/val_slice.bin")
need = (spec["equivalence"]["batches"] + 1) * spec["step"]["tokens_per_step"] + 1
assert holdout.size >= need, f"holdout holds {holdout.size} tokens, {need} are needed"
control = kernel_plan.as_execution(kernel_plan.load("/verifier/tests/default_plan.json"))
reference = kernel_plan.as_execution(kernel_plan.load("/verifier/tests/private/reference_plan.json"))
assert control != reference, "the two anchors are the same plan; there would be no span"
print("verifier conformance OK:", holdout.size, "holdout tokens")
PY

RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
