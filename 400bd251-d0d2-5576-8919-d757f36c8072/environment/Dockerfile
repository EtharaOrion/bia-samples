# Agent environment for OER-28.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-28 \
    OER28_SUBMISSION=/workspace/submission/plan.json \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# The frozen half. harness.py, kernel_plan.py, model/nanogpt.py,
# frozen/task_spec.json and default_plan.json are copied into the VERIFIER image
# too, and tests/Dockerfile refuses to build unless all five are byte-identical.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/kernel_plan.py /app/kernel_plan.py
COPY environment/default_plan.json /app/default_plan.json
COPY environment/bench_local.py /app/bench_local.py
COPY environment/submission_protocol.md /app/submission_protocol.md

# Tokens for the agent's own benchmarking. The verifier times on its own held-out
# slice, but the cost of a matmul does not depend on what it multiplies, so this
# is a faithful surface to benchmark against.
COPY environment/data/ /app/data/

RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, kernel_plan
spec = harness.load_spec()
assert spec["step"]["micro_batch"] * spec["architecture"]["seq_len"] \
    == spec["step"]["tokens_per_step"], "step is inconsistent"
ref = harness.reference_plan(spec)
kernel_plan.validate({**ref, "schema": kernel_plan.SCHEMA_ID})
default = kernel_plan.load("/app/default_plan.json")
rows = spec["step"]["micro_batch"] * spec["architecture"]["seq_len"]
for chunks in kernel_plan.OPTIONS["loss_chunks"]:
    assert rows % chunks == 0, f"loss_chunks {chunks} does not divide {rows} rows"
dev = harness.load_shard("/app/data/devset_slice.bin")
need = (spec["equivalence"]["batches"] + 1) * spec["step"]["tokens_per_step"] + 1
assert dev.size >= need, f"devset slice holds {dev.size} tokens, {need} are needed"
mod = harness.load_model_module()
assert mod.LOGIT_SOFTCAP == spec["architecture"]["logit_softcap"]
assert mod.ATTENTION_SCALE == spec["architecture"]["attention_scale"]
assert set(mod.BACKENDS) == set(kernel_plan.OPTIONS["attention"]), \
    "the model and the schema disagree about which attention backends exist"
print("environment conformance OK:", dev.size, "devset tokens")
PY

RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
