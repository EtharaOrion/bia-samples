# Agent environment for OER-04.
#
# Pinned by @sha256: and never by a floating tag. A tag would validate cleanly and
# silently resolve to different bytes on a later pull, and the whole task rests on
# the agent and the verifier running the same kernels.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-04 \
    OER04_SUBMISSION=/workspace/submission/shape.json \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /app

# The frozen half: the computation, the substrate declaration, the harness that
# trains it and the schema that bounds a shape. harness.py, shape_schema.py,
# model/nanogpt.py, frozen/task_spec.json and default_shape.json are copied into
# the VERIFIER image too, and tests/Dockerfile refuses to build unless all five are
# byte-identical across the two surfaces.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/shape_schema.py /app/shape_schema.py

# The starting point and the measuring instruments.
COPY environment/default_shape.json /app/default_shape.json
COPY environment/shape_tool.py /app/shape_tool.py
COPY environment/train_local.py /app/train_local.py
COPY environment/submission_protocol.md /app/submission_protocol.md

# The corpus. train_slice.bin is EXACTLY the 25165824 tokens the frozen budget
# consumes plus one for the shifted target -- every shape trains on the identical
# stream. devset_slice.bin is a proxy validation cut from a DIFFERENT FineWeb
# training shard, for the agent's own use. The graded split is NOT here.
COPY environment/data/ /app/data/

# Build-time conformance, CPU only on purpose: the accelerator is a run-time
# resource and a build that depended on it would be a build that fails on a
# machine without one.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, shape_schema
spec = harness.load_spec()
budget = spec["budget"]
assert budget["micro_batch"] * spec["computation"]["seq_len"] * budget["micro_steps"] \
    == budget["total_train_tokens"], "budget is inconsistent"
bounds, band = spec["shape_bounds"], spec["parameter_band"]
default = shape_schema.load("/app/default_shape.json", bounds)
model, count = harness.build_model(default, spec, device="cpu")
shape_schema.parameter_band_check(count, band)
del model
need = budget["total_train_tokens"] + 1
got = harness.load_shard("/app/data/train_slice.bin")
assert got.size >= need, f"train slice holds {got.size} tokens, the budget needs {need}"
dev = harness.load_shard("/app/data/devset_slice.bin")
assert dev.size >= spec["evaluation"]["tokens"] + 1, "devset slice is too short"
mod = harness.load_model_module()
assert mod.LOGIT_SOFTCAP == spec["computation"]["logit_softcap"]
assert mod.ATTENTION_SCALE == spec["computation"]["attention_scale"]
print("environment conformance OK:", got.size, "train tokens, default shape holds", count, "parameters")
PY

RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
