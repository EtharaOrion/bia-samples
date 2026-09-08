# Agent environment.
#
# Pinned by @sha256: and never by a floating tag. A tag would validate cleanly and
# silently resolve to different bytes on a later pull, and the whole task rests on the
# agent and the verifier running the same kernels.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-15 \
    OER_SUBMISSION=/workspace/submission/shape.json \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /app

# The frozen half: the decoder module whose SHAPE is the free variable, the substrate
# declaration, the harness that builds and trains it and the schema that bounds a
# shape. harness.py, shape_schema.py, model/nanogpt.py, frozen/task_spec.json and
# default_shape.json are copied into the VERIFIER image too, and tests/Dockerfile
# refuses to build unless all five are byte-identical across the two surfaces. That
# identity is what makes a number measured here predictive of the graded number.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/shape_schema.py /app/shape_schema.py

COPY environment/default_shape.json /app/default_shape.json
COPY environment/train_local.py /app/train_local.py
COPY environment/submission_protocol.md /app/submission_protocol.md

# The corpus. The graded split is NOT here: it is a slice of fineweb_val_000000 staged
# only into the verifier image, which is what makes the graded loss something a
# submission cannot precompute or overfit. devset_slice.bin is a proxy cut from a
# DIFFERENT FineWeb training shard, for the agent's own use.
COPY environment/data/ /app/data/

# Build-time conformance. CPU only on purpose: the accelerator is a run-time resource
# and a build that depended on it would be a build that fails on a machine without
# one.
#
# The load-bearing assertion is the last one: the schema's parameter arithmetic is
# checked against the REAL constructed module for every admissible shape on the grid.
# The band a submission is held to is therefore a statement about the model that will
# be built, not about a formula that might have drifted from it.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, shape_schema
spec = harness.load_spec()
budget = spec["shape_budget"]
grid = spec["shape_grid"]
assert tuple(grid["allowed_num_layers"]) == shape_schema.ALLOWED_NUM_LAYERS
assert tuple(grid["allowed_model_dim"]) == shape_schema.ALLOWED_MODEL_DIM
assert tuple(grid["allowed_head_dim"]) == shape_schema.ALLOWED_HEAD_DIM
assert tuple(grid["allowed_mlp_ratio"]) == shape_schema.ALLOWED_MLP_RATIO
b = spec["budget"]
assert b["micro_batch"] * spec["architecture_frozen"]["seq_len"] * b["micro_steps"] \
    == b["total_train_tokens"], "budget is inconsistent"
shape_schema.load("/app/default_shape.json", budget)

nanogpt = harness.load_model_module()
assert nanogpt.LOGIT_SOFTCAP == spec["architecture_frozen"]["logit_softcap"]
assert nanogpt.ATTENTION_SCALE == spec["architecture_frozen"]["attention_scale"]
admissible = 0
for L in shape_schema.ALLOWED_NUM_LAYERS:
    for d in shape_schema.ALLOWED_MODEL_DIM:
        for h in shape_schema.ALLOWED_HEAD_DIM:
            if d % h:
                continue
            for m in shape_schema.ALLOWED_MLP_RATIO:
                model = nanogpt.GPT(vocab_size=int(spec["architecture_frozen"]["vocab_size"]),
                                    num_layers=L, model_dim=d, head_dim=h, mlp_ratio=m)
                real = sum(p.numel() for n, p in model.named_parameters()
                           if not n.startswith("embed") and not n.startswith("proj"))
                calc = shape_schema.non_embedding_parameters(L, d, h, m)
                assert real == calc, f"L={L} d={d} h={h} m={m}: module {real}, schema {calc}"
                lo = budget["non_embedding_parameters"] * (1 - budget["tolerance_frac"])
                hi = budget["non_embedding_parameters"] * (1 + budget["tolerance_frac"])
                if lo <= real <= hi:
                    admissible += 1
assert admissible >= 8, f"only {admissible} shapes are admissible; the space is too thin"
print("parameter accounting verified against the real module on the whole grid;",
      admissible, "admissible shapes")

need = b["total_train_tokens"] + 1
got = harness.load_shard("/app/data/train_slice.bin")
assert got.size >= need, f"train slice holds {got.size} tokens, the budget needs {need}"
dev = harness.load_shard("/app/data/devset_slice.bin")
assert dev.size >= spec["evaluation"]["tokens"] + 1, "devset slice is too short"
print("environment conformance OK:", got.size, "train tokens,", dev.size, "devset tokens")
PY

RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
