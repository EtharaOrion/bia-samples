# Agent environment.
#
# Pinned by @sha256: and never by a floating tag. A tag would validate cleanly and
# silently resolve to different bytes on a later pull, and the whole task rests on
# the agent and the verifier running the same kernels.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-NANOGPT-RECIPE \
    OER_SUBMISSION=/workspace/submission/recipe.json \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /app

# The frozen half: the architecture, the substrate declaration, the harness that
# trains it and the schema that bounds a recipe. harness.py, recipe_schema.py,
# model/nanogpt.py, frozen/task_spec.json and default_recipe.json are copied into
# the VERIFIER image too, and tests/Dockerfile refuses to build unless all five are
# byte-identical across the two surfaces. That identity is what makes a number
# measured here predictive of the graded number.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/recipe_schema.py /app/recipe_schema.py

# The starting point and the measuring instrument.
COPY environment/default_recipe.json /app/default_recipe.json
COPY environment/train_local.py /app/train_local.py
COPY environment/submission_protocol.md /app/submission_protocol.md

# The corpus. train_slice.bin is EXACTLY the 25165824 tokens the frozen budget
# consumes plus one for the shifted target -- there is no extra training data to
# find, because the budget rather than the data is the scarce thing here.
# devset_slice.bin is a proxy validation cut from a DIFFERENT FineWeb training
# shard, for the agent's own use.
#
# The graded split is NOT here. It is a slice of fineweb_val_000000 that is staged
# only into the verifier image, which is what makes the graded loss something a
# submission cannot precompute or overfit.
COPY environment/data/ /app/data/

# Build-time conformance. CPU only on purpose: the accelerator is a run-time
# resource and a build that depended on it would be a build that fails on a
# machine without one. This catches schema drift, substrate drift and a truncated
# corpus at build rather than at grade.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, recipe_schema
spec = harness.load_spec()
budget = spec["budget"]
assert budget["micro_batch"] * spec["architecture"]["seq_len"] * budget["micro_steps"] \
    == budget["total_train_tokens"], "budget is inconsistent"
recipe_schema.validate(json.load(open("/app/default_recipe.json")))
need = budget["total_train_tokens"] + 1
got = harness.load_shard("/app/data/train_slice.bin")
assert got.size >= need, f"train slice holds {got.size} tokens, the budget needs {need}"
dev = harness.load_shard("/app/data/devset_slice.bin")
assert dev.size >= spec["evaluation"]["tokens"] + 1, "devset slice is too short"
mod = harness.load_model_module()
assert mod.LOGIT_SOFTCAP == spec["architecture"]["logit_softcap"]
assert mod.ATTENTION_SCALE == spec["architecture"]["attention_scale"]
print("environment conformance OK:", got.size, "train tokens,", dev.size, "devset tokens")
PY

# The submission handoff. /workspace is the volume the verifier reads; /app is NOT
# shared with it, so a recipe left in /app is a recipe that is never graded.
RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
