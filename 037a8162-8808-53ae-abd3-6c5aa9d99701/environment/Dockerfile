# Agent environment for OER-01.
#
# Pinned by @sha256: and never by a floating tag. A tag would validate cleanly and
# silently resolve to different bytes on a later pull, and the whole task rests on
# the agent and the verifier running the same kernels.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-01 \
    OER01_SUBMISSION=/workspace/submission/recipe.json \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /app

# The frozen half. harness.py, recipe_schema.py, records.py, model/nanogpt.py,
# frozen/task_spec.json, default_recipe.json and published_records.json are copied
# into the VERIFIER image too, and tests/Dockerfile refuses to build unless all
# seven are byte-identical across the two surfaces.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/recipe_schema.py /app/recipe_schema.py
COPY environment/records.py /app/records.py
COPY environment/published_records.json /app/published_records.json

COPY environment/default_recipe.json /app/default_recipe.json
COPY environment/train_local.py /app/train_local.py
COPY environment/submission_protocol.md /app/submission_protocol.md

# The corpus. train_slice.bin is EXACTLY the 25165824 tokens the ceiling allows plus
# one for the shifted target. devset_slice.bin is a proxy validation cut from a
# DIFFERENT FineWeb training shard, for the agent's own use. The graded split is NOT
# here.
COPY environment/data/ /app/data/

RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, recipe_schema, records
spec = harness.load_spec()
budget = spec["budget"]
assert budget["micro_batch"] * spec["architecture"]["seq_len"] * budget["micro_steps"] \
    == budget["total_train_tokens"], "budget is inconsistent"
ev = spec["evaluation"]
assert budget["micro_steps"] % ev["grid_micro_steps"] == 0, "the grid does not divide the ceiling"
assert ev["grid_batches"] * budget["micro_batch"] * spec["architecture"]["seq_len"] \
    == ev["grid_tokens"], "the grid window is inconsistent"
default = recipe_schema.load("/app/default_recipe.json")
corpus = records.load_records("/app/published_records.json")
assert corpus, "the published-record corpus is empty; the displacement gate would be a no-op"
for rec in corpus:
    recipe_schema.validate(rec["recipe"])
assert records.find_replay(default, corpus) is None, \
    "the shipped default is itself a published record, which would make the control unsubmittable"
need = budget["total_train_tokens"] + 1
got = harness.load_shard("/app/data/train_slice.bin")
assert got.size >= need, f"train slice holds {got.size} tokens, the ceiling needs {need}"
dev = harness.load_shard("/app/data/devset_slice.bin")
assert dev.size >= ev["grid_tokens"] + 1, "devset slice is too short for the grid window"
mod = harness.load_model_module()
assert mod.LOGIT_SOFTCAP == spec["architecture"]["logit_softcap"]
assert mod.ATTENTION_SCALE == spec["architecture"]["attention_scale"]
print("environment conformance OK:", got.size, "train tokens,", len(corpus), "published records")
PY

RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
