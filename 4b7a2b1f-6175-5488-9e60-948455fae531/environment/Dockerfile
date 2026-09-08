# Agent environment.
#
# Pinned by @sha256: and never by a floating tag. A tag would validate cleanly and
# silently resolve to different bytes on a later pull, and the whole task rests on
# the agent and the verifier running the same kernels.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER08_SLOT=OER-08 \
    OER08_SUBMISSION=/workspace/submission/plan.json \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /app

# The frozen half: the architecture, the substrate declaration, the frozen training
# recipe, the harness that assembles and trains a plan, and the schema that bounds
# one. All five are copied into the VERIFIER image too, and tests/Dockerfile refuses
# to build unless they are byte-identical across the two surfaces. That identity is
# what makes a number measured here predictive of the graded number.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/plan_schema.py /app/plan_schema.py

# The starting point and the measuring instrument.
COPY environment/default_plan.json /app/default_plan.json
COPY environment/probe_local.py /app/probe_local.py
COPY environment/submission_protocol.md /app/submission_protocol.md

# The pool. Eight sources of 4194304 tokens each against an 8388608 token budget, so
# the budget buys a quarter of what is here and which quarter is the free variable.
# devset_slice.bin is a proxy validation cut from a DIFFERENT FineWeb training shard,
# for the agent's own use.
#
# The graded split is NOT here. It is a slice of fineweb_val_000000 staged only into
# the verifier image, which is what makes the graded loss something a plan cannot
# precompute or overfit.
COPY environment/data/ /app/data/

# Build-time conformance. CPU only on purpose: the accelerator is a run-time resource
# and a build that depended on it would be a build that fails on a machine without
# one. This catches schema drift, substrate drift and a truncated source at build
# rather than at grade.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, plan_schema
spec = harness.load_spec()
budget = spec["budget"]
assert budget["micro_batch"] * spec["architecture"]["seq_len"] * budget["micro_steps"] \
    == budget["total_train_tokens"], "budget is inconsistent"
pool = harness.load_pool(spec, "/app/data/sources")
assert len(pool) == len(spec["pool"]["sources"]), "the pool on disk is not the declared pool"
assert sum(t.size for t in pool.values()) >= spec["pool"]["total_pool_tokens"]
plan = plan_schema.load("/app/default_plan.json", spec)
stream = harness.assemble(plan, spec, pool)
assert stream.size == budget["total_train_tokens"] + 1, stream.size
dev = harness.load_shard("/app/data/devset_slice.bin")
assert dev.size >= spec["evaluation"]["tokens"] + 1, "devset slice is too short"
mod = harness.load_model_module()
assert mod.LOGIT_SOFTCAP == spec["architecture"]["logit_softcap"]
assert mod.ATTENTION_SCALE == spec["architecture"]["attention_scale"]
print("environment conformance OK:", len(pool), "sources,", stream.size, "assembled tokens")
PY

# The submission handoff. /workspace is the volume the verifier reads; /app is NOT
# shared with it, so a plan left in /app is a plan that is never graded.
RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
