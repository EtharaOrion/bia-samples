# The agent-facing task environment for OER-16.
#
# Pinned by digest, never by a floating tag: a tag is a mutable pointer and a bundle
# pinned to one is not frozen.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER16_SLOT=OER-16 \
    OER16_SUBMISSION=/workspace/submission/allocation.json \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /app

# The frozen half: the model menu and the FLOP accounting, the frozen training recipe
# with its width-scaling rule, the harness that trains an allocation and the schema
# that bounds one. All five are copied into the VERIFIER image too, and
# tests/Dockerfile refuses to build unless they are byte-identical across the two
# surfaces. That identity is what makes the FLOP arithmetic which bounds a submission
# here the same arithmetic that charges it at grading time.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/allocation_schema.py /app/allocation_schema.py

# The starting point and the measuring instrument.
COPY environment/default_allocation.json /app/default_allocation.json
COPY environment/train_local.py /app/train_local.py
COPY environment/submission_protocol.md /app/submission_protocol.md

# The corpus. It is FINITE and it is smaller than the budget can buy at the small end
# of the menu: an allocation that asks for more tokens than this holds cycles back to
# the start rather than being refused, which is the data-constrained half of the trade.
#
# devset_slice.bin is a proxy validation cut from a DIFFERENT FineWeb training shard.
# The graded split is NOT here: it is a slice of fineweb_val_000000 staged only into
# the verifier image, which is what makes the graded loss something an allocation
# cannot precompute or overfit.
COPY environment/data/ /app/data/

# Build-time conformance. CPU only on purpose: the accelerator is a run-time resource
# and a build that depended on it would be a build that fails on a machine without
# one. This catches schema drift, substrate drift and a truncated corpus at build
# rather than at grade.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, allocation_schema
spec = harness.load_spec()
budget = int(spec["budget"]["flop_budget"])
for model in spec["model_menu"]:
    shape = harness.model_shape(spec, model)
    assert shape["model_dim"] % shape["head_dim"] == 0, model
    assert harness.max_micro_steps(spec, model) >= allocation_schema.MIN_MICRO_STEPS, model
alloc = allocation_schema.load("/app/default_allocation.json", spec, harness)
train = harness.load_shard("/app/data/train_slice.bin")
assert train.size >= spec["data"]["corpus_tokens"], f"corpus {train.size} too short"
dev = harness.load_shard("/app/data/devset_slice.bin")
assert dev.size >= spec["evaluation"]["tokens"] + 1, "devset slice is too short"
mod = harness.load_model_module()
assert mod.LOGIT_SOFTCAP == spec["architecture_common"]["logit_softcap"]
assert mod.ATTENTION_SCALE == spec["architecture_common"]["attention_scale"]
print("environment conformance OK:", len(spec["model_menu"]), "menu entries,",
      train.size, "corpus tokens")
PY

# The submission handoff. /workspace is the volume the verifier reads; /app is NOT
# shared with it, so an allocation left in /app is one that is never graded.
RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
