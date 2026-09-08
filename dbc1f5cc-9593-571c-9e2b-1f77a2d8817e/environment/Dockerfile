# Agent environment.
#
# Pinned by @sha256: and never by a floating tag. A tag would validate cleanly and
# silently resolve to different bytes on a later pull, and the whole task rests on
# the agent and the verifier running the same kernels.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-17 \
    OER_SUBMISSION=/workspace/submission/packing.json \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /app

# The frozen half: the architecture, the substrate declaration, the harness that
# trains it and the schema that bounds a plan. harness.py, packing_schema.py,
# model/nanogpt.py, frozen/task_spec.json and default_packing.json are copied into
# the VERIFIER image too, and tests/Dockerfile refuses to build unless all five are
# byte-identical across the two surfaces. That identity is what makes a number
# measured here predictive of the graded number.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/packing_schema.py /app/packing_schema.py

# The starting point and the measuring instrument.
COPY environment/default_packing.json /app/default_packing.json
COPY environment/train_local.py /app/train_local.py
COPY environment/submission_protocol.md /app/submission_protocol.md

# The corpus. train_slice.bin is longer than the frozen token budget so that every
# geometry on the grid can be fed in order without the stream running out; the
# budget rather than the data is the scarce thing here. devset_slice.bin is a proxy
# validation cut from a DIFFERENT FineWeb training shard, for the agent's own use.
#
# The graded split is NOT here. It is a slice of fineweb_val_000000 that is staged
# only into the verifier image, which is what makes the graded loss something a
# submission cannot precompute or overfit.
COPY environment/data/ /app/data/

# Build-time conformance. CPU only on purpose: the accelerator is a run-time
# resource and a build that depended on it would be a build that fails on a machine
# without one. This catches schema drift, substrate drift and a truncated corpus at
# build rather than at grade.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, packing_schema
spec = harness.load_spec()
geometry = spec["geometry"]
# The schema's grid and the substrate declaration must be the same grid.
assert tuple(geometry["allowed_rows"]) == packing_schema.ALLOWED_ROWS
assert tuple(geometry["allowed_seq_len"]) == packing_schema.ALLOWED_SEQ_LEN
assert tuple(geometry["allowed_grad_accum"]) == packing_schema.ALLOWED_GRAD_ACCUM
assert geometry["min_microbatch_tokens"] == packing_schema.MIN_MICROBATCH_TOKENS
assert geometry["max_microbatch_tokens"] == packing_schema.MAX_MICROBATCH_TOKENS
assert geometry["max_phases"] == packing_schema.MAX_PHASES
assert geometry["min_phase_optimizer_steps"] == packing_schema.MIN_PHASE_OPTIMIZER_STEPS
total = int(spec["budget"]["total_train_tokens"])
plan = packing_schema.load("/app/default_packing.json", total)
phases = harness.expand(plan, spec)
consumed = sum(p["tokens"] for p in phases)
assert consumed > 0.99 * total, f"the default plan consumes {consumed} of {total}"
need = total + 1
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
# shared with it, so a plan left in /app is a plan that is never graded.
RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
