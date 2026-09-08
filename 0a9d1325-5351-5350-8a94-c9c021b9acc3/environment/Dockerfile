# Agent environment for OER-09.
#
# Pinned by @sha256: and never by a floating tag. A tag would validate cleanly and
# silently resolve to different bytes on a later pull, and the whole task rests on
# the agent and the verifier running the same kernels.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-09 \
    OER09_SUBMISSION=/workspace/submission/filter.json \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /app

# The frozen half: the architecture, the substrate declaration, the harness that
# trains it, and the filter schema and runner. harness.py, filter_schema.py,
# model/nanogpt.py, frozen/task_spec.json and default_filter.json are copied into
# the VERIFIER image too, and tests/Dockerfile refuses to build unless all five
# are byte-identical across the two surfaces. That identity is what makes a number
# measured here predictive of the graded number.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/filter_schema.py /app/filter_schema.py

# The starting point and the two measuring instruments.
COPY environment/default_filter.json /app/default_filter.json
COPY environment/curate.py /app/curate.py
COPY environment/train_local.py /app/train_local.py
COPY environment/submission_protocol.md /app/submission_protocol.md

# The pool and its register. PUBLIC and identical on both surfaces: the agent can
# read every block and every annotation. What the agent does not have is the
# graded split.
COPY environment/pool/ /app/pool/

# The agent's own yardstick: a proxy validation cut from a DIFFERENT FineWeb
# training shard. The graded split is NOT here -- it is a slice of
# fineweb_val_000000 staged only into the verifier image, which is what makes the
# graded loss something a chain cannot precompute or overfit.
COPY environment/data/ /app/data/

# Build-time conformance, CPU only on purpose: the accelerator is a run-time
# resource and a build that depended on it would be a build that fails on a
# machine without one. This catches schema drift, substrate drift, a truncated
# pool and a register that does not describe the pool it ships with.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, filter_schema
spec = harness.load_spec()
budget, pool = spec["budget"], spec["pool"]
assert budget["micro_batch"] * spec["architecture"]["seq_len"] == budget["block_tokens"]
assert budget["blocks"] * budget["block_tokens"] == budget["total_train_tokens"]
blocks = harness.load_shard("/app/pool/pool_blocks.bin")
assert blocks.size == pool["blocks"] * pool["block_tokens"], \
    f"pool holds {blocks.size} tokens, the spec declares {pool['blocks'] * pool['block_tokens']}"
register = filter_schema.load_register("/app/pool/source_register.jsonl")
assert len(register) == pool["blocks"], \
    f"register holds {len(register)} rows, the pool holds {pool['blocks']} blocks"
assert [r["block_id"] for r in register] == list(range(pool["blocks"])), \
    "the register is not in pool order"
missing = [f for f in filter_schema.KNOWN_FIELDS if f not in register[0]]
assert not missing, f"the register is missing documented field(s): {missing}"
# The shipped default MUST be a chain that resolves nothing: it is the control.
default = filter_schema.load("/app/default_filter.json")
admitted, report = filter_schema.apply(default, register)
assert report["rules_unresolved"] == report["rules_total"], \
    "the shipped default is supposed to resolve no field at all"
assert report["pool_passed_through_unchanged"], \
    "the shipped default is supposed to admit the pool unchanged"
assert len(admitted) >= budget["blocks"]
dev = harness.load_shard("/app/data/devset_slice.bin")
assert dev.size >= spec["evaluation"]["tokens"] + 1, "devset slice is too short"
mod = harness.load_model_module()
assert mod.LOGIT_SOFTCAP == spec["architecture"]["logit_softcap"]
assert mod.ATTENTION_SCALE == spec["architecture"]["attention_scale"]
print("environment conformance OK:", pool["blocks"], "pool blocks,",
      dev.size, "devset tokens")
PY

# The submission handoff. /workspace is the volume the verifier reads; /app is NOT
# shared with it, so a chain left in /app is a chain that is never graded.
RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
