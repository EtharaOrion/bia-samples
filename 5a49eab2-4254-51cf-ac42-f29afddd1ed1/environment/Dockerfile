# Agent environment for OER-05.
#
# Pinned by @sha256: and never by a floating tag. A tag would validate cleanly and
# silently resolve to different bytes on a later pull, and the whole task rests on the
# agent and the verifier running the same kernels.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER05_SUBMISSION=/workspace/submission/update_rule.json \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /app

# The frozen half: the architecture, the substrate declaration, the harness that trains
# it and the schema that bounds a submission. harness.py, rule_schema.py,
# model/nanogpt.py, frozen/task_spec.json and default_rule.json are copied into the
# VERIFIER image too, and tests/Dockerfile refuses to build unless all five are
# byte-identical across the two surfaces. That identity is what makes a number measured
# here predictive of the graded number.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/rule_schema.py /app/rule_schema.py

# The starting point and the measuring instrument.
COPY environment/default_rule.json /app/default_rule.json
COPY environment/train_local.py /app/train_local.py
COPY environment/README.md /app/README.md

# The corpus. The graded split is NOT here: it is a slice of fineweb_val_000000 staged
# only into the verifier image, which is what makes the graded loss something a
# submission cannot precompute or overfit. environment/data/devset_slice.bin is a proxy
# cut from a FineWeb TRAINING shard and is the agent's own yardstick.
COPY environment/data/ /app/data/

# Build-time conformance. CPU only on purpose: the accelerator is a run-time resource
# and a build that depended on it would be a build that fails on a machine without one.
# This catches schema drift, substrate drift and a truncated corpus at build rather
# than at grade.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, rule_schema as schema
spec = harness.load_spec()
budget = spec["budget"]
assert budget["micro_batch"] * spec["architecture"]["seq_len"] * budget["micro_steps"] \
    == budget["total_train_tokens"], "budget is inconsistent"
schema.load("/app/default_rule.json")
need = budget["total_train_tokens"] + 1
got = harness.load_shard("/app/data/train_slice.bin")
assert got.size >= need, f"train slice holds {got.size} tokens, the budget needs {need}"
dev = harness.load_shard("/app/data/devset_slice.bin")
assert dev.size >= spec["evaluation"]["tokens"] + 1, "devset slice is too short"
mod = harness.load_model_module()
assert mod.LOGIT_SOFTCAP == spec["architecture"]["logit_softcap"]
assert mod.ATTENTION_SCALE == spec["architecture"]["attention_scale"]
print("OER-05 environment conformance OK")
PY

# The submission handoff. /workspace is the volume the verifier reads; /app is NOT
# shared with it, so a submission left in /app is a submission that is never graded.
RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
