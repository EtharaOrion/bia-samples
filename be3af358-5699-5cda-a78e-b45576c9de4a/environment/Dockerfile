# Agent environment for slot OER-22.
#
# Pinned by @sha256: and never by a floating tag. A tag would validate cleanly and
# silently resolve to different bytes on a later pull, and the whole task rests on the
# agent and the verifier running the same kernels over the same frozen training run.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-22 \
    OER_SUBMISSION=/workspace/submission/selection.json \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /app

# The frozen half: the architecture, the substrate declaration, the harness that runs the
# frozen training schedule and combines snapshots, and the schema that bounds a selection.
# All four are copied into the VERIFIER image too, and tests/Dockerfile refuses to build
# unless they are byte-identical across the two surfaces. That identity is what makes a
# number measured here predictive of the graded number.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/selection_schema.py /app/selection_schema.py

# The starting point and the measuring instrument.
COPY environment/default_selection.json /app/default_selection.json
COPY environment/train_local.py /app/train_local.py
COPY environment/submission_protocol.md /app/submission_protocol.md

# The corpus. train_slice.bin is EXACTLY the 16777216 tokens the frozen run consumes plus
# one for the shifted target: the training run is frozen, so there is no extra training
# data to find and none would be usable if there were. devset_slice.bin is a proxy
# validation cut from a DIFFERENT FineWeb training shard, for the agent's own use.
#
# The graded split is NOT here. It is a slice of fineweb_val_000000 staged only into the
# verifier image, which is what makes the graded loss something a selection cannot
# precompute, and what makes the devset-to-holdout reordering a real risk to manage
# rather than a hazard the agent can inspect.
COPY environment/data/ /app/data/

# Build-time conformance. CPU only on purpose: the accelerator is a run-time resource and
# a build that depended on it would be a build that fails on a machine without one. This
# catches schema drift, substrate drift, a snapshot schedule that does not tile the
# budget, and a truncated corpus at BUILD rather than at grade.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, selection_schema
spec = harness.load_spec()
budget, snaps = spec["budget"], spec["snapshots"]
assert budget["micro_batch"] * spec["architecture"]["seq_len"] * budget["micro_steps"] \
    == budget["total_train_tokens"], "budget is inconsistent"
assert snaps["every_steps"] * snaps["count"] == budget["micro_steps"], \
    "the snapshot schedule does not tile the budget"
assert len(snaps["ids"]) == snaps["count"], "the snapshot id list is the wrong length"
assert 0 < snaps["storage_budget"] < snaps["count"], \
    "a storage budget that admits every snapshot is not a selection problem"
default = selection_schema.load("/app/default_selection.json", spec)
assert len(default["keep"]) <= snaps["storage_budget"]
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

# The submission handoff. /workspace is the volume the verifier reads; /app is NOT shared
# with it, so a selection left in /app is a selection that is never graded.
RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
