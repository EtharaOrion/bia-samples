# Agent environment.
#
# Pinned by @sha256: and never by a floating tag. A tag would validate cleanly and
# silently resolve to different bytes on a later pull, and the whole task rests on
# the agent and the verifier running the same kernels.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER12_SLOT=OER-12 \
    OER12_SUBMISSION=/workspace/submission/policy.json \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /app

# The frozen half: the architecture, the substrate declaration, the frozen training
# recipe, the harness that runs it and the schema that bounds a policy. All five are
# copied into the VERIFIER image too, and tests/Dockerfile refuses to build unless
# they are byte-identical across the two surfaces. That identity is what makes a
# policy designed here predictable at grading time.
COPY environment/model/ /app/model/
COPY environment/frozen/ /app/frozen/
COPY environment/harness.py /app/harness.py
COPY environment/policy_schema.py /app/policy_schema.py

# The shipped default policy -- the low anchor of the reward scale -- and the
# instrument for designing a better one.
COPY environment/default_policy.json /app/default_policy.json
COPY environment/probe_local.py /app/probe_local.py
COPY environment/submission_protocol.md /app/submission_protocol.md

# The corpus. train_stream.bin is CLEAN and is exactly the tokens the frozen budget
# consumes plus one for the shifted target. The declared pipeline fault is applied
# at run time by harness.degrade() from a ramp start the verifier draws, so the
# stream on this surface is the stream before the fault rather than after it.
#
# devset_slice.bin is a proxy validation cut from a DIFFERENT FineWeb training
# shard, for the agent's own use. Neither the stream a policy pays to read nor the
# graded split is here: both are staged only into the verifier image, which is what
# makes a policy a procedure rather than a lookup.
COPY environment/data/ /app/data/

# Build-time conformance. CPU only on purpose: the accelerator is a run-time
# resource and a build that depended on it would be a build that fails on a machine
# without one. This catches schema drift, substrate drift and a truncated corpus at
# build rather than at grade.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/app")
import harness, policy_schema
spec = harness.load_spec()
budget = spec["budget"]
assert budget["micro_batch"] * spec["architecture"]["seq_len"] * budget["micro_steps"] \
    == budget["total_train_tokens"], "budget is inconsistent"
assert tuple(harness.candidate_steps(spec)) == policy_schema.candidate_steps(), \
    "frozen/task_spec.json and policy_schema.py disagree on the candidate grid"
policy_schema.load("/app/default_policy.json")
need = budget["total_train_tokens"] + 1
got = harness.load_shard("/app/data/train_stream.bin")
assert got.size >= need, f"train stream holds {got.size} tokens, the budget needs {need}"
dev = harness.load_shard("/app/data/devset_slice.bin")
assert dev.size >= spec["evaluation"]["devset_batches"] * spec["evaluation"]["tokens_per_batch"] + 1
mod = harness.load_model_module()
assert mod.LOGIT_SOFTCAP == spec["architecture"]["logit_softcap"]
assert mod.ATTENTION_SCALE == spec["architecture"]["attention_scale"]
print("environment conformance OK:", got.size, "train tokens,", dev.size, "devset tokens")
PY

# The submission handoff. /workspace is the volume the verifier reads; /app is NOT
# shared with it, so a policy left in /app is a policy that is never graded.
RUN mkdir -p /workspace/submission /logs/agent

CMD ["/bin/bash"]
