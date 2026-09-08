# Verifier environment.
#
# Same pinned digest as the agent image, and pinned for the same reason. The two
# surfaces must run the same kernels or the anchors measured here would not be
# comparable to anything the agent measured.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER12_SLOT=OER-12 \
    OER12_SUBMISSION=/workspace/submission/policy.json \
    OER12_LOGDIR=/logs/verifier \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /verifier

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
#
# This brings in the grader, the verifier's OWN copies of the harness, the schema,
# the model, the substrate declaration and the frozen training recipe, the stream a
# policy pays to read under probe/, the graded split under holdout/, and the
# reference policy under private/. The verifier reads its own copies of all of it:
# nothing on the agent surface, and nothing the submission wrote, is imported or
# executed here.
COPY tests/ /verifier/tests/

# The training stream, the same bytes the agent has. It is CLEAN here too: the
# declared fault is applied at run time by harness.degrade() from a ramp start this
# verifier draws, so the corrupted stream never exists on disk on either surface.
COPY environment/data/train_stream.bin /verifier/tests/corpus/train_stream.bin

# solution/ is deliberately NOT copied. The grader imports only tests/harness.py and
# tests/policy_schema.py, and baking the oracle into the verifier image would put
# the reference solver inside the process that grades submissions.

# THE IDENTITY ASSERTION.
#
# The agent is told that a policy designed locally with probe_local.py is replayed
# at grading time by the same code. That promise is only worth anything if the two
# copies cannot drift, so it is enforced here at BUILD time rather than asserted in
# prose: six files must be byte-identical across the agent surface and this one, and
# a mismatch fails the image build instead of silently producing a verifier that
# grades something the agent could not predict.
#
# The mirror is deleted in the same layer it is created in, so these bytes do not
# survive into the shipped verifier image.
COPY environment/harness.py              /verifier/mirror/harness.py
COPY environment/policy_schema.py        /verifier/mirror/policy_schema.py
COPY environment/model/nanogpt.py        /verifier/mirror/model/nanogpt.py
COPY environment/frozen/task_spec.json   /verifier/mirror/frozen/task_spec.json
COPY environment/frozen/train_recipe.json /verifier/mirror/frozen/train_recipe.json
COPY environment/default_policy.json     /verifier/mirror/default_policy.json
RUN set -eu; \
    for f in harness.py policy_schema.py model/nanogpt.py frozen/task_spec.json \
             frozen/train_recipe.json default_policy.json; do \
        cmp "/verifier/tests/$f" "/verifier/mirror/$f" \
          || { echo "DRIFT: tests/$f differs from environment/$f"; exit 1; }; \
    done; \
    echo "agent/verifier surface identity OK"; \
    rm -rf /verifier/mirror

# Build-time conformance, CPU only. Both verifier-only splits must be present and
# long enough, the reference policy must satisfy the SAME public schema a submission
# does, and the default anchor must parse. A verifier that cannot replay an anchor
# is a verifier that would have to invent one, so this fails at build.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/verifier/tests")
import harness, policy_schema
spec = harness.load_spec()
ev = spec["evaluation"]
need_train = spec["budget"]["total_train_tokens"] + 1
need_eval = ev["probe_batches"] * ev["tokens_per_batch"] + 1
train = harness.load_shard("/verifier/tests/corpus/train_stream.bin")
probe = harness.load_shard("/verifier/tests/probe/probe_slice.bin")
holdout = harness.load_shard("/verifier/tests/holdout/val_slice.bin")
assert train.size >= need_train, f"train stream {train.size} < {need_train}"
assert probe.size >= need_eval, f"probe slice {probe.size} < {need_eval}"
assert holdout.size >= need_eval, f"holdout slice {holdout.size} < {need_eval}"
assert tuple(harness.candidate_steps(spec)) == policy_schema.candidate_steps()
policy_schema.load("/verifier/tests/default_policy.json")
policy_schema.load("/verifier/tests/private/reference_policy.json")
print("verifier conformance OK:", train.size, "train tokens,", holdout.size, "holdout tokens")
PY

# The reward carriers. A DIRECTORY, never a VOLUME: an anonymous volume declared at
# /logs/verifier shadows the /logs bind mount the runner supplies, and every carrier
# written under it is discarded with the container.
RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
