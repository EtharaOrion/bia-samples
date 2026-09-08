# Verifier environment.
#
# Same pinned digest as the agent image, and pinned for the same reason. The two
# surfaces must run the same kernels or the anchors measured here would not be
# comparable to anything the agent measured.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER08_SLOT=OER-08 \
    OER08_SUBMISSION=/workspace/submission/plan.json \
    OER08_LOGDIR=/logs/verifier \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /verifier

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
#
# This brings in the grader, the verifier's OWN copies of the harness, the schema,
# the model, the substrate declaration and the frozen training recipe, the held-out
# split under holdout/, and the reference plan under private/. The verifier reads its
# own copies of all of it: nothing on the agent surface, and nothing the submission
# wrote, is imported or executed here.
COPY tests/ /verifier/tests/

# The pool, the same bytes the agent has. The verifier must assemble the default
# anchor, the reference anchor and the submission out of the identical sources or the
# three losses would not be comparable.
COPY environment/data/sources/ /verifier/tests/corpus/sources/

# solution/ is deliberately NOT copied. The grader imports only tests/harness.py and
# tests/plan_schema.py, and baking the oracle into the verifier image would put the
# reference solver inside the process that grades submissions.

# THE IDENTITY ASSERTION.
#
# The agent is told that a plan measured locally with probe_local.py is measured at
# grading time by the same code. That promise is only worth anything if the two
# copies cannot drift, so it is enforced here at BUILD time rather than asserted in
# prose: five files must be byte-identical across the agent surface and this one, and
# a mismatch fails the image build instead of silently producing a verifier that
# grades something the agent could not measure.
#
# The mirror is deleted in the same layer it is created in, so these bytes do not
# survive into the shipped verifier image.
COPY environment/harness.py               /verifier/mirror/harness.py
COPY environment/plan_schema.py           /verifier/mirror/plan_schema.py
COPY environment/model/nanogpt.py         /verifier/mirror/model/nanogpt.py
COPY environment/frozen/task_spec.json    /verifier/mirror/frozen/task_spec.json
COPY environment/frozen/train_recipe.json /verifier/mirror/frozen/train_recipe.json
RUN set -eu; \
    for f in harness.py plan_schema.py model/nanogpt.py frozen/task_spec.json \
             frozen/train_recipe.json; do \
        cmp "/verifier/tests/$f" "/verifier/mirror/$f" \
          || { echo "DRIFT: tests/$f differs from environment/$f"; exit 1; }; \
    done; \
    echo "agent/verifier surface identity OK"; \
    rm -rf /verifier/mirror

# Build-time conformance, CPU only. The held-out split must be present and long
# enough, the whole pool must be here, and BOTH anchor plans must satisfy the SAME
# public schema a submission does and assemble to exactly the frozen budget. A
# verifier that cannot compute an anchor is a verifier that would have to invent one,
# so this fails at build.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/verifier/tests")
import harness, plan_schema
spec = harness.load_spec()
pool = harness.load_pool(spec, "/verifier/tests/corpus/sources")
assert len(pool) == len(spec["pool"]["sources"]), "the pool on disk is not the declared pool"
holdout = harness.load_shard("/verifier/tests/holdout/val_slice.bin")
assert holdout.size >= spec["evaluation"]["tokens"] + 1, f"holdout slice {holdout.size} too short"
need = spec["budget"]["total_train_tokens"] + 1
for name in ("default_plan.json", "private/reference_plan.json"):
    plan = plan_schema.load("/verifier/tests/" + name, spec)
    stream = harness.assemble(plan, spec, pool)
    assert stream.size == need, f"{name} assembles {stream.size} tokens, {need} are needed"
print("verifier conformance OK:", len(pool), "sources,", holdout.size, "holdout tokens")
PY

# The reward carriers. A DIRECTORY, never a VOLUME: an anonymous volume declared at
# /logs/verifier shadows the /logs bind mount the runner supplies, and every carrier
# written under it is discarded with the container.
RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
