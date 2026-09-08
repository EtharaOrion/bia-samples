# Verifier environment.
#
# Same pinned digest as the agent image, and pinned for the same reason. The two
# surfaces must run the same kernels or the anchors measured here would not be
# comparable to anything the agent measured.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-14 \
    OER_SUBMISSION=/workspace/submission/policy.json \
    OER_LOGDIR=/logs/verifier \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /verifier

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
#
# This brings in the grader, the verifier's OWN copies of the harness, the schema,
# the model and the substrate declaration, the held-out split under holdout/, and
# the reference policy under private/. The verifier reads its own copies of all of it:
# nothing on the agent surface, and nothing the submission wrote, is imported or
# executed here.
COPY tests/ /verifier/tests/

# The training corpus, the same bytes the agent trains on. The verifier must train
# the default anchor, the reference anchor and the submission on the identical token
# stream or the three losses would not be comparable.
COPY environment/data/train_slice.bin /verifier/tests/corpus/train_slice.bin

# solution/ is deliberately NOT copied. The grader imports only tests/harness.py and
# tests/policy_schema.py, and baking the oracle into the verifier image would put
# the reference solver inside the process that grades submissions.

# THE IDENTITY ASSERTION.
#
# The agent is told that a plan measured locally with train_local.py is measured at
# grading time by the same code. That promise is only worth anything if the two
# copies cannot drift, so it is enforced here at BUILD time rather than asserted in
# prose: five files must be byte-identical across the agent surface and this one, and
# a mismatch fails the image build instead of silently producing a verifier that
# grades something the agent could not measure.
#
# The mirror is deleted in the same layer it is created in, so these bytes do not
# survive into the shipped verifier image.
COPY environment/harness.py            /verifier/mirror/harness.py
COPY environment/policy_schema.py     /verifier/mirror/policy_schema.py
COPY environment/model/nanogpt.py      /verifier/mirror/model/nanogpt.py
COPY environment/frozen/task_spec.json /verifier/mirror/frozen/task_spec.json
COPY environment/default_policy.json  /verifier/mirror/default_policy.json
RUN set -eu; \
    for f in harness.py policy_schema.py model/nanogpt.py frozen/task_spec.json default_policy.json; do \
        cmp "/verifier/tests/$f" "/verifier/mirror/$f" \
          || { echo "DRIFT: tests/$f differs from environment/$f"; exit 1; }; \
    done; \
    echo "agent/verifier surface identity OK"; \
    rm -rf /verifier/mirror

# Build-time conformance, CPU only. The held-out split must be present and long
# enough, and BOTH anchor policies must satisfy the SAME public schema a submission
# does. A verifier that cannot compute an anchor is a verifier that would have to
# invent one, so this fails at build.
#
# It also asserts that the two anchors are DIFFERENT policies. A verifier whose floor
# and bar were the same policy would measure a zero span and refuse every submission,
# and that would surface as a hard-to-read refusal at grade time rather than here.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/verifier/tests")
import harness, policy_schema
spec = harness.load_spec()
total = int(spec["budget"]["total_train_tokens"])
train = harness.load_shard("/verifier/tests/corpus/train_slice.bin")
holdout = harness.load_shard("/verifier/tests/holdout/val_slice.bin")
assert train.size >= total + 1, f"train slice {train.size} < {total + 1}"
assert holdout.size >= spec["evaluation"]["tokens"] + 1, f"holdout slice {holdout.size} too short"
default = policy_schema.load("/verifier/tests/default_policy.json")
reference = policy_schema.load("/verifier/tests/private/reference_policy.json")
assert (default["decay"], default["objective"]) != (reference["decay"], reference["objective"]), \
    "the two anchors are the same policy"
print("default anchor :", default["decay"], default["objective"])
print("reference anchor:", reference["decay"], reference["objective"])
print("verifier conformance OK:", train.size, "train tokens,", holdout.size, "holdout tokens")
PY

# The reward carriers.
RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
