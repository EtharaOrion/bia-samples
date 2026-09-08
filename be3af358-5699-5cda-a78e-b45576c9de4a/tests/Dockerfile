# Verifier environment for slot OER-22.
#
# Same pinned digest as the agent image, and pinned for the same reason. The two surfaces
# must run the same kernels, or the snapshots the agent reasoned about locally and the
# snapshots this verifier grades would not be the same objects.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-22 \
    OER_SUBMISSION=/workspace/submission/selection.json \
    OER_LOGDIR=/logs/verifier \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /verifier

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
#
# This brings in the grader, the verifier's OWN copies of the harness, the schema, the
# model and the substrate declaration, the held-out split under holdout/, and the
# reference selection under private/. The verifier reads its own copies of all of it:
# nothing on the agent surface, and nothing the submission wrote, is imported or executed
# here.
COPY tests/ /verifier/tests/

# The training corpus, the same bytes the agent's local run consumes. The frozen run has
# to walk the identical token stream or its snapshots would not be the snapshots the
# agent selected among.
COPY environment/data/train_slice.bin /verifier/tests/corpus/train_slice.bin

# solution/ is deliberately NOT copied. The grader imports only tests/harness.py and
# tests/selection_schema.py, and baking the oracle into the verifier image would put the
# reference solver inside the process that grades submissions.

# THE IDENTITY ASSERTION.
#
# The agent is told that a selection measured locally with train_local.py is combined and
# evaluated at grading time by the same code over the same frozen run. That promise is
# only worth anything if the two copies cannot drift, so it is enforced here at BUILD time
# rather than asserted in prose: four files must be byte-identical across the agent
# surface and this one, and a mismatch fails the image build instead of silently producing
# a verifier that grades something the agent could not measure.
#
# The mirror is deleted in the same layer it is created in, so these bytes do not survive
# into the shipped verifier image.
COPY environment/harness.py            /verifier/mirror/harness.py
COPY environment/selection_schema.py   /verifier/mirror/selection_schema.py
COPY environment/model/nanogpt.py      /verifier/mirror/model/nanogpt.py
COPY environment/frozen/task_spec.json /verifier/mirror/frozen/task_spec.json
COPY environment/default_selection.json /verifier/mirror/default_selection.json
RUN set -eu; \
    for f in harness.py selection_schema.py model/nanogpt.py frozen/task_spec.json default_selection.json; do \
        cmp "/verifier/tests/$f" "/verifier/mirror/$f" \
          || { echo "DRIFT: tests/$f differs from environment/$f"; exit 1; }; \
    done; \
    echo "agent/verifier surface identity OK"; \
    rm -rf /verifier/mirror

# Build-time conformance, CPU only. The held-out split must be present and long enough,
# and BOTH anchor selections must satisfy the SAME public schema a submission does --
# including the storage budget, so the high anchor can never be a selection a submission
# is forbidden to write. A verifier that cannot compute an anchor is a verifier that would
# have to invent one, so this fails at build.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/verifier/tests")
import harness, selection_schema
spec = harness.load_spec()
need_train = spec["budget"]["total_train_tokens"] + 1
need_eval = spec["evaluation"]["tokens"] + 1
train = harness.load_shard("/verifier/tests/corpus/train_slice.bin")
holdout = harness.load_shard("/verifier/tests/holdout/val_slice.bin")
assert train.size >= need_train, f"train slice {train.size} < {need_train}"
assert holdout.size >= need_eval, f"holdout slice {holdout.size} < {need_eval}"
budget = int(spec["snapshots"]["storage_budget"])
for path in ("/verifier/tests/default_selection.json",
             "/verifier/tests/private/reference_selection.json"):
    loaded = selection_schema.load(path, spec)
    assert len(loaded["keep"]) <= budget, f"{path} exceeds the storage budget"
print("verifier conformance OK:", train.size, "train tokens,", holdout.size, "holdout tokens")
PY

# The reward carriers.
#
# THERE IS DELIBERATELY NO `VOLUME ["/logs/verifier"]` HERE.
# /logs is a BIND MOUNT the runtime supplies and one of only two paths this container
# shares with the agent's. Declaring it a VOLUME creates an ANONYMOUS volume mounted OVER
# the bind at exactly that path, so every carrier written here would land in a scratch
# volume discarded when the container exits: the reward computed correctly and thrown
# away, and the runtime reading silence. mkdir is all that is wanted.
RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
