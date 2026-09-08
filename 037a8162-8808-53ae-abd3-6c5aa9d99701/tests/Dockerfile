# Verifier environment for OER-01.
#
# Same pinned digest as the agent image, and pinned for the same reason.
#
# NOTE: this image declares NO VOLUME. A `VOLUME ["/logs/verifier"]` line would
# create an anonymous volume that shadows the /logs bind mount and silently discard
# the reward carriers.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-01 \
    OER01_SUBMISSION=/workspace/submission/recipe.json \
    OER01_LOGDIR=/logs/verifier \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /verifier

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
COPY tests/ /verifier/tests/
COPY environment/data/train_slice.bin /verifier/tests/corpus/train_slice.bin

# solution/ is deliberately NOT copied.

# THE IDENTITY ASSERTION. Seven files must be byte-identical across the two
# surfaces or this image does not build.
COPY environment/harness.py             /verifier/mirror/harness.py
COPY environment/recipe_schema.py       /verifier/mirror/recipe_schema.py
COPY environment/records.py             /verifier/mirror/records.py
COPY environment/model/nanogpt.py       /verifier/mirror/model/nanogpt.py
COPY environment/frozen/task_spec.json  /verifier/mirror/frozen/task_spec.json
COPY environment/default_recipe.json    /verifier/mirror/default_recipe.json
COPY environment/published_records.json /verifier/mirror/published_records.json
RUN set -eu; \
    for f in harness.py recipe_schema.py records.py model/nanogpt.py \
             frozen/task_spec.json default_recipe.json published_records.json; do \
        cmp "/verifier/tests/$f" "/verifier/mirror/$f" \
          || { echo "DRIFT: tests/$f differs from environment/$f"; exit 1; }; \
    done; \
    echo "agent/verifier surface identity OK"; \
    rm -rf /verifier/mirror

# Build-time conformance, CPU only. The held-out split must be long enough for the
# grid window, both anchors must satisfy the SAME public schema a submission does,
# and the reference must NOT itself be a published record -- a bar that is excluded
# from the submission space is a bar nothing can legally reach.
RUN python3 - <<'PY'
import sys
sys.path.insert(0, "/verifier/tests")
import harness, recipe_schema, records
spec = harness.load_spec()
ev = spec["evaluation"]
train = harness.load_shard("/verifier/tests/corpus/train_slice.bin")
holdout = harness.load_shard("/verifier/tests/holdout/val_slice.bin")
assert train.size >= spec["budget"]["total_train_tokens"] + 1
assert holdout.size >= ev["grid_tokens"] + 1, "holdout is too short for the grid window"
assert holdout.size >= ev["tokens"] + 1, "holdout is too short for the full window"
default = recipe_schema.load("/verifier/tests/default_recipe.json")
reference = recipe_schema.load("/verifier/tests/private/reference_recipe.json")
corpus = records.load_records("/verifier/tests/published_records.json")
assert corpus, "the published-record corpus is empty"
assert records.find_replay(reference, corpus) is None, \
    "the private reference is itself a published record"
assert records.canonical(default) != records.canonical(reference), \
    "the two anchors are the same recipe; there would be no span to scale between"
print("verifier conformance OK:", train.size, "train tokens,", len(corpus), "published records")
PY

RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
