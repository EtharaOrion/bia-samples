# Verifier environment.
#
# Same pinned digest as the agent image, and pinned for the same reason. The two
# surfaces must run the same kernels or the anchors measured here would not be
# comparable to anything the agent measured.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-15 \
    OER_SUBMISSION=/workspace/submission/shape.json \
    OER_LOGDIR=/logs/verifier \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /verifier

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
COPY tests/ /verifier/tests/

# The training corpus, the same bytes the agent trains on. The verifier must train the
# default anchor, the reference anchor and the submission on the identical token
# stream or the three losses would not be comparable.
COPY environment/data/train_slice.bin /verifier/tests/corpus/train_slice.bin

# solution/ is deliberately NOT copied. The grader imports only tests/harness.py and
# tests/shape_schema.py, and baking the oracle into the verifier image would put the
# reference solver inside the process that grades submissions.
#
# NOTE ON VOLUMES: this file declares none, and specifically does not declare
# /logs/verifier. A VOLUME at that path creates an anonymous volume that is layered ON
# TOP of a /logs bind mount, so the reward carriers are written successfully into a
# volume nobody reads and the run looks like an infrastructure fault instead of the
# score the grader computed.

# THE IDENTITY ASSERTION, enforced at BUILD time rather than asserted in prose.
COPY environment/harness.py            /verifier/mirror/harness.py
COPY environment/shape_schema.py       /verifier/mirror/shape_schema.py
COPY environment/model/nanogpt.py      /verifier/mirror/model/nanogpt.py
COPY environment/frozen/task_spec.json /verifier/mirror/frozen/task_spec.json
COPY environment/default_shape.json    /verifier/mirror/default_shape.json
RUN set -eu; \
    for f in harness.py shape_schema.py model/nanogpt.py frozen/task_spec.json default_shape.json; do \
        cmp "/verifier/tests/$f" "/verifier/mirror/$f" \
          || { echo "DRIFT: tests/$f differs from environment/$f"; exit 1; }; \
    done; \
    echo "agent/verifier surface identity OK"; \
    rm -rf /verifier/mirror

# Build-time conformance, CPU only. The held-out split must be present and long
# enough, both anchor shapes must satisfy the SAME public schema a submission does,
# and the two anchors must be DIFFERENT shapes -- a verifier whose floor and bar were
# the same shape would measure a zero span and refuse every submission.
#
# The parameter arithmetic the band is enforced with is checked against the REAL
# constructed module for every shape on the grid, here, at build.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/verifier/tests")
import harness, shape_schema
spec = harness.load_spec()
budget = spec["shape_budget"]
b = spec["budget"]
train = harness.load_shard("/verifier/tests/corpus/train_slice.bin")
holdout = harness.load_shard("/verifier/tests/holdout/val_slice.bin")
assert train.size >= b["total_train_tokens"] + 1, f"train slice {train.size} too short"
assert holdout.size >= spec["evaluation"]["tokens"] + 1, f"holdout slice {holdout.size} too short"
default = shape_schema.load("/verifier/tests/default_shape.json", budget)
reference = shape_schema.load("/verifier/tests/private/reference_shape.json", budget)
assert default["shape"] != reference["shape"], "the two anchors are the same shape"

nanogpt = harness.load_model_module()
checked = 0
for L in shape_schema.ALLOWED_NUM_LAYERS:
    for d in shape_schema.ALLOWED_MODEL_DIM:
        for h in shape_schema.ALLOWED_HEAD_DIM:
            if d % h:
                continue
            for m in shape_schema.ALLOWED_MLP_RATIO:
                model = nanogpt.GPT(vocab_size=int(spec["architecture_frozen"]["vocab_size"]),
                                    num_layers=L, model_dim=d, head_dim=h, mlp_ratio=m)
                real = sum(p.numel() for n, p in model.named_parameters()
                           if not n.startswith("embed") and not n.startswith("proj"))
                assert real == shape_schema.non_embedding_parameters(L, d, h, m), \
                    f"parameter accounting disagrees at L={L} d={d} h={h} m={m}"
                checked += 1
print("verifier conformance OK:", train.size, "train tokens,", holdout.size,
      "holdout tokens,", checked, "grid shapes accounted")
for label, doc in (("default", default), ("reference", reference)):
    print(label, "anchor:", doc["shape"], doc["_measured_non_embedding_parameters"],
          "non-embedding parameters")
PY

# The reward carriers.
RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
