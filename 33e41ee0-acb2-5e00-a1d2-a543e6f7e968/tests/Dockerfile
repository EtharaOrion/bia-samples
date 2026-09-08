# Verifier environment for OER-04.
#
# Same pinned digest as the agent image, and pinned for the same reason.
#
# NOTE: this image declares NO VOLUME. A `VOLUME ["/logs/verifier"]` line would
# create an anonymous volume that shadows the /logs bind mount and silently
# discard the reward carriers.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-04 \
    OER04_SUBMISSION=/workspace/submission/shape.json \
    OER04_LOGDIR=/logs/verifier \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /verifier

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
COPY tests/ /verifier/tests/

# The training corpus, the same bytes the agent trains on.
COPY environment/data/train_slice.bin /verifier/tests/corpus/train_slice.bin

# solution/ is deliberately NOT copied.

# THE IDENTITY ASSERTION. Five files must be byte-identical across the two
# surfaces or this image does not build. The mirror is deleted in the layer that
# creates it, so these bytes do not survive into the shipped verifier image.
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

# Build-time conformance, CPU only. Both anchors must satisfy the SAME public
# schema a submission does, both must land inside the parameter band when actually
# instantiated, and they must be different shapes. A verifier that cannot compute
# an anchor is a verifier that would have to invent one, so this fails at build.
RUN python3 - <<'PY'
import sys
sys.path.insert(0, "/verifier/tests")
import harness, shape_schema
spec = harness.load_spec()
bounds, band = spec["shape_bounds"], spec["parameter_band"]
train = harness.load_shard("/verifier/tests/corpus/train_slice.bin")
holdout = harness.load_shard("/verifier/tests/holdout/val_slice.bin")
assert train.size >= spec["budget"]["total_train_tokens"] + 1
assert holdout.size >= spec["evaluation"]["tokens"] + 1
control = shape_schema.load("/verifier/tests/default_shape.json", bounds)
reference = shape_schema.load("/verifier/tests/private/reference_shape.json", bounds)
counts = {}
for name, shape in (("control", control), ("reference", reference)):
    model, n = harness.build_model(shape, spec, device="cpu")
    shape_schema.parameter_band_check(n, band)
    counts[name] = n
    del model
assert {k: v for k, v in control.items() if k != "notes"} != \
       {k: v for k, v in reference.items() if k != "notes"}, \
    "the two anchors are the same shape; there would be no span to scale between"
print("verifier conformance OK:", train.size, "train tokens,", counts)
PY

RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
