# Verifier environment for OER-09.
#
# Same pinned digest as the agent image, and pinned for the same reason. The two
# surfaces must run the same kernels or the anchors measured here would not be
# comparable to anything the agent measured.
#
# NOTE: this image declares NO VOLUME. A `VOLUME ["/logs/verifier"]` line would
# create an anonymous volume that shadows the /logs bind mount and silently
# discard the reward carriers.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER_SLOT=OER-09 \
    OER09_SUBMISSION=/workspace/submission/filter.json \
    OER09_LOGDIR=/logs/verifier \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /verifier

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/.
#
# This brings in the grader, the verifier's OWN copies of the harness, the filter
# schema, the model and the substrate declaration, the held-out split under
# holdout/, and the reference chain under private/. The verifier reads its own
# copies of all of it: nothing on the agent surface, and nothing the submission
# wrote, is imported or executed here.
COPY tests/ /verifier/tests/

# The pool, the same bytes the agent reasons about. The verifier must train the
# control, the reference and the submission on blocks drawn from the identical
# pool or the three losses would not be comparable.
COPY environment/pool/ /verifier/tests/pool/

# solution/ is deliberately NOT copied. The grader imports only tests/harness.py
# and tests/filter_schema.py, and baking the oracle into the verifier image would
# put the reference solver inside the process that grades submissions.

# THE IDENTITY ASSERTION.
#
# The agent is told that a chain measured locally with train_local.py is measured
# at grading time by the same code. That promise is only worth anything if the two
# copies cannot drift, so it is enforced here at BUILD time rather than asserted
# in prose: five files must be byte-identical across the agent surface and this
# one, and a mismatch fails the image build instead of silently producing a
# verifier that grades something the agent could not measure.
#
# The mirror is deleted in the same layer it is created in, so these bytes do not
# survive into the shipped verifier image.
COPY environment/harness.py            /verifier/mirror/harness.py
COPY environment/filter_schema.py      /verifier/mirror/filter_schema.py
COPY environment/model/nanogpt.py      /verifier/mirror/model/nanogpt.py
COPY environment/frozen/task_spec.json /verifier/mirror/frozen/task_spec.json
COPY environment/default_filter.json   /verifier/mirror/default_filter.json
RUN set -eu; \
    for f in harness.py filter_schema.py model/nanogpt.py frozen/task_spec.json default_filter.json; do \
        cmp "/verifier/tests/$f" "/verifier/mirror/$f" \
          || { echo "DRIFT: tests/$f differs from environment/$f"; exit 1; }; \
    done; \
    echo "agent/verifier surface identity OK"; \
    rm -rf /verifier/mirror

# Build-time conformance, CPU only. The held-out split must be present and long
# enough; the pool and its register must agree; the reference chain must satisfy
# the SAME public schema a submission does and must leave at least the budget
# standing; and the shipped control must be the fail-open chain it claims to be.
# A verifier that cannot compute an anchor is a verifier that would have to invent
# one, so this fails at build.
RUN python3 - <<'PY'
import sys
sys.path.insert(0, "/verifier/tests")
import harness, filter_schema
spec = harness.load_spec()
need_eval = spec["evaluation"]["tokens"] + 1
holdout = harness.load_shard("/verifier/tests/holdout/val_slice.bin")
assert holdout.size >= need_eval, f"holdout slice {holdout.size} < {need_eval}"
blocks = harness.load_shard("/verifier/tests/pool/pool_blocks.bin")
assert blocks.size == spec["pool"]["blocks"] * spec["pool"]["block_tokens"]
register = filter_schema.load_register("/verifier/tests/pool/source_register.jsonl")
assert len(register) == spec["pool"]["blocks"]
budget = spec["budget"]["blocks"]
control = filter_schema.load("/verifier/tests/default_filter.json")
reference = filter_schema.load("/verifier/tests/private/reference_filter.json")
c_ids, c_rep = filter_schema.apply(control, register)
r_ids, r_rep = filter_schema.apply(reference, register)
assert len(c_ids) >= budget, f"the control anchor admits {len(c_ids)} < {budget}"
assert len(r_ids) >= budget, f"the reference anchor admits {len(r_ids)} < {budget}"
assert c_rep["pool_passed_through_unchanged"], "the control is supposed to be the uncurated arm"
assert r_rep["rules_unresolved"] == 0, "every predicate of the reference chain must resolve"
assert len(r_ids) < len(c_ids), "the reference chain is supposed to remove blocks"
print("verifier conformance OK:", holdout.size, "holdout tokens,",
      len(c_ids), "control blocks,", len(r_ids), "reference blocks")
PY

# The reward carriers.
RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
