# Isolated verifier environment for OER-16.
#
# Same pinned digest as the agent environment, and pinned for the same reason: a
# floating tag validates cleanly and silently resolves to different bytes.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The verifier's network policy is DENIED at run time. This differs from the agent
# surface by design: the agent has open egress so it can fetch packages, datasets and
# models; the verifier fetches nothing once the run starts. The asymmetry is bound in
# task.toml.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER16_VERIFIER_EGRESS=denied \
    OER16_SLOT=OER-16 \
    OER16_SUBMISSION=/workspace/submission/allocation.json \
    OER16_LOGDIR=/logs/verifier \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    TRITON_CACHE_DIR=/tmp/triton

WORKDIR /verifier

# THE BUILD CONTEXT IS THE BUNDLE ROOT, NOT tests/. tests/grade.py imports
# tests/harness.py and tests/allocation_schema.py and reads tests/frozen/,
# tests/corpus/, tests/holdout/ and tests/private/. A tests-only context builds an
# image that carries none of the bundle-root bytes the identity assertion below
# needs, and the first read dies on FileNotFoundError inside a container that had
# already started, at which point test.sh writes reward 0.0 with reason
# grading-chain-failed and nothing is ever graded. Whatever the entry point reads has
# to be in the image that runs it.
#
# The grading tree is never mounted on the agent surface: Harbor assembles a
# non-oracle agent surface from task.toml, instruction.md and environment/ only, so
# tests/ and solution/ are absent from it by construction rather than removed by a
# filter that could be misconfigured.
COPY tests/ /verifier/tests/

# The training corpus, the same bytes the agent trains on. All three runs of a
# grading pass must draw from the identical token stream or their losses would not be
# comparable. Staged by COPY rather than fetched: the previous form of this file ran
# `cached_fineweb10B.py 20` at build time, which needs twenty shards of build egress
# for a task that reads one slice, and a build that depends on the network is a build
# that stops reproducing the day the network does.
COPY environment/data/train_slice.bin /verifier/tests/corpus/train_slice.bin

# solution/ is deliberately NOT copied. tests/ reads no file under it, and baking
# solution/solve.sh in would put the reference solver inside the image the verifier
# runs, which is the reference's own work and not a submission.

# THE IDENTITY ASSERTION.
#
# The agent is told that an allocation measured locally with train_local.py is
# measured at grading time by the same code, and -- more sharply here than in most
# slots -- that the FLOP arithmetic which BOUNDS a submission on the agent surface is
# the same arithmetic that CHARGES it here. That promise is only worth anything if
# the two copies cannot drift, so it is enforced at BUILD time rather than asserted
# in prose: five files must be byte-identical across the two surfaces, and a mismatch
# fails the image build instead of silently producing a verifier that grades
# something the agent could not measure.
#
# The mirror is deleted in the same layer it is created in, so these bytes do not
# survive into the shipped verifier image.
COPY environment/harness.py               /verifier/mirror/harness.py
COPY environment/allocation_schema.py     /verifier/mirror/allocation_schema.py
COPY environment/model/nanogpt.py         /verifier/mirror/model/nanogpt.py
COPY environment/frozen/task_spec.json    /verifier/mirror/frozen/task_spec.json
COPY environment/frozen/train_recipe.json /verifier/mirror/frozen/train_recipe.json
RUN set -eu; \
    for f in harness.py allocation_schema.py model/nanogpt.py frozen/task_spec.json \
             frozen/train_recipe.json; do \
        cmp "/verifier/tests/$f" "/verifier/mirror/$f" \
          || { echo "DRIFT: tests/$f differs from environment/$f"; exit 1; }; \
    done; \
    echo "agent/verifier surface identity OK"; \
    rm -rf /verifier/mirror

# Build-time conformance, CPU only on purpose: the accelerator is a run-time resource
# and a build that depended on it would fail on a machine without one. The held-out
# split must be present and long enough, the corpus must be long enough for the
# largest number of micro-batches any admissible allocation can buy at one epoch, and
# BOTH anchor allocations must satisfy the SAME public schema a submission does and
# fit the SAME declared FLOP budget. A verifier that cannot compute an anchor is a
# verifier that would have to invent one, so this fails at build.
RUN python3 - <<'PY'
import sys, json
sys.path.insert(0, "/verifier/tests")
import harness, allocation_schema
spec = harness.load_spec()
train = harness.load_shard("/verifier/tests/corpus/train_slice.bin")
holdout = harness.load_shard("/verifier/tests/holdout/val_slice.bin")
assert holdout.size >= spec["evaluation"]["tokens"] + 1, f"holdout slice {holdout.size} too short"
assert train.size >= spec["data"]["corpus_tokens"], f"corpus {train.size} too short"
for name in ("default_allocation.json", "private/reference_allocation.json"):
    alloc = allocation_schema.load("/verifier/tests/" + name, spec, harness)
    spent = alloc["micro_steps"] * harness.micro_batch_flops(spec, alloc["model"])
    assert spent <= int(spec["budget"]["flop_budget"]), f"{name} overspends the budget"
    print(name, alloc["model"], alloc["micro_steps"], "steps,",
          round(100.0 * spent / int(spec["budget"]["flop_budget"]), 1), "% of budget")
print("verifier conformance OK:", train.size, "train tokens,", holdout.size, "holdout tokens")
PY

# The reward carriers. /logs/verifier/reward.txt takes the bare float and
# /logs/verifier/score.json the reason and the metric block.
#
# A DIRECTORY, never a VOLUME. `VOLUME ["/logs/verifier"]` makes Docker create an
# ANONYMOUS volume at that path on every run, and an anonymous volume there SHADOWS
# the /logs bind mount the runner supplies: the trap runs, the grader writes, and
# every carrier lands in a throwaway that is discarded with the container, so the run
# reports no reward at all while every line of the emit path executed correctly.
RUN mkdir -p /logs/verifier /logs/agent

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
