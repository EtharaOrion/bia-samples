# The verifier surface. Isolated from the agent surface and pinned to the same
# digest, so the two differ by policy and by nothing else.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Egress DENIED at this surface. The solving agent has open egress; the verifier
# does not. That asymmetry is bound in task.toml and repeated here so the image
# and the contract cannot drift apart silently.
ENV OER20_SLOT=OER-20 \
    OER20_VERIFIER_EGRESS=denied \
    OER20_CHECKPOINT=/checkpoint/nanogpt_fp32.pt \
    OER20_HELDOUT=/verifier/heldout \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /verifier

# The build context is the BUNDLE ROOT, not tests/. tests/test.sh resolves its own
# bundle root as the parent of the directory it was laid down in and then reaches
# for two trees beside itself: ${BUNDLE_DIR}/environment, which tests/runner.py
# opens for the frozen declarations it digests, and ${BUNDLE_DIR}/solution, whose
# reference tests/grade.py imports for the reference digest it stamps into the
# verdict. A tests-only context builds an image that carries neither, and both
# reads then die inside a container that had already started: runner.py on the
# frozen inputs and grade.py on FileNotFoundError for the reference, at which
# point the exit trap writes verifier-aborted-before-grading. Whatever the entry
# point reads has to be in the image that runs it.
COPY tests/test.sh /verifier/tests/test.sh
COPY tests/runner.py /verifier/tests/runner.py
COPY tests/decoder.py /verifier/tests/decoder.py
COPY tests/grade.py /verifier/tests/grade.py
COPY tests/checkers.py /verifier/tests/checkers.py
COPY tests/checkers.yaml /verifier/tests/checkers.yaml
COPY tests/calibration.py /verifier/tests/calibration.py
COPY tests/emit_reward.py /verifier/tests/emit_reward.py
COPY tests/test_output.py /verifier/tests/test_output.py
COPY tests/rubrics.jsonl /verifier/tests/rubrics.jsonl

# The frozen declarations tests/runner.py digests at the open and at the close of
# every graded run, so that tests/checkers.py::check_frozen_inputs_intact grades a
# pair of readings taken over bytes this image owns.
COPY environment/ /verifier/environment/

# The private reference. tests/grade.py imports it for REFERENCE_SHA256 alone and
# never for a number: nothing here is graded against it, and it reaches no
# agent-visible container, because this image is the verifier's and is built from
# a definition the solver never runs.
COPY solution/reference.py /verifier/solution/reference.py

# The verifier's OWN copy of the frozen architecture declaration, laid down BESIDE
# tests/decoder.py, which resolves it relative to itself. tests/decoder.py reads
# this copy and never the copy inside an agent-visible container, so editing
# environment/nanogpt_substrate.json moves a digest the checkers grade and moves
# nothing the model is built from. It sits inside tests/ rather than one level up
# because an image-baked asset addressed by an absolute path reads, to anything
# that derives mount points from these frozen bytes, as a directory this container
# expects from outside: /verifier was bound from the host on that reading and the
# entry point below was replaced by an empty mount.
COPY tests/nanogpt_substrate.json /verifier/tests/nanogpt_substrate.json

RUN chmod +x /verifier/tests/test.sh

# The held-out FineWeb validation slice. Mounted here and NOWHERE else. It is the
# only corpus the graded perplexity is computed over, the evaluation order is
# derived from the digest of its manifest, and no agent-visible container carries
# it or names its points. If this mount is absent the harness produces zero
# evaluation points and the metric is undefined, which is the intended behaviour
# and not a degraded one.
VOLUME ["/verifier/heldout"]

# The frozen checkpoint, the same parameters the agent surface was handed.
VOLUME ["/checkpoint"]

# The bound reward carrier and its companion score document.
VOLUME ["/logs/verifier"]

ENTRYPOINT ["/verifier/tests/test.sh"]
