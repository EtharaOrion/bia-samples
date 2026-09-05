# Agent environment for OER-16.
#
# Pinned by @sha256: and never by a floating tag. Harbor accepts any string for
# an image reference and does not itself require a digest, so a tag would
# validate cleanly while silently resolving to different bytes on a later pull.
# The digest requirement is this contract's, and this line is where it is kept.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The solving agent has OPEN egress inside the sandbox and may fetch its own
# packages, datasets and models. Nothing beyond the corpus below is pre-fetched
# here on the agent's behalf, so what the agent chose to install stays visible
# in its trajectory.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OER16_SLOT=OER-16 \
    OER16_TRAIN_DATA=/app/data/fineweb10B

WORKDIR /app

# The frozen half of the task: the model, optimizer, compute-budget and data
# declarations. Copied read-only; the verifier fingerprints these at the start
# of the run and again at the end.
COPY frozen/ /app/frozen/

# The canonical operating point, replicated into this slot's own tree so no
# slot reads a sibling's bytes. The verifier re-reads its OWN copy at grading
# time, so editing this one produces a refused run with reason
# frozen-axis-mutated rather than a different grade.
COPY nanogpt_substrate.json /app/frozen/nanogpt_substrate.json

# The free half: the interface the submission implements and a complete working
# baseline that is valid today.
COPY tokenizer_interface.py /app/tokenizer_interface.py
COPY baseline_tokenizer.py /app/baseline_tokenizer.py
COPY submission_protocol.md /app/submission_protocol.md

# The TRAIN half of FineWeb10B, and only the train half. Build egress is open;
# run-time solver egress is not, so the corpus is staged here rather than left
# for the agent to fetch. The val glob is NOT staged into this image: the graded
# split lives only in the verifier image, which is what makes the graded figure
# something the solver cannot precompute.
RUN pip install --no-cache-dir --break-system-packages huggingface_hub \
 && python3 -c "import urllib.request, pathlib; pathlib.Path('/app/data').mkdir(parents=True, exist_ok=True)" \
 && python3 -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/data/cached_fineweb10B.py', '/app/data/cached_fineweb10B.py')" \
 && cd /app/data && python3 cached_fineweb10B.py 20 \
 && rm -f /app/data/fineweb10B/fineweb_val_*.bin \
 && ls /app/data/fineweb10B/fineweb_train_000001.bin

# The graded deliverable. The agent writes it; nothing here writes it for them.
# /app/submission.py is snapshotted as an artifact by the delivery manifest.

CMD ["/bin/bash"]
