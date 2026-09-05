# The verifier environment. Isolated from the agent surface and pinned by the same
# digest, so the grading run and the solving run cannot diverge on the interpreter.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Verifier egress is DENIED at run time. The solving agent's surface is open; this
# one is not. The asymmetry is bound in task.toml and recorded in
# solution/grounding.yaml. Everything this image needs is baked in below.
ENV OER15_VERIFIER_EGRESS=denied
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /verifier

RUN pip install --no-cache-dir --break-system-packages tiktoken==0.7.0 huggingface_hub==0.24.6

COPY test.sh /verifier/tests/test.sh
COPY runner.py /verifier/tests/runner.py
COPY grade.py /verifier/tests/grade.py
COPY checkers.py /verifier/tests/checkers.py
COPY checkers.yaml /verifier/tests/checkers.yaml
COPY anchors.json /verifier/tests/anchors.json
COPY test_output.py /verifier/tests/test_output.py
COPY rubrics.jsonl /verifier/tests/rubrics.jsonl

# THE HELD-OUT SLICE. The FineWeb10B VALIDATION shard is staged here and only
# here. environment/Dockerfile stages the training shards and never this one, so
# the split that produces the grade exists in the verifier image and in no
# agent-visible container. tests/runner.py resolves it from tests/anchors.json.
RUN python3 - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="kjj0/fineweb10B-gpt2",
    filename="fineweb_val_000000.bin",
    repo_type="dataset",
    local_dir="/verifier/data/fineweb10B",
)
PY
ENV OER15_HELDOUT_ROOT=/verifier/data/fineweb10B

# The reward carrier the runtime mounts. tests/test.sh terminates by writing it.
VOLUME ["/logs/verifier"]

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
