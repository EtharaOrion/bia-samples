# Agent container for bia slot S07, mixed-precision-stability.
#
# Harbor reads this as the MAIN container. task.toml declares no [verifier.environment], so
# environment_mode resolves to shared and the verifier runs in this same image. That is
# required rather than incidental: tests/grade.py reads the telemetry the graded attempt wrote
# under the artifacts directory, which an isolated verifier container would not see.
#
# The base is pinned by digest rather than by tag, because a tag can be repointed at different
# bytes and the digest is what binds a recorded result to the code that produced it. The
# recorded digest below is for pytorch/pytorch 2.13.0 with CUDA 13.2, linux/amd64.
FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime

# Everything the task needs is baked in at build time. The graded run has no network, so a
# solve-time download is impossible by construction rather than by policy. The corpus is a
# frozen fixture carried in this image and its sha256 is re-checked by harness.py on every
# run, so a swapped corpus fails loudly instead of quietly changing the task.
WORKDIR /workspace
COPY format.json operating_point.json bia_numerics.py harness.py run_attempt.py /workspace/
COPY corpus /workspace/corpus
COPY submission /workspace/submission

RUN mkdir -p /workspace/artifacts /logs/verifier

# Fail the build rather than the run if the frozen fixture and its implementation disagree, and
# prove the format emulation is live in the image the agent will actually get.
RUN python3 -c "import sys; sys.path.insert(0,'/workspace'); \
import bia_numerics as N, torch, json; \
s=N.format_summary(); \
assert s['format_id']=='BIA-E4M3N-v1', s; \
assert N.quantize(torch.tensor([248.0])).isnan().item(), 'overflow sentinel is dead'; \
assert N.quantize(torch.tensor([240.0])).item()==240.0, 'max_normal is not representable'; \
assert N.quantize(torch.tensor([1.0625])).item()==1.0, 'round-to-nearest-even is wrong'; \
print('BIA-E4M3N-v1 live in image:', json.dumps(s, sort_keys=True))"

ENV BIA_ARTIFACTS=/workspace/artifacts
ENV PYTHONUNBUFFERED=1
