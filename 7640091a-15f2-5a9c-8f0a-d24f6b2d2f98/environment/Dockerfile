# The agent-facing task environment.
#
# Pinned by digest, never by a floating tag. Harbor accepts any string and does not itself
# require a digest, so the digest requirement is this contract's and this file enforces it.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The serving substrate is a deterministic discrete-event simulator over a frozen request
# trace and a frozen hardware envelope. It needs no accelerator runtime and no third-party
# package: everything it uses is in the standard library. That is deliberate, because a
# reproducible grading path cannot rest on a wheel resolving the same way twice.
#
# The solving agent has OPEN internet egress inside the sandbox and may install whatever it
# wants for its own search. Nothing it installs reaches the graded path.

WORKDIR /workspace

COPY serving_sim.py /task/environment/serving_sim.py
COPY refine.py /task/environment/refine.py
COPY trace.json /task/environment/trace.json
COPY envelope.json /task/environment/envelope.json

ENV PYTHONPATH=/task/environment
ENV PYTHONDONTWRITEBYTECODE=1

# notes/ is the durable, agent-writable recovery area. The refinement loop never compacts it,
# never truncates it and never rewrites it. context/ is written by the loop and is lossy.
RUN mkdir -p /workspace/notes /workspace/attempts /workspace/context /workspace/logs

CMD ["/bin/bash"]
