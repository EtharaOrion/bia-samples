# The agent-facing task environment.
#
# Pinned by digest, never by a floating tag. Harbor accepts any string and does not itself
# require a digest, so the digest requirement is this contract's and this file enforces it.
#
# Build context: the BUNDLE ROOT, not environment/. The build needs the environment tree and
# nothing else from the bundle, and copying the whole tree would put the private solution on
# the agent's surface.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The substrate is exact finite arithmetic over a ground set of a few hundred elements. It
# needs no accelerator runtime and no third-party package: everything it uses is in the
# standard library. That is deliberate, because a reproducible grading path cannot rest on a
# wheel resolving the same way twice. Every dependency is therefore baked at build time by
# being already present in the pinned base.
#
# The solving agent has open internet egress inside the sandbox and may install whatever it
# wants for its own search. Nothing it installs reaches the graded path.

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0
ENV OER25_ENV_ROOT=/task/environment
ENV PYTHONPATH=/task/environment

COPY environment/instance.json /task/environment/instance.json
COPY environment/envelope.json /task/environment/envelope.json
COPY environment/establish_state.py /task/environment/establish_state.py
COPY environment/harness.py /task/environment/harness.py
COPY environment/fitness.py /task/environment/fitness.py

# ESTABLISH THE SEAM OFFSET IN BUILT STATE. This is the only place the value comes into
# existence. It is derived by arithmetic over the two frozen inputs above, written to
# /task/environment/state/seam_state.json inside this image, and read back at run time
# through environment/harness.py read_seam_offset. The value is not a literal in any file
# copied above and is not a word of instruction.md.
RUN python3 /task/environment/establish_state.py

WORKDIR /app

# /app is where the agent writes its producer program. /workspace is scratch the agent owns.
RUN mkdir -p /app /workspace/notes /workspace/candidates

CMD ["/bin/bash"]
