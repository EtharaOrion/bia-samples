# The isolated verifier environment.
#
# Same digest pin as the task environment, and pinned by digest for the same reason: a
# floating tag would let the grading interpreter move underneath a frozen bundle.
#
# Build context: the BUNDLE ROOT. The verifier needs the frozen environment inputs so it can
# establish its OWN built state and read the seam offset back for itself, rather than being
# handed an offset by the surface it is grading. solution/ is deliberately not copied.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# EGRESS IS DENIED on this surface. The solving agent's surface has open egress; these two
# surfaces differ by design and the asymmetry is bound in task.toml. A verifier that can
# reach the network is a verifier whose verdict can be fetched.
ENV OER25_VERIFIER_EGRESS=denied
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0

WORKDIR /task

COPY environment/instance.json /task/environment/instance.json
COPY environment/envelope.json /task/environment/envelope.json
COPY environment/establish_state.py /task/environment/establish_state.py
COPY environment/harness.py /task/environment/harness.py
COPY environment/fitness.py /task/environment/fitness.py
COPY tests/ /task/tests/

# The verifier establishes the seam offset in its own built state, independently of the
# agent's image, by running the same generator over the same frozen bytes. Neither image
# takes the value from the other, and tests/checkers.py holds the two derivations against
# tests/bound.json, which solution/recompute.py derived from solution/grounding.yaml.
RUN python3 /task/environment/establish_state.py

ENTRYPOINT ["/bin/bash", "/task/tests/test.sh"]
