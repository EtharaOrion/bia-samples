# Isolated verifier environment.
#
# Isolated because grading re-executes the WHOLE refinement session: the verifier
# hands each iteration its own summary, launches the submitted planner as a
# separate process group, trains the frozen substrate with the returned recipe,
# and evaluates the weights its own loop held on a split that is mounted HERE and
# never into the agent environment.
#
# Pinned by @sha256:, never a floating tag, and the same digest the agent
# environment pins.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    BIA_ROLE=verifier \
    BIA_SPLIT=val

# Egress is DENIED for the verifier. The solving agent's sandbox has open egress;
# these two surfaces differ by design and the divergence is bound in task.toml
# and recorded in solution/grounding.yaml.
ENV http_proxy=127.0.0.1:9 \
    https_proxy=127.0.0.1:9 \
    no_proxy=""

WORKDIR /verifier

# The build context is the BUNDLE ROOT rather than `tests/`, because two files
# outside `tests/` have to be inside this image and neither can be reached from a
# tests-only context.
#
# `task.toml` is here because it is the CANONICAL declaration of the optimization
# bindings and `tests/session_driver.py` now reads them off it instead of
# restating them. A restated binding is what let this bundle carry 50 attempts at
# 0.12 h in the driver against 20 at 0.3 h in the manifest, and a reader that
# cannot reach the declaration is a reader that has to restate it.
#
# `solution/` is deliberately NOT copied. The verifier grades a submission and
# must not carry the reference it is grading against.
COPY tests/ /verifier/tests/
COPY task.toml /verifier/task.toml

# NO PACKAGE INSTALL, and that is the repair rather than an omission. The former
# line was `pip install --no-index pytest pyyaml || true`, which cannot succeed:
# the image ships no pytest, `--no-index` gives pip nothing to resolve it from,
# the proxy above is a closed port, and `|| true` turned that guaranteed failure
# into a silent one. The entry point then invoked `python3 -m pytest` and exited 1
# on `No module named pytest` on every run, so the verifier reported a failure its
# own assertions never made. `pyyaml` is already present in this image and needed
# no install. The assertions are now run by `tests/run_assertions.py`, which is
# standard library only, so the runner the entry point invokes is carried by this
# surface rather than assumed of the base image.

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
