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
COPY . /verifier/tests/
RUN python3 -m pip install --no-index pytest pyyaml || true

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
