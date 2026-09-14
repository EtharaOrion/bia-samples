# Isolated verifier environment for OER-12. Pinned by digest, never by a floating tag.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# VERIFIER EGRESS IS DENIED. The solving agent's surface has open egress; this one does
# not. The two surfaces differ by design, the asymmetry is bound in task.toml, and the
# divergence from seed/contract.yaml batch.optimization_bindings.solver_egress is
# recorded in solution/grounding.yaml under solver_egress_divergence.
ENV OER_SLOT=OER-12 \
    OER_VERIFIER_EGRESS=denied \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /verifier

COPY . /verifier/tests/

# The checkers import only json, math, hashlib, pathlib, dataclasses and typing, so no
# package is installed here. An installed package would be a surface the AST allowlist
# does not cover.

ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
