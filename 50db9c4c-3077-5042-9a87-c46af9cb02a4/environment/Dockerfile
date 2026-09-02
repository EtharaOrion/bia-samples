# OER-13 agent environment. The surface the solving agent works on.
#
# The image is pinned by digest, never by a floating tag. A tag is a moving
# pointer, so a bundle pinned to one is not the same bundle tomorrow, and the
# frozen axes this task grades would silently move underneath the grader.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Solver egress is OPEN inside the sandbox by design, so the agent can fetch its
# own packages, datasets and models. The deny-list bound in task.toml under
# [agent].solver_egress_deny is a friction, not a containment proof: mirrors,
# caches, paraphrased write-ups and model weights all carry a recipe past any
# deny-list. The load-bearing control is the deterministic checker in tests/,
# which recomputes the graded quantity from bytes the verifier owns.
ENV OER13_SOLVER_EGRESS=open
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=0
ENV LC_ALL=C
ENV LANG=C

WORKDIR /workspace

# The whole agent-visible substrate. task.toml and instruction.md are mounted by
# Harbor alongside it; tests/ and solution/ are absent from this surface by
# construction and no COPY here can reach them.
COPY . /workspace/environment

# The three files the agent is licensed to rewrite. Everything else under
# environment/ is frozen and digested against the delivered bytes by the
# verifier: pipeline/train.py, pipeline/protocol.json, eval/* and corpus/*.
LABEL oer13.writable="environment/pipeline/parse.py,environment/pipeline/tokenize.py,environment/pipeline/compose.json"
LABEL oer13.frozen="environment/pipeline/train.py,environment/pipeline/protocol.json,environment/eval/*,environment/corpus/*"
LABEL oer13.accelerators="1"
LABEL oer13.compute_envelope="one H100, single accelerator, no multi-GPU scaling"

CMD ["/bin/bash"]
