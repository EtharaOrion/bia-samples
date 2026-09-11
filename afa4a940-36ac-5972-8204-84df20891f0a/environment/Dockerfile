# Agent environment for slot OER-10.
#
# Pinned by digest, never by a floating tag. A tag re-resolves under the author's feet and
# a run that cannot name the bytes it ran on has measured nothing.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Solver egress is OPEN inside the sandbox: the agent may fetch its own packages, datasets
# and models. The upstream deny-list is carried in task.toml [agent] and is hygiene, never
# containment. A curation recipe travels through mirrors, caches, paraphrased write-ups and
# model weights, so the deterministic behavioural screen in tests/ is the load-bearing
# control and this Dockerfile makes no containment claim.

ENV OER10_SUBMISSION=/workspace/submission/recipe.py \
    OER10_RUN_RECORD=/telemetry/run_record.json \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /workspace

# The frozen substrate. The agent reads all of it and modifies none of it: the token
# budget, the model, the optimizer and the evaluation split are frozen axes.
COPY frozen_train.py /workspace/environment/frozen_train.py
COPY recipe_api.py /workspace/environment/recipe_api.py
COPY probe_pool.json /workspace/environment/probe_pool.json
COPY raw_pool.json /workspace/environment/raw_pool.json
COPY held_out_split.json /workspace/environment/held_out_split.json
COPY exclusion_set.json /workspace/environment/exclusion_set.json
COPY published_mixtures.md /workspace/environment/published_mixtures.md
COPY README.md /workspace/environment/README.md

RUN mkdir -p /workspace/submission /telemetry /logs/verifier

CMD ["/bin/bash"]
