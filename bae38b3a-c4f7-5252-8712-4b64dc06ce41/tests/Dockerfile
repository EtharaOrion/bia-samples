# The isolated verifier environment. Pinned by digest, never by a floating tag: a tag
# is a mutable pointer and a bundle pinned to one is a bundle whose grading environment
# can change without a byte of this tree moving.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The verifier surface keeps the bound network policy: egress denied. task.toml binds
# [verifier] egress = "denied" while [agent] binds solver_egress = "open". The two
# surfaces differ by design and the divergence is recorded in solution/grounding.yaml.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    LC_ALL=C \
    LANG=C

# No package install. The pinned image already carries PyYAML, and tests/grade.py
# guards the import regardless. A bare pip install exits non-zero on this
# externally-managed base under PEP 668, which would stop the verifier building.

WORKDIR /verifier
COPY . /verifier/tests

# The grading tree is copied in whole and the submission is never copied in beside it.
# tests/runner.py copies the submission alone into a fresh temporary directory at run
# time, so the generation process never shares a working directory with the benchmark.
ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
