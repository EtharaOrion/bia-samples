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

# MEASURED CORRECTION, wave-11. This line was
#   RUN python3 -m pip install --no-cache-dir "pyyaml==6.0.2"
# and it made this image unbuildable, so nothing under tests/ could ever run in the
# environment task.toml pins. Measured against the digest above: python3 is the
# system interpreter, it carries the PEP 668 marker
# /usr/lib/python3.12/EXTERNALLY-MANAGED, and pip refuses with
# externally-managed-environment at exit code 1.
#
# The install was also unnecessary: the pinned image already ships PyYAML 6.0.3, the
# only third-party import in this whole grading tree. Rejected on measured grounds:
# --break-system-packages downgrades the pinned image's own PyYAML and pulls
# unpinned transitive dependencies; python3 -m venv cannot bootstrap because
# ensurepip is absent from this base; another base image moves the pinned @sha256:.
#
# The dependency is proven at build time rather than assumed, so a base image that
# drops PyYAML fails this build loudly instead of failing grade.py at grading time.
RUN python3 -c "import yaml, sys; sys.stdout.write('pyyaml ' + yaml.__version__ + ' at ' + yaml.__file__ + '\n')"

WORKDIR /verifier
COPY . /verifier/tests

# The grading tree is copied in whole and the submission is never copied in beside it.
# tests/runner.py copies the submission alone into a fresh temporary directory at run
# time, so the generation process never shares a working directory with the benchmark.
ENTRYPOINT ["/bin/bash", "/verifier/tests/test.sh"]
