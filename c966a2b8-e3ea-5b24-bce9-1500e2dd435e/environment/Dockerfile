# Agent container spec. Harbor reads this as the MAIN container. This task
# declares no [verifier.environment], so environment_mode is "shared" and the
# verifier runs in this same image, which is required because grade.py must
# reach the same /opt/bia/substrate tree and the same corpus the agent trained
# against.
#
# On a normal run this file is never read: Harbor pulls [environment].docker_image
# by digest. It exists for --force-build, and starting FROM the identical digest
# is what makes both paths converge. An independent base would let that flag
# silently swap the graded runtime.
#
# Build context is this environment/ directory.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# Setup-only egress. Everything the graded run needs is fetched here, at build
# time, so the scored work runs with no network at all.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl unzip \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "pyyaml==6.0.3" "pytest==9.1.1"

# The corpus is real text: enwik8, the first 100 MB of a 2006 English Wikipedia
# XML dump, CC-BY-SA 3.0. Both the archive digest and the extracted digest are
# verified, so a substituted corpus fails the build rather than the grade.
RUN mkdir -p /opt/bia/data \
 && curl -fsSL -o /tmp/enwik8.zip http://mattmahoney.net/dc/enwik8.zip \
 && echo "547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc  /tmp/enwik8.zip" | sha256sum -c - \
 && unzip -p /tmp/enwik8.zip enwik8 > /opt/bia/data/enwik8 \
 && echo "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8  /opt/bia/data/enwik8" | sha256sum -c - \
 && rm -f /tmp/enwik8.zip

# The frozen substrate. Read only to the agent. Its bytes are hash pinned in
# SUBSTRATE_MANIFEST.json and the verifier rehashes them at run start and at run
# end, so an edit here is caught by the INVARIANT checker rather than tolerated.
COPY substrate /opt/bia/substrate
COPY fixtures /opt/bia/fixtures
COPY starter /opt/bia/starter
RUN chmod -R a-w /opt/bia/substrate /opt/bia/fixtures

# The submission path the verifier imports, the artifact path arms write to, and
# the anchor cache a session reuses across its fifty attempts.
RUN mkdir -p /workspace/submission /workspace/artifacts /logs/verifier /anchors \
 && cp /opt/bia/starter/update_rule.py /workspace/submission/update_rule.py

ENV BIA_SUBSTRATE_DIR=/opt/bia/substrate \
    BIA_SUBMISSION_DIR=/workspace/submission \
    BIA_WORKDIR=/workspace/artifacts \
    BIA_LOGDIR=/logs/verifier \
    BIA_ANCHOR_DIR=/anchors \
    BIA_CORPUS=/opt/bia/data/enwik8 \
    PYTHONPATH=/opt/bia

WORKDIR /workspace
