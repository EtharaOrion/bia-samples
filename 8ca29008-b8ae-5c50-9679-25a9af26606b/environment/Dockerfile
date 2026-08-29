# Agent container spec for bia S08.
#
# Harbor reads this as the MAIN container. task.toml declares no [verifier.environment],
# so environment_mode is shared and the verifier runs in this same image. That is
# required rather than incidental: a separate verifier container could not read the
# agent's /telemetry, which tests/grade.py must read.
#
# On a normal run this file is never read, because harbor pulls [environment].docker_image
# by digest. It exists for --force-build. Starting FROM the identical digest is what makes
# both paths converge, since an independent base would let that flag silently swap the
# graded runtime.
#
# The digest below is transcribed from the verified track3 reference bundle's recorded
# base image and has not been re-resolved against a registry from the authoring host.
# That is carried as the named gap image-digest-unreresolved in solution/grounding.yaml.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# Egress is permitted at build time only. task.toml narrows [agent].network_mode to none
# for the scored run, so everything the solver needs is installed here.
RUN pip install --no-cache-dir pytest pyyaml

ENV S08_STATE_DIR=/opt/bia/s08 \
    S08_FIXTURE_DIR=/opt/bia/s08/fixtures \
    S08_TELEMETRY_DIR=/telemetry \
    S08_PROFILE=scaled

# The frozen core, the harness runner, the corpus tool and the frozen checkpoint fixture.
# The fixture is a shipped artifact covered by the bundle content hash and is never
# regenerated inside a graded attempt.
COPY runner/   /opt/bia/s08/runner/
COPY tools/    /opt/bia/s08/tools/
COPY fixtures/ /opt/bia/s08/fixtures/

# Materialize the frozen corpus at build time so no graded attempt pays for it. The
# corpus is produced from a pinned seed and is not downloaded, so this step needs no
# network even though the build has one.
RUN python3 /opt/bia/s08/tools/make_corpus.py --profile scaled

RUN mkdir -p /telemetry /workspace/submission /logs/verifier
WORKDIR /workspace
