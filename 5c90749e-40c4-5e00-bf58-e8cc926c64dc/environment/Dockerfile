# Agent container spec for BIA S01.
#
# Harbor reads this as the MAIN container. This task declares no
# [verifier.environment], so environment_mode is shared and the verifier runs in
# this same image, which is required because tests/grade.py must read the
# telemetry directory the agent's runner wrote.
#
# On a normal run this file is never read, because Harbor pulls
# [environment].docker_image by digest. It exists for --force-build, where Harbor
# builds this instead. Starting FROM the identical digest is what makes both
# paths converge: an independent base would let that flag silently swap the
# graded runtime. Nothing is installed on top, for the same reason. An extra
# layer would exist only under --force-build and be absent on the pull path.
#
# The digest below is the linux/amd64 pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime
# manifest digest recorded by the verified track3 reference bundle. This lane had
# no registry egress, so the digest is inherited rather than re-resolved here, and
# that is carried as the named coverage gap image-digest-inherited in
# solution/grounding.yaml.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
