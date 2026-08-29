# Agent container spec for the BIA-GSN-1 kernel throughput task.
#
# Harbor reads this as the MAIN container. task.toml declares no [verifier.environment],
# so environment_mode is shared and the verifier runs in this same image. That matters
# here because the verifier must reach /workspace/artifacts, which the agent's run
# produced, and a separate verifier container could not.
#
# On a normal run this file is never read, because Harbor pulls [environment].docker_image
# by digest. It exists for a forced build, and it starts FROM the identical digest so
# both paths converge. Nothing is installed on top, for the same reason: an extra layer
# would exist only on the build path and be absent on the pull path, which would make
# the graded runtime depend on how the image was obtained.
#
# The base carries torch with CUDA. This task needs nothing else: the operator uses only
# add, subtract, multiply, divide, square root, and maximum on float32, and the harness
# uses only the standard library and numpy, which the base already provides.
FROM pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# The frozen operator and the agent visible tooling. The verifier keeps its own private
# copy of the operator under /tests, which is the copy grading reads.
COPY reference_impl.py /environment/reference_impl.py
COPY spec.json /environment/spec.json
COPY submission.py /environment/submission.py
COPY selfcheck.py /environment/selfcheck.py

WORKDIR /workspace
