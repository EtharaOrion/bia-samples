# Agent container spec. Harbor reads this as the MAIN container, not the verifier's; a
# verifier gets its own only when task.toml declares [verifier.environment], which this task
# omits, so environment_mode is "shared" and the verifier runs in this same image. A separate
# verifier container could not reach the agent's /telemetry, which grade.py must read.
#
# On a normal run this file is never read: harbor pulls [environment].docker_image by digest.
# It exists for --force-build, where harbor builds this instead. Starting FROM the identical
# digest is what makes both paths converge; an independent base would let that flag silently
# swap the graded runtime. Nothing is installed on top for the same reason: an extra layer
# would exist only under --force-build and be absent on the pull path.
FROM 426628337772.dkr.ecr.ap-south-1.amazonaws.com/bia@sha256:e2bfe5d9bb5f3a770346f7b63872ae3de620ccc793cfe58dbe6b6aa16bee494d
