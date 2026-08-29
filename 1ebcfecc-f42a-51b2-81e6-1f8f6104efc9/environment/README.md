# Environment

`bia_loader.py` is the provided data loader. Draw every training token through it: it counts the tokens that are graded, and it writes your checkpoints at the verifier's token milestones. A run that bypasses it produces no evaluable checkpoints and scores zero.

The frozen dataset shards and the concrete model shape are mounted into this image at build time. They are not committed here, because the scaled operating point that fixes their dimensions has not been measured yet.
