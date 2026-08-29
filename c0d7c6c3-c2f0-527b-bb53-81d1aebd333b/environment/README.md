# Environment

`train_baseline.py` is the shipped baseline recipe and the invocation contract the verifier re-executes. Start from it.

`bia_loader.py` is the provided step loader. Draw every optimizer step through it. It serves the frozen batch size, counts the steps that are graded, writes your checkpoints at the verifier's step milestones, and records the run ledger the verifier reconciles against. A run that bypasses it produces no evaluable checkpoints and scores zero.

`frozen_gpt.py` is the frozen architecture. The verifier loads every graded checkpoint into its own copy of this class with `strict=True`, so a submission that changed depth, width, head dimension or vocabulary fails to load rather than being trusted to have left them alone.

`shape.json` carries the bound operating point. `target_loss` is null until it is measured at that point; the verifier refuses to score while it is.

The frozen dataset shards are mounted into this image at build time and are not committed here. The withheld validation split is never mounted into this image at all: it exists only in the verifier environment, and the verifier computes every graded loss itself from the checkpoints you wrote.
