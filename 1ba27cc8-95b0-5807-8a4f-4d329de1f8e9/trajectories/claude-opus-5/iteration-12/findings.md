Submission is written and verified end-to-end.

**What I shipped** (`/workspace/refine.py`): iteration 11's `ortho_momentum` + WSD core, `max_steps` held at the 2000 ceiling, with the one axis that measurably paid last time pushed further — `embed_lr_mult` 16 → 32, `head_lr_mult` 6 → 5.

**Evidence:** two concurrent 250-step dev probes with the schedule made LR-identical to the first 250 steps of the real WSD run (constant LR, matched 40-step warmup). embed 16/head 6 → 5.850; embed 32/head 5 → **5.801** at step 250, and the new setting was ahead at every eval point from step 200 on. That is a ~0.05 nat lead, larger than the 0.024 separation that preceded iteration 11's 60-step crossing gain, so the direction of the previous improvement is not yet exhausted.

**Risk control:** `max_steps` stays at 2000 (the iteration-4 failure mode of shortening the horizon and losing the crossing entirely is avoided), `report.crossing_step` is null so no claimed crossing can diverge from the verifier's, `summary_digest` is echoed verbatim from the handed `state.json`, `inherits_from` is `[]`, and the file is stdlib-only and self-contained.