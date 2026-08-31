# Re-graded 2026-08-22

The original graded run never executed. `test.sh` launched it while an orphaned agent probe
still held torch's rendezvous port in the shared container, so `init_process_group` failed with
`EADDRINUSE` on port 29677 and training exited rc 1 before step 1. The verifier recorded
`reward 0.0 / reason graded_training_failed_rc_1 / n_seeds 0`, which measured the harness fault,
not the submission.

That original verdict is preserved verbatim under `stale/verifier.port-collision.orig/`.

## What was re-run

The same `optimizer.py` from `artifacts/workspace/submission/` (2,819 bytes, unmodified), 2 seeds
x 3500 steps, `--mode full`, signed with the campaign chain key. Both seeds completed the full
horizon: seed 0 final val_loss 3.26263, seed 1 final val_loss 3.26531. The supervisor-signed
chain (`run_record.jsonl`, 47,681 bytes) verified with no chain break.

    reason      graded_step=3250
    score       0.4166666666666667
    seeds       2

## Limitation of this directory

`reward.json`, `reward_full.json`, `score.json`, `grade-stdout.md` and `train-status.json`
carry the re-graded values and agree with each other and with the campaign ledger.

`test-stdout.txt` and `train-stdout.md` are the ORIGINAL artifacts of the failed run and are
kept deliberately: `train-stdout.md` holds the `EADDRINUSE` traceback that is the evidence for
everything above. They were not regenerated because the re-grade scratch was deleted before
this directory was written, so the raw stdout of the successful run no longer exists.
