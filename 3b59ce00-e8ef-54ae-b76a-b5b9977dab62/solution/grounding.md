# Grounding — where every number in this bundle came from

Nothing in this bundle is a declared anchor. Both endpoints of the reward scale are
measured by the verifier on every grading run. What is recorded here is the AUTHORING
search that chose the shipped default, the private reference and the oracle.

## The measurement

Every row is a full training run of `environment/harness.py` over the frozen
25,165,824-token budget, evaluated on `tests/holdout/val_slice.bin` — the split the
verifier grades against — as PLAIN cross entropy through that run's own softcap.

| policy | softcap | smoothing | z_loss | decay (embed/hidden/head/scalar) | val loss |
|---|---|---|---|---|---|
| `bare_z1e3` | 15.0 | 0.0 | 0.001 | 0.0/0.0/0.0/0.0 | 5.06324 |
| `hidden_decay` | 15.0 | 0.0 | 0.0 | 0.0/0.05/0.0/0.0 | 5.06698 |
| `bare_cap64` | 64.0 | 0.0 | 0.0 | 0.0/0.0/0.0/0.0 | 5.06824 |
| `smooth002` | 15.0 | 0.02 | 0.0 | 0.0/0.0/0.0/0.0 | 5.06883 |
| `bare_cap30` | 30.0 | 0.0 | 0.0 | 0.0/0.0/0.0/0.0 | 5.06894 |
| `best_guess` | 30.0 | 0.0 | 0.001 | 0.0/0.0/0.0/0.0 | 5.07167 |
| `bare_z1e4` | 15.0 | 0.0 | 0.0001 | 0.0/0.0/0.0/0.0 | 5.07408 |
| `decay_only` | 15.0 | 0.0 | 0.0 | 0.1/0.1/0.1/0.1 | 5.07579 |
| `bare` | 15.0 | 0.0 | 0.0 | 0.0/0.0/0.0/0.0 | 5.07829 |
| `best_guess_c` | 22.0 | 0.0 | 0.0003 | 0.0/0.0/0.0/0.0 | 5.08455 |
| `bare_cap8` | 8.0 | 0.0 | 0.0 | 0.0/0.0/0.0/0.0 | 5.10022 |
| `bare_z5e3` | 15.0 | 0.0 | 0.005 | 0.0/0.0/0.0/0.0 | 5.12548 |
| `shipped_default` | 15.0 | 0.1 | 0.0 | 0.1/0.1/0.1/0.1 | 5.13324 |
| `smoothing_only` | 15.0 | 0.1 | 0.0 | 0.0/0.0/0.0/0.0 | 5.13471 |
| `bare_cap4` | 4.0 | 0.0 | 0.0 | 0.0/0.0/0.0/0.0 | 5.99377 |

## The three artifacts

* **`environment/default_policy.json`** — `shipped_default`, measured 5.13324. Uniform 0.1
  decay, 0.1 smoothing, the upstream softcap. The LOW anchor; the verifier retrains it
  on every grading run.
* **`tests/private/reference_policy.json`** — `bare_z1e3`, measured 5.06324. The HIGH
  anchor, staged only into the verifier image.
* **`solution/reference.py`** — `bare_cap30`, measured 5.06894. A DIFFERENT point of
  the same search. Against the two anchors it closes
  (5.13324 - 5.06894) / (5.13324 - 5.06324) = **0.919** of the gap.

The span between the anchors is 0.07000 nats, and the two closest distinct policies in
the table sit 0.00011 nats apart, which bounds the resolution a submission can
expect to be graded at.
