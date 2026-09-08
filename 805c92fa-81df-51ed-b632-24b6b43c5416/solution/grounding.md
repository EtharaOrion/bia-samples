# Grounding — where every number in this bundle came from

Nothing in this bundle is a declared anchor. Both endpoints of the reward scale are
measured by the verifier on every grading run, by building and training the two anchor
shapes from scratch alongside the submission's. What is recorded here is the AUTHORING
search that chose the shipped default, the private reference and the oracle.

## The measurement

Every row is a full build-and-train of `environment/harness.py` over the frozen
6,291,456-token budget under the frozen recipe, evaluated on `tests/holdout/val_slice.bin`
— the split the verifier grades against. Losses are validation cross entropy in nats.

| shape | non-embedding params | total params | val loss |
|---|---|---|---|
| `L=6 d=384 head=64 mlp=4` | 10,642,944 | 49,326,720 | 5.97004 |
| `L=9 d=320 head=64 mlp=4` | 11,091,520 | 43,336,384 | 5.97661 |
| `L=16 d=256 head=64 mlp=3` | 10,527,232 | 36,333,184 | 5.99296 |
| `L=14 d=256 head=64 mlp=4` | 11,049,984 | 36,855,936 | 5.99786 |
| `L=7 d=384 head=64 mlp=3` | 10,349,568 | 49,033,344 | 6.01000 |
| `L=4 d=512 head=64 mlp=3` | 10,507,264 | 62,068,864 | 6.02175 |
| `L=5 d=512 head=64 mlp=2` | 10,509,824 | 62,071,424 | 6.02543 |
| `L=10 d=320 head=64 mlp=3` | 10,272,640 | 42,517,504 | 6.03922 |

## The three artifacts

* **`environment/default_shape.json`** — `L5_d512_m2`, measured 6.02543. The LOW anchor;
  the verifier rebuilds and retrains it on every grading run.
* **`tests/private/reference_shape.json`** — `L6_d384_m4`, measured 5.97004. The HIGH
  anchor, staged only into the verifier image.
* **`solution/reference.py`** — `L9_d320_m4`, measured 5.97661. A DIFFERENT point of
  the same search. Against the two anchors it closes
  (6.02543 - 5.97661) / (6.02543 - 5.97004) = **0.881** of the gap.

The span between the anchors is 0.05539 nats, and the two closest distinct shapes in
the table sit 0.00367 nats apart, which bounds the resolution a submission can
expect to be graded at.
