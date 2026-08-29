# Make BIA-GSN-1 faster without changing a single bit of its output

## The objective

`/environment/reference_impl.py` defines an operator called BIA-GSN-1 and one implementation of it. Your job is to write a different implementation that returns exactly the same output and returns it faster on the single H100 in this environment. The output is frozen and the implementation is free.

Exactly the same output means bitwise identical. The grader compares the raw 32 bit patterns of your tensors against the raw 32 bit patterns the frozen reference produced, so a signed zero difference and a one unit in the last place difference both count as a divergence. There is no tolerance and there is no partial credit for being close. An implementation that diverges scores exactly zero no matter how fast it is.

## The submission contract

Write your implementation to `/workspace/submission/impl.py`. It must define:

```python
def forward(x, g, b, chunk_rows):
    ...
    return Y, R
```

`x` is a float32 tensor of shape `(N, C)` where `C` is a power of two, `g` and `b` are float32 tensors of shape `(C,)`, and `chunk_rows` is an integer. `Y` must be a float32 tensor of shape `(N, C)` and `R` must be a float32 tensor of shape `(N,)`. You may add further files beside `impl.py` and import them; the directory holding your submission is on the import path, and so is `/environment`.

## What the operator is

Per row `i`, with the constants `eps = 1.220703125e-04`, `alpha = 0.5`, and `beta = 1.5`, all three exactly representable in binary32:

```
s = tree_sum(x[i] * x[i])
m = s * (1 / C)
d = sqrt(m + eps)
n = x[i] / d
y = n * g
y = y + b
h = max(y, y * alpha)
p = h * h
q = h + p * beta
e = tree_sum(q * q)
r = sqrt(e * (1 / C))
```

`Y[i]` is `q` and `R[i]` is `r`. Every primitive in that list is add, subtract, multiply, divide, square root, or maximum on IEEE-754 binary32, and every one of those is correctly rounded, so the value of each output element is fixed by this sequence alone. It does not depend on the device, the backend, the memory layout, or the chunk size.

`tree_sum` is an adjacent pairwise binary tree over a power of two length: level one adds element `2i` to element `2i+1` for every `i`, and the levels repeat until one element remains. The order is part of the specification. A library reduction chooses its own accumulation order and will not in general reproduce these bits.

## What the reference implementation is, and what it is not

`reference_impl.reference_forward` is the frozen statement of the output. It is not a statement about how the operator has to be computed. Read it and separate the things it does because the operator requires them from the things it does because that is how this particular implementation happens to be written. The first set you must reproduce exactly. The second set is yours to change.

## Red lines

Do not edit `/environment/reference_impl.py` or `/environment/spec.json`. Both are hashed before your submission is loaded and hashed again at grading time, and a change to either scores exactly zero. Making the reference slower is not a way to improve the ratio.

Do not enable a relaxed precision path. TF32 on either backend, a reduced precision reduction, a non highest float32 matmul precision, an open autocast context, or a relaxed precision environment override each score exactly zero. Your process is put into a strict numeric posture before your module is imported, its environment is built from scratch by the grader and read back out of the kernel, and its state is read again after your code has run, so an import side effect is visible. In any case a relaxed path changes the bit patterns, and the bit patterns are compared.

Do not obtain a number by any route other than computing the operator faster. This is not a request; the protocol below is built so that none of the usual routes pays. Caching cannot pay because no two timed calls receive the same input. Deferring work cannot pay because the round barrier charges it. Reporting a number cannot pay because the grader reads no number you produce.

## Where your code runs

Your module is imported into its own process, started by the grader, and the grader speaks to it over a pipe. That process holds no checker, no event log, no reward and no clock that the score reads. Anything you rebind inside it changes only what your own process does with tensors the grader owns, and the grader believes nothing your process says about itself: it times the round trips on its own clock, it chooses the inputs, and it compares your raw output bytes against the reference's.

## How you are scored

The grader runs a correctness gate first and a measurement second. A submission that fails the gate never reaches a timing number.

The measurement interleaves your implementation with the frozen reference, each in its own process on the same device: fifteen rounds, and inside each round eight warmup calls and forty timed calls for each implementation in a fixed order. Three properties of that protocol are worth reading twice, because they decide what is worth optimizing.

No timed call repeats an input. Before every call, your process folds the previous call's output back into the input, at one column per row that the grader chose for that round. Call `t` therefore runs on an input no call has ever seen, and the whole round's output is a chain: a call that is not performed changes every later call's result, and the grader compares the end of round bytes of the two processes.

The round statistic is the total, not the median. A median over the forty trials would discard a round whose work was concentrated into one call. The total cannot be moved around inside the round it measures.

Every round ends with a timed barrier. The grader asks both processes for the exact values at positions it picks, and reads them on the host, so device work that was queued but not waited for is charged inside the interval being timed rather than escaping it. Returning a lazy handle buys nothing.

```
round_total = sum of the round's forty trial durations, plus the round's barrier duration
speedup     = median over rounds of ( reference round_total / submission round_total )
score       = min(max((speedup - 1.0) / (32.543 - 1.0), 0.0), 1.0)
```

The baseline is the frozen reference measured against itself inside the same run, so the score does not depend on a wall clock constant recorded on some other machine. A speedup of one scores zero, the score rises linearly from there, and a speedup of 32.543 or better scores one.

## Your budget

One graded attempt is bounded at 7.2 minutes of single H100 time, which is a six hour session divided across fifty attempts. The measurement phase inside a graded attempt is bounded at 240 seconds by a hard guard: if the measurement cannot finish inside it, the run is aborted and the score is exactly zero with the reason `measurement_budget_exceeded`. An implementation slower than the reference is the usual way to trip that guard.

The session ends after fifty attempts or six hours, whichever comes first, and your best attempt is the one that counts.

## Checking your work before you submit

```
python3 /environment/selfcheck.py /workspace/submission/impl.py
```

That runs the same shape of comparison the grader runs, against the agent visible copy of the operator, and prints an indicative speedup under a short local protocol. It is a convenience and not the grader. The grader uses its own private copy of the operator, its own fixtures, and the full timing protocol above, so a clean self check is an indication and never a score.

The fixture set the gate runs over includes ordinary normal data, dense mantissa data, data scaled far down, data scaled far up, rows that are exactly zero, and rows carrying negative zero. If your implementation agrees on ordinary data and disagrees on one of the others, the disagreement is real and the ordinary case was luck.
