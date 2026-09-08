Recovered from the run transcript: this attempt was cut off before it could write a closing account, so what follows is its own reasoning as it went, in order, stopping where the attempt stopped.

1. I’ll inspect the schema and optimizer internals, then immediately save a baseline.
2. I found the likely lever: start WSD decay earlier while preserving 2000 steps.
3. Now I’ll verify evaluation timing, initialization behavior, and available GPU capacity.
4. I’ll run a compressed baseline probe to establish schedule sensitivity safely.
5. The shell rejected the compound syntax; I’ll rerun it as one chain.
6. Baseline reached 5.840; I’ll test the earlier-decay candidate identically.
7. Earlier decay hurt final loss; I’ll compare aligned curves before choosing another refinement.
8. The decay shift is unpromising; I’ll probe a modest 10% learning-rate increase.