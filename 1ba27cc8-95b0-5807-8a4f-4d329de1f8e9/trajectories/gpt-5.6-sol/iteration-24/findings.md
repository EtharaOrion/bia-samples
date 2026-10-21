Recovered from the run transcript: this attempt was cut off before it could write a closing account, so what follows is its own reasoning as it went, in order, stopping where the attempt stopped.

1. I’ll first preserve the best known recipe, then inspect tuning options.
2. Now I’ll inspect optimizer choices, frozen shape, and local runner behavior.
3. I’ve confirmed AdamW mechanics; next I’ll inspect initialization and available artifacts.
4. I’ll benchmark the known recipe on a compressed 400-step schedule first.
5. The shell rejected that compound form; I’ll rerun it compactly.
6. Baseline returned safely; I’ll inspect its raw evaluation trajectory now.
7. The baseline improves sharply during decay; I’ll test starting annealing earlier.
8. The earlier-decay probe completed; I’ll compare losses at matched steps.
9. Earlier decay sacrifices too much late quality; I’ll probe a modest peak-rate increase.