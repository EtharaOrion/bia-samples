# Attempt 1 — no previous attempts

This is the first attempt at this task, so there is no prior account to carry.

CURRENT budget is 0.0833333 hours of wall clock.

## Your wall clock, in seconds

This attempt gets 300 seconds in total. That covers reading, planning,
experiments and writing the submission. It is not a per-command allowance, and
nothing warns you as it runs down.

Do not give any single command a timeout above 75 seconds. A command
budgeted for longer than the time you have left cannot return before the phase
ends, so it yields no result at all: the attempt simply stops where it stands and
everything you had not yet written is lost. Previous attempts were lost exactly
this way, running one long sweep instead of several short probes.

Training costs about 0.166 s per optimizer step here, so 75 seconds buys
roughly 451 steps. Size probes against that number. Two short probes that both
return beat one long one that does not.

Write `/workspace/refine.py` before you experiment. It is the only artifact that is
graded, and it survives the end of the phase. An unfinished experiment does not.
