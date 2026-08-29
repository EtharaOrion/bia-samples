# Reach the target validation loss in fewer optimizer steps

You are given a training script at `train_baseline.py` that trains a GPT-2 class language model. It trains with stochastic gradient descent and momentum at a constant learning rate, and it reaches the target validation loss eventually. Your objective is to reach the same target in as few optimizer steps as possible, and to do it by improving the optimization algorithm rather than by any other route.

## What is frozen

The dataset and its order, the batch size, the model architecture, and the rule of exactly one forward-backward pass per optimizer step. You must not change any of them. All four are bound in `shape.json` and none of them is a parameter you can pass.

The batch size is served by the provided loader and is not an argument to it. The architecture is enforced by loading: the verifier loads every checkpoint you produce into its own copy of the frozen model definition, so a changed depth, width, head dimension or vocabulary fails to load rather than being graded. The forward-backward rule is counted by the frozen model and recorded at every milestone.

## What you may change

The optimization algorithm, the hyperparameters and their schedules, and the model initialization. A method that is slow in wall-clock terms is acceptable, because the metric is optimizer steps and not time.

## What you submit

`recipe.py`, a single self-contained training script, and `claim.json` carrying one field `steps`.

The script must contain every line of code needed to reproduce your run. Do not import a third-party optimizer library; copy the code you need into the script, however long that makes it. Keep the invocation contract of the shipped baseline: read your seed from `BIA_SEED`, build the frozen model, draw every optimizer step from the provided loader, and call `loader.checkpoint(model)` once per step. Read your step horizon from the loader rather than writing it down, because a script that names a graded constant is rejected.

`claim.json` declares the step at which you assert the run reaches the target. Choose it once and apply it identically to every run. Selecting a stopping point per run against that run's own validation loss is not permitted, and a run whose milestones differ between seeds is rejected for exactly that reason.

## How you are graded

By measurement, not by report. The grader draws a set of random seeds after you submit, runs your script itself once per seed in an isolated process, and then loads the checkpoints the provided loader wrote and computes their validation loss itself, on a split your environment never sees. Nothing your script prints is read by anything. Your claim is compared against what the grader measured; it can only lower your outcome and it never enters your score.

Your score is the fraction of the distance from the shipped baseline's crossing step to the reference solution's crossing step that your own measured crossing step covers, clipped to the closed interval zero to one. Reaching the reference or better scores one. Reaching the target no sooner than the shipped baseline scores zero. Reaching the target must hold with statistical significance across the seeds the grader ran, under the rule `(target_loss - mean_loss) * sqrt(n) >= margin`.

Grading is a chain. The submission must parse, it must not replay a published record, it must not name a graded constant, every run must complete, checkpoints must appear on the grader's own milestones, the frozen axes must hold, the stopping point must be uniform across seeds, two seeds must move the measured loss, the target must be reached with significance, and the claim must reconcile. A break at any link ends the chain, so a near miss earns nothing.

## Boundaries

The grading tree and the validation split are part of the environment and are not yours to modify. Both are compared before and after your run and either one changing ends the grading with a zero. The validation split is not present on the filesystem while your process is alive, and no path to it is in your environment, because your score is computed from state the grader observes and fitting the thing you are measured on is not a route to a better score.
