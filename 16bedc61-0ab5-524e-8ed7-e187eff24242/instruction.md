# Reach the target validation loss on fewer training tokens

You are given a working training script for a small GPT and a data loader. Your job is to reach a target validation loss while consuming **fewer training tokens** than the shipped script does.

## What is measured

**Training tokens consumed to first reach the target validation loss.** Not optimizer steps, and not wallclock time.

Training tokens are the tokens the provided loader serves to your script. This is a different quantity from the LLM tokens you spend while working on the problem, and only training tokens are graded.

Because training tokens equal optimizer steps multiplied by batch size, and because batch size is yours to choose, reducing the step count is not by itself progress. A run that halves the steps while doubling the batch consumes exactly the same training tokens and scores exactly the same. The trade between batch size and step count is the problem.

## What you may change

The optimizer, its schedule, its hyperparameters, the model initialization, and the batch size, including scheduling the batch size over the course of training.

## What is frozen

The model architecture, the dataset and its ordering, the validation split, the evaluation cadence, and the seed set. These are enforced rather than requested. Your checkpoints are loaded into the verifier's own model definition with strict shape checking, so a submission that altered the architecture fails to load rather than being trusted to have left it alone.

## How grading works

You submit a training script. The verifier runs it in a clean container on seeds it draws after your submission, so the seed set cannot be targeted in advance and running many seeds to submit the luckiest gains nothing.

The verifier counts training tokens itself, at the loader. It evaluates loss itself, on a validation split that never enters your environment. No number you write to any file is read as a graded quantity, so there is nothing to be gained by what your script prints or logs.

Evaluation happens at a fixed grid of training-token counts that the verifier owns. You cannot change how often evaluation happens, and evaluating more often on your own gains nothing, because only points measured on every seed enter the comparison.

A crossing is declared at the earliest token count on that grid where the mean loss across seeds clears the target by a significance margin. A single lucky seed does not clear it.

## Scoring

Reward is a continuous float from 0 to 1, higher is better. It reports how much of the distance between the shipped baseline and the reference solution your submission closed, measured in training tokens. A separate binary pass flag reports whether you reached the bar the task set. A submission can be valid, complete and still not pass, because reaching the target on no fewer tokens than the shipped script satisfies nothing.

Every zero score carries a machine-readable reason.

## Rules

Draw every training token through the provided loader. It is what counts them, and it is what writes your checkpoints at the verifier's token milestones. A run that bypasses it produces no evaluable checkpoints and scores zero.

Do not choose where to stop by consulting a validation loss you computed during the run. The stopping point must be fixed in advance and identical across every seed. Printing a validation loss is not forbidden; deciding with it is.

You have full internet access. Use it.
