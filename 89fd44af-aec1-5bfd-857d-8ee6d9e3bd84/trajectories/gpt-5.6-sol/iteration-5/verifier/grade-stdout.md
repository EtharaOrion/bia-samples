{"detail": {"novelty": {"status": "deferred"}, "seeds": 2}, "reason": "graded_step=3225", "score": 0.4583333333333333}

--- full record ---
{
  "composite": 0.458333,
  "detail": {
    "novelty": {
      "status": "deferred"
    },
    "seeds": 2
  },
  "formula": "graded_score * (pytests_passed/executed) * (rubrics_passed/total)",
  "graded_score": 0.4583333333333333,
  "loss": {
    "at_graded_step": 3.27673,
    "per_seed": {
      "seed0": 3.27587,
      "seed1": 3.27759
    },
    "steps": 3225
  },
  "metrics": {
    "n_seeds": 2
  },
  "note": "graded_score is what grade.py computed from the telemetry. score is that value after the rubric veto, which can only lower it to 0.0 and can never raise it; the two differ only when a reviewed rubric failed. rubric_gate is 1 clean, 0 vetoed, -1 no verdict, -2 indeterminate, and any negative value is an unreviewed run rather than a clean one. composite is a review aid bounded by score. score.json carries only numeric keys because harbor parses them all as numbers; this sidecar carries the rest.",
  "pytests": {
    "executed": 0,
    "failed": 0,
    "failed_tests": [],
    "passed": 0,
    "skipped": 0
  },
  "reason": "graded_step=3225",
  "reward": 0.4583333333333333,
  "rubrics": {
    "failed": [],
    "gate": -1,
    "passed": 0,
    "total": 0,
    "veto_reason": null
  },
  "score": 0.4583333333333333
}
