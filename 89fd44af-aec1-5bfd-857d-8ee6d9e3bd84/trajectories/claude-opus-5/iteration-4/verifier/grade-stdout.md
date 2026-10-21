{"detail": {"novelty": {"status": "deferred"}, "seeds": 2}, "reason": "graded_step=3200", "score": 0.5}

--- full record ---
{
  "composite": 0.470588,
  "detail": {
    "novelty": {
      "status": "deferred"
    },
    "seeds": 2
  },
  "formula": "graded_score * (pytests_passed/executed) * (rubrics_passed/total)",
  "loss": {
    "at_graded_step": 3.276675,
    "per_seed": {
      "seed0": 3.27626,
      "seed1": 3.27709
    },
    "steps": 3200
  },
  "note": "score is the graded value the verifier computed and is authoritative. composite is a review aid bounded by it and can never exceed it. score.json carries only numeric keys because harbor parses them all as numbers; this sidecar carries the rest.",
  "pytests": {
    "executed": 17,
    "failed": 1,
    "failed_tests": [
      "test_full_score_target_reached"
    ],
    "passed": 16,
    "skipped": 0
  },
  "reason": "graded_step=3200",
  "rubrics": {
    "failed": [],
    "passed": 9,
    "total": 9
  },
  "score": 0.5
}
