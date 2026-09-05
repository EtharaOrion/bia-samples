# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml negative_submissions.neg-claim-nothing
# Claims nothing. This is NOT a way out: every checker passes, the absence is recorded in the score document, and the run is graded on the verifier's measurement exactly as a truthful claimant is. Its gate reward is asserted equal to the truthful reference's, so silence buys nothing that truth does not already buy.
import json, sys
from pathlib import Path
def main(argv):
    out = Path(argv[2]); out.mkdir(parents=True, exist_ok=True)
    tokens = [bytes([i]).hex() for i in range(256)]
    (out / "tokenizer.json").write_text(
        json.dumps({"schema": "oer16.tokenizer/v1", "tokens": tokens},
                   sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return 0
if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
