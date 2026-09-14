# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml negative_submissions.neg-claim-without-measurement
# a claimed figure with no vocabulary behind it, so nothing produced it
import json, sys
from pathlib import Path
def main(argv):
    out = Path(argv[2]); out.mkdir(parents=True, exist_ok=True)
    (out / "claim.json").write_text(
        json.dumps({"schema": "oer16.claim/v1", "claimed_bits_per_byte": 0.9},
                   sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return 0
if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
