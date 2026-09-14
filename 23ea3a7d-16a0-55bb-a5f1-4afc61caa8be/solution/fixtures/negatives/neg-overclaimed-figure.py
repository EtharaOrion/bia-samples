# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml negative_submissions.neg-overclaimed-figure
# a valid byte-level vocabulary and a claimed figure far below the measurement
import json, sys
from pathlib import Path
def main(argv):
    out = Path(argv[2]); out.mkdir(parents=True, exist_ok=True)
    tokens = [bytes([i]).hex() for i in range(256)]
    (out / "tokenizer.json").write_text(
        json.dumps({"schema": "oer16.tokenizer/v1", "tokens": tokens},
                   sort_keys=True, separators=(",", ":")), encoding="utf-8")
    (out / "claim.json").write_text(
        json.dumps({"schema": "oer16.claim/v1", "claimed_bits_per_byte": 0.4123},
                   sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return 0
if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
