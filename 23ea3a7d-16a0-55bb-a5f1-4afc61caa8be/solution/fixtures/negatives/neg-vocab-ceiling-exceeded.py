# GENERATED SECTION. DO NOT HAND-EDIT.
# Source: solution/grounding.yaml negative_submissions.neg-vocab-ceiling-exceeded
# a vocabulary above the announced ceiling, so the harness refuses to load it
import json, sys
from pathlib import Path
def main(argv):
    out = Path(argv[2]); out.mkdir(parents=True, exist_ok=True)
    tokens = [bytes([i]).hex() for i in range(256)]
    tokens += [(b"z" * 3 + bytes([i // 256, i % 256])).hex() for i in range(9000)]
    (out / "tokenizer.json").write_text(
        json.dumps({"schema": "oer16.tokenizer/v1", "tokens": tokens},
                   sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return 0
if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
