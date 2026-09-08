import json
import sys

DELTA_CAP = 0.05


def decide(observation, memory):
    """Choose the next attempt.

    This starting point is deliberately naive: it feeds one direction for the
    whole session. It answers every turn, so it holds the attempt budget, and it
    reaches whatever that one direction is worth and then nothing further.

    `memory` persists across turns. It is the only place a session can hold what
    earlier attempts established.
    """
    memory.setdefault("history", []).append(observation.get("previous"))
    return {"axis": "a", "delta": DELTA_CAP}


def main():
    memory = {}
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            observation = json.loads(text)
        except ValueError:
            break
        sys.stdout.write(json.dumps(decide(observation, memory)) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
