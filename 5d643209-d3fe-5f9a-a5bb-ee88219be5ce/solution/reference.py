import json
import sys

DELTA_CAP = 0.05
AXIS_ROTATION = ("a", "b", "c")


class Session:
    """A session policy that holds what earlier attempts established.

    The whole decision this task grades is where the remaining attempts go. This
    policy keeps three pieces of state across the entire budget: which direction
    it is currently feeding, the best multi-seed mean any attempt has produced,
    and which directions it has already watched go flat.

    Nothing announces that a direction stopped paying. This policy learns it the
    only way available: it compares the harness-measured mean of its last attempt
    against the best mean it has ever seen, and when an attempt on the current
    direction buys nothing, that direction is retired and every remaining attempt
    goes somewhere else. Exactly one attempt is spent confirming the flat, which
    is the cheapest evidence available, and none is spent after that.
    """

    def __init__(self):
        self.axis_index = 0
        self.best_mean = None
        self.retired = set()

    @property
    def axis(self):
        return AXIS_ROTATION[self.axis_index]

    def retire_current(self):
        self.retired.add(self.axis)
        while self.axis_index < len(AXIS_ROTATION) - 1 and self.axis in self.retired:
            self.axis_index += 1

    def observe(self, previous):
        if not previous:
            return
        mean = previous.get("mean_steps")
        if mean is None:
            return
        if self.best_mean is None:
            self.best_mean = mean
            return
        if mean >= self.best_mean and previous.get("axis") == self.axis:
            self.retire_current()
        else:
            self.best_mean = min(self.best_mean, mean)

    def propose(self):
        return {"axis": self.axis, "delta": DELTA_CAP}


def main():
    session = Session()
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            observation = json.loads(text)
        except ValueError:
            break
        session.observe(observation.get("previous"))
        sys.stdout.write(json.dumps(session.propose()) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
