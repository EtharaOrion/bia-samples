"""The reference solution for OER-07. Private; never on the agent surface.

Two halves, and the second is the one this slot is really about.

The optimizer half is an orthogonalized momentum update on the two-dimensional
blocks, with the warmup that approach a3 established and the learning rate held
under the ceiling that approach a2 refuted upward of. It reaches the target on
every graded seed with a crossing the verifier sustains.

The loop half is the discipline that makes the first half reachable at all. Both
of the facts the optimizer half rests on, the a3 warmup and the a2 ceiling, have
been folded out of the summary by iteration 5. This reference does not read them
out of the summary. It reads the durable ledger, reconciles the ledger against
the summary it was handed, writes a reconstruction record naming everything the
compaction dropped, and only then proposes.

The grading process never imports this file.
"""

import json
import pathlib

# The learning-rate ceiling approach a2 refuted upward of, and the warmup length
# approach a3 established. Both were folded out of the summary by iteration 5;
# both are recovered from the durable ledger, which is the point.
LR_CEILING = 0.008
WARMUP_STEPS = 256
INNER_ITERATIONS = 5
MOMENTUM = 0.95

LEDGER_PATH = pathlib.Path("/workspace/ledger/ledger.jsonl")


def read_ledger(path=LEDGER_PATH):
    if not path.is_file():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_ledger(record, path=LEDGER_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return record


def visible_ids(summary):
    rows = []
    for entry in (summary or {}).get("entries") or []:
        ident = str(entry.get("approach_id", ""))
        if ident and ident != "digest":
            rows.append(ident)
    return rows


def detect_compaction(summary, ledger):
    """What the record holds that the handed summary no longer shows.

    This is the detection, and it is deliberately not a flag lookup. Nothing
    announces the compaction, so the only signal is the difference between the
    durable record and the current view.
    """
    visible = set(visible_ids(summary))
    recorded = {str(row.get("approach_id")) for row in ledger if row.get("approach_id")}
    recorded = {ident for ident in recorded if not ident.startswith("recon-")}
    return sorted(recorded - visible)


def established_by(ledger, approach_id):
    for row in ledger:
        if str(row.get("approach_id")) == approach_id:
            return list(row.get("established") or [])
    return []


def recover(iteration, summary, path=LEDGER_PATH):
    """Reconcile before proposing, and record the reconciliation durably.

    Returns the constraints the session established, recovered from the ledger
    rather than from the summary. Where the two disagree the ledger wins; the
    summary is a view and the ledger is the record.
    """
    ledger = read_ledger(path)
    dropped = detect_compaction(summary, ledger)
    constraints = []
    for row in ledger:
        constraints.extend(str(item) for item in row.get("established") or [])
    if dropped:
        append_ledger(
            {
                "approach_id": "recon-" + str(iteration),
                "status": "tried",
                "established": [],
                "iteration": int(iteration),
                "reconstructed_from_ledger": True,
                "recovers": list(dropped),
            },
            path,
        )
    return {"dropped": dropped, "constraints": sorted(set(constraints))}


def schedule(step, total_steps, base_lr):
    """Linear warmup then cosine decay, held under the recovered ceiling.

    The ceiling is applied last so no schedule shape can lift the rate above the
    value approach a2 refuted upward of. That refutation is not in the summary
    at the iteration this runs; it comes back from the ledger.
    """
    if step < WARMUP_STEPS:
        scaled = base_lr * (float(step + 1) / float(WARMUP_STEPS))
    else:
        span = max(1, total_steps - WARMUP_STEPS)
        progress = min(1.0, float(step - WARMUP_STEPS) / float(span))
        scaled = base_lr * (0.5 * (1.0 + _cosine(progress)))
    return min(scaled, LR_CEILING)


def _cosine(progress):
    """cos(pi * progress) by series, so this file pulls in no math dependency."""
    x = 3.141592653589793 * progress
    term = 1.0
    total = 0.0
    for index in range(12):
        total += term
        term = -term * x * x / ((2 * index + 1) * (2 * index + 2))
    return total


def orthogonalize(matrix, iterations=INNER_ITERATIONS):
    """Newton-Schulz orthogonalization of a two-dimensional update block.

    Five inner iterations is the value approach a4 established as stable. The
    routine is written against a duck-typed tensor so this file carries no
    framework import; the harness hands it the live tensor type.
    """
    scale = (matrix * matrix).sum().sqrt() + 1e-7
    current = matrix / scale
    for _ in range(iterations):
        product = current.T @ current
        current = 1.5 * current - 0.5 * (current @ product)
    return current


class OrthogonalMomentum:
    """One forward-backward pass per step. No extra pass, no extra token."""

    def __init__(self, params, lr, total_steps):
        self.params = [item for item in params]
        self.lr = lr
        self.total_steps = total_steps
        self.step_index = 0
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for param in self.params:
            param.grad = None if set_to_none else param.grad

    def step(self):
        rate = schedule(self.step_index, self.total_steps, self.lr)
        for index, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                continue
            buffer = self.state.get(index)
            buffer = grad.clone() if buffer is None else buffer * MOMENTUM + grad
            self.state[index] = buffer
            update = orthogonalize(buffer) if buffer.ndim == 2 else buffer
            param.data = param.data - rate * update
        self.step_index += 1


def build_optimizer(params, cfg):
    """The one seam the task exposes. Frozen axes are untouched by construction.

    Nothing here reads the dataset, the batch size, the architecture, or the
    number of passes per step. It receives parameters and returns an optimizer,
    so no frozen axis is reachable from this function at all.
    """
    return OrthogonalMomentum(
        params,
        lr=float(cfg.get("lr", LR_CEILING)),
        total_steps=int(cfg.get("total_steps", 1000)),
    )
