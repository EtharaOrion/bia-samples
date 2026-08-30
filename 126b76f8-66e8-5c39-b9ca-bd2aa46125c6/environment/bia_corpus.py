"""The deterministic corpus the demonstration operating point trains and grades on.

The pinned image carries no FineWeb byte and the bundle ships none, so
`environment/bia_loader.py` had nothing to read at /data/fineweb/train. This
module supplies a corpus that is deterministic, analytically characterised, and
generated from bound seeds rather than sampled at run time.

WHAT THE LANGUAGE IS

An order-2 Markov chain over a small alphabet. For every one of the
`vocab ** order` contexts the module draws a permutation of `successors`
distinct successor tokens and attaches THE SAME weight profile to them. Because
the profile is shared, the conditional entropy is identical in every context and
the Bayes-optimal per-token cross-entropy of the language is the analytic
entropy of that one profile. It is a computed number, not an estimated one, so
the distance between the bound bar and the floor of the language is exact.

WHAT IS PUBLIC AND WHAT IS NOT

The transition table seed is public: it is on the agent surface in
environment/operating_point.json, because it defines the language and an agent
that cannot see the language it is training on is not solving an optimization
task. The stream seed the VERIFIER evaluates on is not on this surface. It lives
in tests/holdout.json inside the verifier image, so the graded split is drawn
from the same distribution and from tokens the agent container never receives.

NO NETWORK, NO CLOCK, NO AMBIENT RANDOM SOURCE. Every draw comes from a
numpy.random.Generator explicitly seeded from a bound integer.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Dict, Tuple

HEADER_BYTES = 1024
HEADER_COUNT_SLOT = 2


def weight_profile(successors: int, zipf_exponent: float):
    """The one successor weight profile every context shares.

    A Zipf profile over `successors` ranks. The exponent is the single knob that
    sets how far the language's entropy floor sits below log(successors), and it
    is bound in environment/operating_point.json rather than chosen here.
    """
    import numpy

    ranks = numpy.arange(1, int(successors) + 1, dtype=numpy.float64)
    weights = ranks ** (-float(zipf_exponent))
    return weights / weights.sum()


def entropy_floor(successors: int, zipf_exponent: float) -> float:
    """The Bayes-optimal per-token cross-entropy of the language, in nats.

    Analytic. Every context carries the same profile, so the conditional entropy
    is the same everywhere and no averaging over a stationary distribution is
    needed. This is the number the bound bar target_loss is measured against.
    """
    import numpy

    probs = weight_profile(successors, zipf_exponent)
    return float(-(probs * numpy.log(probs)).sum())


def transition_table(vocab: int, order: int, successors: int, zipf_exponent: float, table_seed: int):
    """(successor_ids, cumulative_weights). Deterministic in `table_seed` alone."""
    import numpy

    if int(successors) > int(vocab):
        raise ValueError("successors " + str(successors) + " exceeds vocab " + str(vocab))
    contexts = int(vocab) ** int(order)
    rng = numpy.random.default_rng(int(table_seed))
    ids = numpy.empty((contexts, int(successors)), dtype=numpy.int64)
    for start in range(0, contexts, 4096):
        stop = min(start + 4096, contexts)
        block = rng.random((stop - start, int(vocab))).argsort(axis=1)
        ids[start:stop] = block[:, : int(successors)]
    probs = weight_profile(successors, zipf_exponent)
    return ids, numpy.cumsum(probs)


def stream(spec: Dict, stream_seed: int, n_tokens: int):
    """`n_tokens` tokens of the language, as uint16. Vectorised over lanes.

    Lanes are independent chains of the SAME language, concatenated lane-major.
    That keeps generation a fixed number of vectorised steps instead of a Python
    loop per token, and puts one context discontinuity every `n_tokens / lanes`
    positions, which is 0.05 percent of positions at the bound lane count.
    """
    import numpy

    vocab = int(spec["vocab_size"])
    corpus = spec["corpus"]
    order = int(corpus["order"])
    if order != 2:
        raise ValueError("only order 2 is implemented; operating point asks for " + str(order))
    lanes = int(corpus["lanes"])
    ids, cum = transition_table(
        vocab, order, int(corpus["successors"]), float(corpus["zipf_exponent"]), int(corpus["table_seed"])
    )
    rng = numpy.random.default_rng(int(stream_seed))
    per_lane = int(math.ceil(int(n_tokens) / lanes))
    out = numpy.empty((lanes, per_lane), dtype=numpy.uint16)
    previous = rng.integers(0, vocab, size=lanes, dtype=numpy.int64)
    current = rng.integers(0, vocab, size=lanes, dtype=numpy.int64)
    out[:, 0] = previous
    if per_lane > 1:
        out[:, 1] = current
    for position in range(2, per_lane):
        context = previous * vocab + current
        draw = rng.random(lanes)
        choice = numpy.searchsorted(cum, draw, side="right")
        numpy.clip(choice, 0, ids.shape[1] - 1, out=choice)
        nxt = ids[context, choice]
        out[:, position] = nxt
        previous, current = current, nxt
    return out.reshape(-1)[: int(n_tokens)]


def write_shard(path: pathlib.Path, tokens) -> pathlib.Path:
    """Write one shard in exactly the layout environment/bia_loader._read_shard reads.

    1024 header bytes whose third int32 is the token count, then the tokens as
    little-endian uint16. The reader is delivered, unmodified bytes; this writer
    is written against it rather than the other way round.
    """
    import numpy

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = numpy.zeros(HEADER_BYTES // 4, dtype=numpy.int32)
    header[HEADER_COUNT_SLOT] = int(tokens.size)
    scratch = path.with_suffix(path.suffix + ".partial")
    with scratch.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(numpy.asarray(tokens, dtype=numpy.uint16).tobytes())
    scratch.replace(path)
    return path


def materialize_shard(directory, spec: Dict, stream_seed: int, n_tokens: int, name: str = "shard_000.bin"):
    """Create the shard if it is absent, and never rewrite one that exists.

    Idempotent and deterministic: the same seeds produce the same bytes, so a
    shard the image baked at build time and a shard a bare checkout synthesises
    at first use are the same file.
    """
    directory = pathlib.Path(directory)
    target = directory / name
    if target.is_file():
        return target
    return write_shard(target, stream(spec, stream_seed, n_tokens))


# ---------------------------------------------------------------------------
# Operating point resolution. One lookup, used by the loader, the verifier's
# evaluator and the image builds, so the executed shape has exactly one source.
# ---------------------------------------------------------------------------

def pin_determinism() -> None:
    """Remove kernel nondeterminism from the graded path.

    The graded quantity is a crossing step recomputed from real GPU training, so
    an atomics-ordered backward kernel is a source of run-to-run drift in the
    number the reward is computed from. This pins the deterministic algorithm
    set and the cuBLAS workspace before any CUDA context exists. It is called at
    import of environment/bia_loader.py, which every run reaches before its
    first tensor operation.
    """
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError:
        return
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        # The fused attention backends reduce with atomics and are documented
        # non-deterministic in their backward. The math backend is not, and at
        # the demonstration sequence length it is not the cost driver.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:  # noqa: BLE001  determinism is pinned where it can be, never faked
        pass


POINT_FILENAME = "operating_point.json"
SEARCH_DIRS = ("/app/env", "/verifier")


def _document() -> Dict:
    here = pathlib.Path(__file__).resolve().parent
    for candidate in (here,) + tuple(pathlib.Path(item) for item in SEARCH_DIRS):
        path = candidate / POINT_FILENAME
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(POINT_FILENAME + " is not beside " + str(here) + " or on the bound search path")


def operating_point(name: str = "") -> Dict:
    """The selected point, plus its own name under the key `name`.

    Selection order: the explicit argument, then the BIA_OPERATING_POINT
    environment variable the document itself names, then the document's own
    `selected` field. Nothing else selects a point.
    """
    import os

    document = _document()
    chosen = str(
        name
        or os.environ.get(str(document.get("selection_override_env", "BIA_OPERATING_POINT")), "")
        or document["selected"]
    )
    points = document["points"]
    if chosen not in points:
        raise KeyError("operating point " + repr(chosen) + " is not one of " + ", ".join(sorted(points)))
    resolved = dict(points[chosen])
    resolved["name"] = chosen
    return resolved


def resolved_entropy_floor(point: Dict) -> float:
    corpus = point.get("corpus") or {}
    return entropy_floor(int(corpus["successors"]), float(corpus["zipf_exponent"]))


def ensure_train_shard(point: Dict) -> Tuple[pathlib.Path, bool]:
    """(directory, synthesised). Refuses rather than inventing a bound-point corpus."""
    directory = pathlib.Path(str(point["train_shard_dir"]))
    existing = sorted(directory.glob("*.bin")) if directory.is_dir() else []
    if existing:
        return directory, False
    if not bool(point.get("synthesize_train_shard_if_absent")):
        raise FileNotFoundError(
            "operating point " + repr(point.get("name")) + " binds a corpus at " + str(directory)
            + " that does not exist, and that point does not permit synthesis"
        )
    corpus = point["corpus"]
    materialize_shard(directory, point, int(corpus["train_stream_seed"]), int(corpus["train_tokens"]))
    return directory, True


if __name__ == "__main__":  # image build entry point
    import sys

    selected = operating_point()
    if selected.get("corpus"):
        where, made = ensure_train_shard(selected)
        sys.stdout.write(
            json.dumps(
                {
                    "point": selected["name"],
                    "train_shard_dir": str(where),
                    "synthesised": bool(made),
                    "entropy_floor_nats": round(resolved_entropy_floor(selected), 6),
                }
            )
            + "\n"
        )
