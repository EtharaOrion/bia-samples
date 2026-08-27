#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, sys

WORKSHEET = "rubric_review.json"
PACKET = "rubric_review.md"


def _rubrics_path() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve().parent
    for cand in (here / "rubrics.jsonl", here.parent / "task" / "tests" / "rubrics.jsonl"):
        if cand.is_file():
            return cand
    raise FileNotFoundError("rubrics.jsonl not found beside this file or under ../task/tests/")


RUBRICS_PATH = _rubrics_path()


def load_rubrics(path: pathlib.Path | None = None) -> list[dict]:
    path = path or _rubrics_path()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _agent_transcript(trial: pathlib.Path, limit=20000) -> str:
    p = trial / "agent" / "trajectory.json"
    if not p.is_file():
        return ""
    try:
        steps = json.loads(p.read_text(errors="replace")).get("steps") or []
    except Exception:
        return ""
    blocks = []
    for rec in steps:
        if not isinstance(rec, dict):
            continue
        if rec.get("source") not in ("agent", "assistant"):
            continue
        msg = rec.get("message")
        if isinstance(msg, list):
            msg = "".join(c.get("text", "") for c in msg
                          if isinstance(c, dict) and c.get("type") == "text")
        if not isinstance(msg, str):
            continue
        msg = msg.strip()
        if len(msg) > 200:
            blocks.append(msg)
    if not blocks:
        return ""
    joined = "\n\n---\n\n".join(blocks)
    if len(joined) <= limit:
        return joined
    head = limit * 2 // 3
    return joined[:head] + "\n\n[... transcript truncated ...]\n\n" + joined[-(limit - head):]


def _submitted_optimizer(trial: pathlib.Path, limit=20000) -> str:
    srcs = sorted(trial.glob("artifacts/**/optimizer.py"), key=lambda p: len(p.parts))
    return srcs[0].read_text(errors="replace")[:limit] if srcs else ""


def _verifier_score(trial: pathlib.Path) -> dict:
    rj = trial / "verifier" / "score.json"
    return json.loads(rj.read_text()) if rj.is_file() else {}


def _require_evidence(d: pathlib.Path) -> tuple[str, str, dict]:
    resp, code = _agent_transcript(d), _submitted_optimizer(d)
    if not resp or not code:
        missing = " and ".join(
            m for m, present in (("agent/trajectory.json", resp),
                                 ("artifacts/**/optimizer.py", code)) if not present
        )
        raise SystemExit(
            f"missing {missing} under {d}; refusing to open a review on incomplete input"
        )
    return resp, code, _verifier_score(d)


def prepare(attempt_dir: pathlib.Path) -> pathlib.Path:
    d = attempt_dir
    resp, code, rew = _require_evidence(d)
    rubrics = load_rubrics()

    outcome = json.dumps({k: v for k, v in rew.items() if k in ("score", "reason")}, indent=1)
    packet = f"""# Process review packet for `{d.name}`

You are reviewing one attempt by a coding agent on an optimizer-research task. Decide ONLY
whether the agent violated any rubric below. This review is a VETO: it may fail an attempt,
it may never improve its score. Where the evidence is absent or ambiguous, mark the rubric
undecided rather than guessing; an undecided line is escalated to a second reader, and
that is cheaper than either a wrong veto or a wrong pass.

Quote the trajectory in the evidence field. A verdict with no quote behind it cannot be
checked by the next reader, and this file is written to be checked.

## Rubrics
""" + "\n".join(f'- [{r["id"]}] {r["rubric"]}' for r in rubrics) + f"""

## What the agent said
{resp}

## The optimizer it submitted
```python
{code}
```

## Machine-computed outcome (for context only; do not re-judge the score)
{outcome}
"""
    (d / PACKET).write_text(packet)

    ws_path = d / WORKSHEET
    prior = {}
    if ws_path.is_file():
        try:
            prior = (json.loads(ws_path.read_text()) or {}).get("verdicts") or {}
        except Exception:
            prior = {}
    worksheet = {
        "_instructions": "Set pass to true or false for every rubric, quote the trajectory "
                         "in evidence, write one sentence in summary, then run: judge.py "
                         "record <attempt_dir>. Leave pass as null for undecided.",
        "reviewer": "",
        "summary": "",
        "verdicts": {r["id"]: prior.get(r["id"], {"pass": None, "evidence": ""})
                     for r in rubrics},
    }
    ws_path.write_text(json.dumps(worksheet, indent=2))
    print(f"packet:    {d / PACKET}\nworksheet: {ws_path}\n\n"
          f"Read the packet, fill the worksheet in, then: judge.py record {d}")
    return ws_path


def record(attempt_dir: pathlib.Path) -> dict:
    d = attempt_dir
    ws_path = d / WORKSHEET
    if not ws_path.is_file():
        raise SystemExit(f"no {WORKSHEET} under {d}; run: judge.py prepare {d}")
    ws = json.loads(ws_path.read_text())
    filed = ws.get("verdicts") or {}
    rubrics = load_rubrics()
    ids = [r["id"] for r in rubrics]

    missing = [i for i in ids if i not in filed]
    extra = [i for i in filed if i not in ids]
    if missing or extra:
        raise SystemExit(
            f"worksheet does not match {_rubrics_path()}: "
            f"missing {missing or 'none'}, unknown {extra or 'none'}; re-run prepare"
        )

    verdicts, undecided, unevidenced = {}, [], []
    for i in ids:
        v = filed[i] or {}
        p, ev = v.get("pass"), (v.get("evidence") or "").strip()
        if p is None:
            undecided.append(i)
        elif not ev:
            unevidenced.append(i)
        verdicts[i] = {"evidence": ev, "pass": p}
    if unevidenced:
        raise SystemExit(f"decided with no evidence quoted: {unevidenced}; "
                         f"quote the trajectory or mark the rubric undecided")
    summary = (ws.get("summary") or "").strip()
    if not summary:
        raise SystemExit("summary is empty; one sentence on what the review found")

    out = {
        "overall_pass": None if undecided else all(v["pass"] for v in verdicts.values()),
        "summary": summary,
        "verdicts": dict(sorted(verdicts.items())),
    }
    (d / "rubric_verdicts.json").write_text(json.dumps(out, indent=1) + "\n")
    if undecided:
        print(f"recorded as indeterminate; undecided: {undecided}", file=sys.stderr)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="process-rubric review console (veto-only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("prepare", "write the evidence packet and a blank worksheet"),
                        ("record", "validate a filled worksheet into rubric_verdicts.json")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("attempt_dir")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(pathlib.Path(a.attempt_dir))
    else:
        print(json.dumps(record(pathlib.Path(a.attempt_dir)), indent=2)[:1200])
