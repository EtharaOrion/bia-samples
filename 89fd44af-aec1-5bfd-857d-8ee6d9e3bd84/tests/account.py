"""Canonical account extractor. ONE implementation, used by every lane.

Rationale: four separate measurements of this corpus were wrong because each guessed
at JSON key names ('.steps[].message', '.arguments.content', ...) and silently returned
empty for layouts that did not match. Bundle A has no findings.md; bundle B sometimes does.
This module never guesses a key: it walks EVERY string in EVERY file under an attempt
directory. It is deliberately over-inclusive - for verbatim-quote checking, over-inclusion
can only make a check more permissive, never falsely accuse.
"""
import json, os, re, unicodedata

SKIP_PREFIX = ("rubric_verdicts",)

def _walk(o, buf):
    if isinstance(o, dict):
        for v in o.values(): _walk(v, buf)
    elif isinstance(o, list):
        for v in o: _walk(v, buf)
    elif isinstance(o, str):
        buf.append(o)

def raw(attempt_dir):
    """Every string under the attempt, excluding the verdict file itself."""
    buf = []
    for root, _, files in os.walk(attempt_dir):
        for f in sorted(files):
            if f.startswith(SKIP_PREFIX): continue
            p = os.path.join(root, f)
            try:
                if f.endswith(".json"):
                    _walk(json.load(open(p, encoding="utf-8")), buf)
                elif f.endswith(".jsonl"):
                    for ln in open(p, encoding="utf-8"):
                        try: _walk(json.loads(ln), buf)
                        except Exception: buf.append(ln)
                else:
                    buf.append(open(p, encoding="utf-8", errors="replace").read())
            except Exception:
                pass
    return "\n".join(buf)

def norm(s):    return unicodedata.normalize("NFC", s)
def ws(s):      return re.sub(r"\s+", " ", norm(s)).strip()

def account(attempt_dir):
    """Whitespace-normalised, NFC-normalised full account text."""
    return ws(raw(attempt_dir))

def quoted_fragments(evidence, rubric_text, historical_rubric_text=""):
    """Fragments an evidence string claims to quote FROM THE RECORD.

    Excludes spans that merely restate the rubric (evidence often quotes the rubric it
    is arguing about), and splits '...'/'…' elisions into separately-checkable fragments.
    """
    ev, rub = norm(evidence or ""), norm(rubric_text or "")
    hist = ws(historical_rubric_text or "")
    out = []
    parts = ev.split('"')
    for i in range(1, len(parts), 2):
        span = parts[i].strip()
        if len(span) < 25 or span in rub:
            continue
        # evidence often argues against the rubric it is judging, including the
        # PRE-SPLIT parent wording. Those are rubric quotes, not record quotes.
        if hist and ws(span) in hist:
            continue
        for frag in re.split(r"\s*(?:\.\.\.|…)\s*", span):
            frag = ws(frag)
            if len(frag) >= 25:
                out.append(frag)
    return out
