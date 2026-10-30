# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Trace every number in a draft back to a computed result.

Deterministic, no model involved. For each sentence:
  * every number must match, at the precision written, a value produced by an
    analysis the sentence cites ([A:analysis_id]), or a dataset count;
  * a sentence citing an exploratory analysis must say "exploratory".
Anything that fails is reported; nothing is silently corrected.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

CITE = re.compile(r"\[A:([A-Za-z0-9_.\-]+)\]")
NUM = re.compile(r"(?<![\w.])(<\s*)?(-?\d+(?:,\d{3})*(?:\.\d+)?)(\s*%)?")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n+")
# Numbers that are conventions, not results.
CONVENTIONAL = (re.compile(r"\b95\s*%\s*(?:CI|confidence)", re.I), re.compile(r"\b(?:19|20)\d{2}\b"),
                re.compile(r"\bp\s*<\s*0?\.05\b", re.I), re.compile(r"\balpha\b.*0\.05", re.I),
                re.compile(r"\b(?:Fig(?:ure)?s?\.?|Tables?)\s*S?\d+(?:\s*(?:and|,|-|to)\s*S?\d+)*\b", re.I))


def numeric_leaves(obj: Any) -> Iterable[float]:
    """Numbers in an analysis result, including numbers written inside its text
    fields (e.g. notes produced by the analysis code: "18 GDM events")."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, str):
        for m in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?", obj):
            yield float(m.group(0))
        return
    if isinstance(obj, (int, float)):
        if not (isinstance(obj, float) and math.isnan(obj)):
            yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from numeric_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from numeric_leaves(v)


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def _matches(token: str, is_less_than: bool, is_percent: bool, values: list[float]) -> str | None:
    x = float(token.replace(",", ""))
    d = _decimals(token)
    for v in values:
        if is_less_than and 0 <= v < x:
            return f"{v:.3g} < {token}"
        candidates = [v, v * 100] if is_percent else [v]
        for c in candidates:
            if round(c, d) == x or (d == 0 and abs(c - x) < 0.5 and float(int(x)) == x and round(c) == x):
                return f"{c:.6g}"
    return None


def _conventional_spans(sentence: str) -> list[tuple[int, int]]:
    spans = []
    for pat in CONVENTIONAL:
        spans += [m.span() for m in pat.finditer(sentence)]
    return spans


def trace(text: str, analyses: dict[str, dict[str, Any]], outputs: dict[str, Any],
          dataset_counts: Iterable[float] = (), required_refs: Iterable[str] = (),
          must_state: dict[str, list[float]] | None = None,
          must_disclose: Iterable[str] = ()) -> dict[str, Any]:
    """analyses: id -> analysis record (needs 'label'); outputs: id -> results JSON.
    required_refs: analyses the draft must report (guards against selective reporting).
    must_state: id -> numbers that a sentence citing that analysis must state (e.g. the planned
    sample size when the achieved sample fell short of it).
    must_disclose: analyses whose primary role departs from the protocol; some sentence citing each must
    say so (it must mention both "protocol" and "primary"). A minimal check; the reviewer judges the rest."""
    counts = [float(c) for c in dataset_counts]
    rows, untraced, unlabelled, unknown_refs = [], [], [], set()
    stated: dict[str, set[float]] = {}
    disclosed: set[str] = set()
    for para in text.split("\n\n"):
        # Drop heading LINES only. A heading followed directly by text on the next
        # line is common in model output; skipping the whole paragraph would hide it.
        para = "\n".join(ln for ln in para.splitlines() if not ln.lstrip().startswith("#"))
        for sent in filter(None, (x.strip() for x in SENTENCE.split(para))):
            refs = CITE.findall(sent)
            unknown_refs.update(r for r in refs if r not in analyses)
            low = sent.lower()
            if "protocol" in low and "primary" in low:
                disclosed.update(refs)
            pool = [v for r in refs if r in outputs for v in numeric_leaves(outputs[r])] + counts
            plain = CITE.sub("", sent)
            skip = _conventional_spans(plain)
            for m in NUM.finditer(plain):
                if any(a <= m.start() < b for a, b in skip):
                    continue
                token = m.group(2)
                for r in refs:
                    stated.setdefault(r, set()).add(float(token.replace(",", "")))
                hit = _matches(token, bool(m.group(1)), bool(m.group(3)), pool)
                row = {"number": m.group(0).strip(), "sentence": sent[:300], "cited": refs,
                       "status": "traced" if hit else "untraced", "matched": hit}
                rows.append(row)
                if not hit:
                    untraced.append(row)
            exploratory = [r for r in refs if analyses.get(r, {}).get("label") == "exploratory"]
            if exploratory and "exploratory" not in sent.lower() and "exploratory" not in para.lower():
                unlabelled.append({"sentence": sent[:300], "analyses": exploratory})
    # Fail safe: a check that found nothing to check must not report a pass when
    # the body text plainly contains numbers.
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    body = CITE.sub("", body)
    for pat in CONVENTIONAL:
        body = pat.sub("", body)
    nothing_checked = not rows and bool(re.search(r"\d", body))
    not_reported = sorted(set(required_refs) - set(CITE.findall(text)))
    shortfall_unreported = sorted(k for k, vals in (must_state or {}).items()
                                  if any(float(v) not in stated.get(k, set()) for v in vals))
    departure_undisclosed = sorted(set(must_disclose) - disclosed)
    return {
        "numbers_checked": len(rows), "traced": len(rows) - len(untraced), "untraced": untraced,
        "exploratory_not_labelled": unlabelled, "unknown_analysis_refs": sorted(unknown_refs),
        "nothing_checked_but_numbers_present": nothing_checked, "analyses_not_reported": not_reported,
        "shortfall_not_reported": shortfall_unreported, "departure_not_disclosed": departure_undisclosed,
        "ok": not (untraced or unlabelled or unknown_refs or nothing_checked or not_reported or shortfall_unreported
                   or departure_undisclosed),
    }


# Null value by measure, for "does the interval include no effect?". Order matters:
# a log ratio has null 0, so it is tested before plain ratios.
NULL_VALUES = (
    (re.compile(r"\blog\b.*ratio|\blog[- ]?(?:odds|or|hr|rr)\b", re.I), 0.0),
    (re.compile(r"ratio|\b(?:a?OR|RR|HR|IRR)\b", re.I), 1.0),
    (re.compile(r"\bAUC\b|\bAUROC\b|c-?statistic|area under|concordance", re.I), 0.5),
    (re.compile(r"difference|coefficient|\bbeta\b|slope|mean change", re.I), 0.0),
)


def interval_facts(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    """For every estimate with a confidence interval, whether the interval includes
    the null value for its measure. Computed here so no model has to compare numbers.
    Estimates whose measure is not recognised are left out rather than guessed."""
    facts = []
    for aid, out in sorted(outputs.items()):
        for e in (out or {}).get("estimates") or []:
            lo, hi = e.get("ci_lower"), e.get("ci_upper")
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (lo, hi)):
                continue
            label = f"{e.get('measure') or ''} {e.get('name') or ''}"
            null = next((v for pat, v in NULL_VALUES if pat.search(label)), None)
            if null is None:
                continue
            facts.append({"analysis": aid, "name": e.get("name"), "measure": e.get("measure"),
                          "estimate": e.get("estimate"), "ci_lower": lo, "ci_upper": hi, "null_value": null,
                          "interval_includes_null": bool(lo <= null <= hi)})
    return facts
