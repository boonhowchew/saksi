# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Confirmatory or exploratory: decided by the engine from the protocol, before any result.

An analysis is CONFIRMATORY only if all of these hold (ICH E9 sense):
  1. it is in the analysis plan every author confirmed before any result existed;
  2. the protocol names it as the primary hypothesis (a verbatim quotation, found in the protocol);
  3. the protocol's sample-size justification covers it (a verbatim quotation, found in the protocol),
     and any numeric target given for it appears in that quotation.
Anything else is EXPLORATORY: "pre-specified" if it is in the plan, "post hoc" if it is not.

The quotations are checked here, by code, against the stored protocol, whoever proposed
the plan. A missing or unfound quotation fails safe to exploratory.

After the analysis runs, the achieved sample (and events) is compared with the protocol's
target. A shortfall does not change the label, which was fixed before any result; it is
flagged, and the draft must state it.
"""

from __future__ import annotations

import re
from typing import Any

from .documents import locate

ROLES = {"primary", "secondary", "other"}
_INT = re.compile(r"\d[\d,]*")


def _ints(text: str) -> set[int]:
    return {int(m.group(0).replace(",", "")) for m in _INT.finditer(text or "")}


def _target(raw: Any) -> dict[str, int | None]:
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for k in ("n", "events"):
        v = raw.get(k)
        out[k] = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None
    return out


def _found(quote: str, text: str) -> dict[str, Any]:
    if not quote.strip():
        return {"given": False, "status": "not_given"}
    if not text:
        return {"given": True, "status": "not_found", "why": "the protocol's text could not be read"}
    return {"given": True, **locate(quote, text)}


def assess(analysis: dict[str, Any], protocol_text: str | None) -> dict[str, Any]:
    """Design assessment for one planned analysis. Overrides anything the proposer claimed."""
    text = protocol_text or ""
    role = analysis.get("role") if analysis.get("role") in ROLES else "other"
    primary_q = str(analysis.get("primary_quote") or "")
    size_q = str(analysis.get("sample_size_quote") or "")
    target = _target(analysis.get("sample_size_target"))
    described = _found(str(analysis.get("protocol_quote") or ""), text)
    primary = _found(primary_q, text) if role == "primary" else {"given": bool(primary_q), "status": "not_checked"}
    size = _found(size_q, text)
    ok = lambda f: f["status"] in ("exact", "close")
    # Planned numbers must appear in the protocol's own wording of the justification.
    size_words = size.get("protocol_text") or size_q
    targets_ok = all(v in _ints(size_words) for v in target.values() if v is not None)

    def detail(f: dict[str, Any], what: str) -> str:
        if f["status"] == "exact":
            return "found in the protocol"
        if f["status"] == "close":
            return "found in the protocol with small differences; the protocol's wording is shown"
        if f["status"] == "not_given":
            return f"no {what} given. If the protocol has one, edit the plan and paste it"
        return (f"not found in the protocol ({f.get('why', 'not in the protocol')}). "
                "Edit the plan and paste the protocol's exact words")

    conditions = [
        {"key": "described", "label": "Described in the protocol", "met": ok(described),
         "detail": detail(described, "protocol sentence describing it")},
        {"key": "primary", "label": "Named as the primary analysis in the protocol",
         "met": role == "primary" and ok(primary),
         "detail": (detail(primary, "sentence naming it primary") if role == "primary" else
                    f"listed as a {role} analysis. If the protocol names it primary, set its role to primary "
                    "and paste that sentence")},
        {"key": "powered", "label": "Covered by the protocol's sample-size calculation",
         "met": ok(size) and targets_ok,
         "detail": (detail(size, "sample-size justification") if not ok(size) else
                    "found in the protocol" + ("" if targets_ok else
                    ", but the planned numbers are not in that passage. Correct the planned numbers"))},
    ]
    if not text:
        for c in conditions:
            c["met"], c["detail"] = False, "Saksi could not read the protocol's text, so nothing can be checked"
    confirmatory = all(c["met"] for c in conditions)
    reasons = [f"{c['label'].lower()}: no ({c['detail']})" for c in conditions if not c["met"]]
    checks = {"protocol_available": bool(text), "protocol_quote_verified": ok(described),
              "primary_quote_verified": role == "primary" and ok(primary), "sample_size_quote_verified": ok(size),
              "targets_in_quote": targets_ok}
    return {"role": role, "label": "confirmatory" if confirmatory else "exploratory", "basis": "pre-specified",
            "primary_quote": primary_q, "sample_size_quote": size_q, "sample_size_target": target,
            "quotes": {"described": described, "primary": primary, "sample_size": size},
            "conditions": conditions, "checks": checks, "reasons_not_confirmatory": reasons}


def sample_check(design: dict[str, Any] | None, outputs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Achieved versus planned sample for a confirmatory analysis. None when there is no target."""
    if not design or design.get("label") != "confirmatory" or not isinstance(outputs, dict):
        return None
    t = design.get("sample_size_target") or {}
    if t.get("n") is None and t.get("events") is None:
        return None
    achieved = {"n": outputs.get("n_analysed"), "events": outputs.get("n_events")}
    rows, shortfall, unknown = {}, False, []
    for k in ("n", "events"):
        if t.get(k) is None:
            continue
        a = achieved[k]
        if not isinstance(a, (int, float)) or isinstance(a, bool):
            unknown.append(k)
            rows[k] = {"target": t[k], "achieved": None}
            continue
        rows[k] = {"target": t[k], "achieved": a, "short": a < t[k]}
        shortfall = shortfall or a < t[k]
    return {"targets": rows, "shortfall": shortfall, "not_reported_by_analysis": unknown}
