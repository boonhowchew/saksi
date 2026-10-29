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

from .documents import quote_found

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


def assess(analysis: dict[str, Any], protocol_text: str | None) -> dict[str, Any]:
    """Design assessment for one planned analysis. Overrides anything the proposer claimed."""
    text = protocol_text or ""
    role = analysis.get("role") if analysis.get("role") in ROLES else "other"
    primary_q = str(analysis.get("primary_quote") or "")
    size_q = str(analysis.get("sample_size_quote") or "")
    target = _target(analysis.get("sample_size_target"))
    checks = {
        "protocol_available": bool(text),
        "protocol_quote_verified": bool(text) and quote_found(str(analysis.get("protocol_quote") or ""), text),
        "primary_quote_verified": role == "primary" and bool(text) and quote_found(primary_q, text),
        "sample_size_quote_verified": bool(text) and bool(size_q) and quote_found(size_q, text),
        "targets_in_quote": all(v in _ints(size_q) for v in target.values() if v is not None),
    }
    reasons = []
    if not text:
        reasons.append("the protocol text is not available to check against")
    if role != "primary":
        reasons.append(f"the plan lists it as a {role} analysis, not the primary hypothesis")
    elif not checks["primary_quote_verified"]:
        reasons.append("no quotation from the protocol naming it as primary was found in the protocol")
    if not checks["sample_size_quote_verified"]:
        reasons.append("no sample-size justification quoted from the protocol covers it")
    elif not checks["targets_in_quote"]:
        reasons.append("the numeric sample-size target does not appear in the quoted justification")
    if not checks["protocol_quote_verified"]:
        reasons.append("the quotation specifying the analysis was not found in the protocol")
    confirmatory = not reasons
    return {"role": role, "label": "confirmatory" if confirmatory else "exploratory", "basis": "pre-specified",
            "primary_quote": primary_q, "sample_size_quote": size_q, "sample_size_target": target,
            "checks": checks, "reasons_not_confirmatory": reasons}


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
