# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""What Saksi can suggest at sign-off, from the record, so authors confirm rather than recall.

- Answers to the sign-off questions that code can establish (is the primary claim protocol-specified; are the
  exploratory analyses identified), from the analyses and the number check of the version being signed.
- CRediT roles for each author, from what the record shows they did (uploading the data, settling the plan,
  asking for analyses, accepting the results, editing the text...). Lead for whoever did most of a role.
- For each verification area, the evidence in the record (opened the version, accepted the results, ...).

Everything here is a suggestion: the author agrees, changes or rejects it, and the record keeps both.
"""

from __future__ import annotations

from typing import Any

from .workflow import EVIDENCE_AREAS, SIGNOFF_QUESTIONS, current

# Event type -> CRediT roles it is evidence of, for the human who did it (or asked for it).
ROLE_EVIDENCE = {
    "material.uploaded:dataset": ["Data curation", "Resources"],
    "material.uploaded:protocol": ["Conceptualization", "Methodology"],
    "material.uploaded:other": ["Resources"],
    "data.prepared": ["Data curation"],
    "question.answered": ["Methodology"],
    "plan.proposed": ["Methodology"],
    "plan.confirmed": ["Methodology"],
    "plan.outline_edited": ["Methodology", "Writing – review & editing"],
    "analysis.requested": ["Formal analysis"],
    "analysis.completed": ["Formal analysis"],
    "analyses.accepted": ["Validation"],
    "analysis.described": ["Writing – review & editing"],
    "draft.edited": ["Writing – review & editing"],
    "figure.omitted": ["Visualization"],
    "figure.restored": ["Visualization"],
    "labels.set": ["Visualization"],
    "analysis.placed": ["Visualization"],
    "references.found": ["Investigation"],
    "reference.removed": ["Investigation"],
    "response.submitted": ["Writing – review & editing"],
    "review.item_responded": ["Writing – review & editing"],
    "project.created": ["Project administration"],
}


def _who(ev: Any, humans: set[str]) -> str | None:
    if ev.actor_id in humans:
        return ev.actor_id
    rb = (ev.payload or {}).get("requested_by")
    return rb if rb in humans else None


def suggest(project: Any, version: int | None = None) -> dict[str, Any]:
    s = project.state()
    events = project.log.events()
    authors = list(s.authors)
    humans = set(authors)
    counts: dict[str, dict[str, int]] = {a: {} for a in authors}
    for ev in events:
        key = ev.type
        if ev.type == "material.uploaded":
            k = (ev.payload or {}).get("kind")
            key = f"material.uploaded:{k if k in ('dataset', 'protocol') else 'other'}"
        if ev.type == "analysis.described" and not (ev.payload or {}).get("edited_by_author"):
            continue
        if ev.type == "material.uploaded" and (ev.payload or {}).get("prepared_from"):
            key = "data.prepared"
        who = ev.actor_id if ev.type == "project.created" else _who(ev, humans)
        if ev.type == "project.created":
            who = next((a for a in authors if "corresponding_author" in (s.actors.get(a) or {}).get("roles", [])), None)
        if who not in counts:
            continue
        for role in ROLE_EVIDENCE.get(key, []):
            counts[who][role] = counts[who].get(role, 0) + 1
    credit: dict[str, dict[str, str]] = {a: {} for a in authors}
    roles = {r for c in counts.values() for r in c}
    for r in roles:
        top = max(c.get(r, 0) for c in counts.values())
        leaders = [a for a in authors if counts[a].get(r, 0) == top]
        for a in authors:
            if counts[a].get(r):
                credit[a][r] = ("equal" if len(leaders) > 1 else "lead") if a in leaders else "supporting"

    # Evidence for each verification area, from the record.
    v = version or s.draft_version
    evidence: dict[str, dict[str, list[str]]] = {a: {k: [] for k, _, _ in EVIDENCE_AREAS} for a in authors}
    suggested: dict[str, dict[str, str]] = {a: {} for a in authors}
    refs = [x for x in s.references.values() if not x.get("removed")]
    checks = (s.drafts.get(v) or {}).get("checks") or {}
    for a in authors:
        e = evidence[a]
        if refs:
            e["sources"].append(f"Saksi fetched {len(refs)} references from PubMed and checked each cited finding "
                                f"against its abstract; whether you opened them is yours to say.")
        if (a, v) in s.opened:
            e["facts"].append(f"You opened version {v} on {s.opened[(a, v)][:16].replace('T', ' ')}.")
            e["reasoning"].append(f"You opened version {v}; the Discussion's interpretation is yours to judge.")
        if any(a in acc for acc in s.analyses_accepted_by.values()):
            e["data"].append("You accepted the results on the Analyses page.")
            suggested[a]["data"] = "confirmed"
        if checks:
            e["data"].append(f"Code checked {checks.get('numbers_checked', 0)} numbers in this version against the "
                             f"computed results{' and all were traced' if checks.get('ok') else '; some were not traced'}.")
        if a in s.plan_confirmed_by or (s.plan_locked and a in s.authors):
            e["methods"].append("You confirmed the analysis plan before any analysis ran; every analysis's code, "
                                "data fingerprint and environment are in the record and can be rerun.")
        e["integrity"].append("Signing off states your final approval of this version and your responsibility for it.")

    # Answers code can establish.
    cur = current(s)
    conf = [k for k, x in cur.items() if x.get("label") == "confirmatory"]
    if conf:
        q2 = ("Yes. The claims about the primary result rest on " + ", ".join(conf) + ", the protocol's pre-specified "
              "primary analysis, which reached its planned sample.")
    else:
        q2 = ("There is no confirmatory claim: no analysis was both the protocol's pre-specified primary analysis and "
              "at its planned sample, so the draft reports every finding as exploratory.")
    n_unl = checks.get("exploratory_not_labelled") or 0
    q3 = ("Yes. Code checked that each exploratory analysis is identified as exploratory in this version."
          if not n_unl else f"No. Code found {n_unl} exploratory analys{'es' if n_unl > 1 else 'is'} not identified as "
          "exploratory; see the number check above the draft.")
    model = next((x for x in reversed(getattr(s, "signoff_suggestions", []) or []) if x.get("version") == v), None)
    return {"version": v, "answers": {SIGNOFF_QUESTIONS[1]: q2, SIGNOFF_QUESTIONS[2]: q3},
            "claim": (model or {}).get("claim"), "least_comfortable": (model or {}).get("least_comfortable"),
            "credit": credit, "evidence": evidence, "verification": suggested}


def contributions_statement(s: Any) -> dict[str, Any]:
    """Author contributions (CRediT) and the use of AI, from the latest sign-offs, ready for the manuscript."""
    v = max((ver for ver, so in s.signoffs.items() if so), default=None)
    rows = []
    for a in s.authors:
        so = (s.signoffs.get(v) or {}).get(a) if v else None
        if not so or not so.get("disclosure"):
            continue
        name = (s.actors.get(a) or {}).get("name") or a
        roles = "; ".join(f"{r} ({d})" for r, d in so["disclosure"]["credit"].items())
        rows.append({"author": name, "roles": roles, "guarantorship": so["disclosure"]["guarantorship"],
                     "verification": so["disclosure"]["verification"]})
    agents = [x for x in s.actors.values() if x.get("kind") == "agent" or str(x.get("model") or "")]
    ai = ("Saksi, an AI-verified research platform, was used for data checking (Data Curator), analysis code written "
          "and run in an isolated sandbox (Analysis Agent), and drafting from the computed results (Writing Agent)"
          + (", with a Reviewer Assistant (a different, open-weights model) supporting peer review"
             if any(x.get("role") == "reviewer_assist" for x in agents) else "")
          + ". Models: " + ", ".join(sorted({f"{x.get('model')}" for x in agents if x.get("model")})) + ". "
          "Every number in the text was checked by code against the computed results, every step is recorded in a "
          "hash-chained log, and the authors reviewed, edited and take responsibility for the content.")
    return {"version": v, "authors": rows, "ai_use": ai}
