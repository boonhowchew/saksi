# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Reviewer Assistant: helps the human reviewer examine the draft. Read-only.

Independent of the authors' side by construction:
  - a different model on a different deployment (Muse Spark Open, open weights, on Baseten);
  - a fresh conversation for every briefing, with no memory of earlier calls;
  - it receives only what a reviewer is entitled to see (SEES below), and the briefing
    records the hash of every input, so anyone can check what it was given.

It cannot edit the draft, run code, or decide anything. It asks questions; the
reviewer decides which to put to the authors, in their own words.

Its output is checked by code before the reviewer sees it: every quotation must be
found in the draft, and item assessments must refer to real items.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any

from .agents import AgentError, parse_json, protocol_text
from .documents import quote_found
from .llm import ModelClient
from .prompts import REVIEW_ASSIST, REVIEW_FOLLOWUP, prompt_ref
from .tracing import interval_facts
from .workflow import Project, State, WorkflowError, summarise_checks

REVIEWER_ASSIST_ID = "agent:reviewer_assist"
PROVIDER = "baseten"

SEES = ("the draft under review", "the protocol", "the analysis plan every author confirmed",
        "computed results with their confirmatory/exploratory labels",
        "interval facts computed by code (does each interval include its null value)", "the analysis code as it ran",
        "planned analyses that failed", "the aggregate data dictionary (no records)",
        "the automatic number check of this version",
        "in later rounds: the review items, the authors' responses and the changes since the last round")
WITHHELD = ("any data record (it never opens the dataset)", "authors' sign-off answers and claim restatements",
            "draft edit summaries and who made each edit", "the Writing Agent's attempts and prompts",
            "authors' analysis requests as worded", "any other model call", "the Contribution Ledger")

CATEGORIES = {"protocol_adherence", "interpretation", "statistics", "reporting", "limitations", "other"}
ASSESSMENTS = {"addressed", "partly_addressed", "not_addressed", "cannot_tell"}
MAX_POINTS = 10
SETTINGS = {"temperature": 0.2, "max_tokens": 16000, "timeout": 600.0}   # it always reasons before answering
CODE_CHARS, DIFF_CHARS, PROTOCOL_CHARS = 6000, 20000, 40000


@dataclass
class ReviewerAssistant:
    project: Project
    client: ModelClient

    def brief(self, requested_by: str) -> State:
        s = self.project.state()
        if s.stage != "REVIEW":
            raise WorkflowError(f"the Reviewer Assistant works in REVIEW; the project is in {s.stage}")
        if requested_by not in s.reviewers:
            raise WorkflowError("only a reviewer can ask the Reviewer Assistant for a briefing")
        user, inputs, to_assess, facts = self._materials(s)
        followup = bool(to_assess) or s.round > 1
        system = REVIEW_ASSIST + ("\n\n" + REVIEW_FOLLOWUP if followup else "")
        r = self.client.chat(PROVIDER, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                             purpose=f"review_brief:round{s.round}", **SETTINGS)
        call = r.record.to_payload()
        try:
            out = parse_json(r.text)
        except AgentError:
            self.project.record_model_call(REVIEWER_ASSIST_ID, call)
            raise
        draft = self.project.blobs.get_text(s.drafts[s.draft_version]["text_sha256"])
        briefing = {
            "round": s.round, "draft_version": s.draft_version,
            "primary_claim": str(out.get("primary_claim", "")), "summary": str(out.get("summary", "")),
            **check_points(out.get("points_of_inquiry"), draft),
            **check_assessments(out.get("item_assessments"), to_assess, draft),
            "automatic_number_check": s.drafts[s.draft_version].get("checks"),
            "interval_facts": facts,
            "inputs": inputs, "sees": list(SEES), "withheld": list(WITHHELD),
            "prompts": [prompt_ref("review_assist")] + ([prompt_ref("review_followup")] if followup else []),
        }
        return self.project.add_briefing(REVIEWER_ASSIST_ID, briefing, call, requested_by=requested_by)

    # -- what it is given ------------------------------------------------------
    def _materials(self, s: State) -> tuple[str, dict[str, Any], list[str], list[dict[str, Any]]]:
        blobs = self.project.blobs
        d = s.drafts[s.draft_version]
        draft = blobs.get_text(d["text_sha256"])
        protocol, pm = protocol_text(self.project, s)
        plan = {aid: {k: v for k, v in a.items() if k not in ("prompt",)}
                for aid, a in (s.plan_locked or {}).get("analyses", {}).items()}
        ok = {k: v for k, v in s.analyses.items() if v["status"] == "ok" and blobs.has(v.get("outputs_sha256") or "")}
        results = {k: {"label": v["label"], "basis": v.get("basis"), "plan_analysis_id": v["plan_analysis_id"],
                       "title": v["spec"].get("title"), "sample_check": v.get("sample_check"),
                       "results": blobs.get_json(v["outputs_sha256"])}
                   for k, v in ok.items()}
        facts = interval_facts({k: v["results"] for k, v in results.items()})
        code = {k: blobs.get_text(v["code_sha256"])[:CODE_CHARS] for k, v in ok.items() if blobs.has(v["code_sha256"])}
        failed = {k: {"title": v["spec"].get("title"), "error": (v.get("error") or "")[:300]}
                  for k, v in s.analyses.items() if v["status"] != "ok" and v["plan_analysis_id"]
                  and not any(o["plan_analysis_id"] == v["plan_analysis_id"] for o in ok.values())}
        dictionary, report_sha = None, None
        if s.curator_reports:                      # the curator's stored report; the dataset itself is never opened
            report_sha = s.curator_reports[-1]["report_sha256"]
            rep = blobs.get_json(report_sha)
            dictionary = {"rows": rep.get("profile", {}).get("rows"), "dictionary": rep.get("dictionary")}
        checks = blobs.get_json(d["checks_sha256"]) if d.get("checks_sha256") else None

        parts = [f"DRAFT UNDER REVIEW (version {s.draft_version}):\n{draft}",
                 f"PROTOCOL:\n{protocol[:PROTOCOL_CHARS]}",
                 f"ANALYSIS PLAN CONFIRMED BY EVERY AUTHOR BEFORE ANY RESULT:\n{json.dumps(plan, indent=1)}",
                 f"COMPUTED RESULTS (draft cites them as [A:<key>]):\n{json.dumps(results, indent=1)}",
                 f"INTERVAL FACTS (computed by code):\n{json.dumps(facts, indent=1)}",
                 "ANALYSIS CODE AS RUN:\n" + "\n\n".join(f"# [A:{k}]\n{c}" for k, c in code.items()),
                 f"PLANNED ANALYSES THAT FAILED:\n{json.dumps(failed, indent=1)}",
                 "FIGURES (captions only; drawn by code, never by a model):\n" + ("\n".join(
                     f"Figure {f['number']} [A:{f['analysis_id']}], drawn by {f['drawn_by']}: {f['caption']}"
                     for f in s.figures if f["analysis_id"] in ok) or "(none)"),
                 f"DATA DICTIONARY (aggregate):\n{json.dumps(dictionary, indent=1)}",
                 f"AUTOMATIC NUMBER CHECK OF THIS VERSION:\n{json.dumps(_check_brief(checks), indent=1)}"]

        to_assess = sorted(i for i, it in s.items.items() if it["status"] == "responded")
        prev = s.reviewed_versions.get(s.round - 1)
        if s.items:
            items = {i: {"comment": it["comment"], "location": it.get("location"), "status": it["status"],
                         "responses": [{"response": r["response"], "note": r.get("note", ""),
                                        "draft_version": r["draft_version"]} for r in it["responses"]],
                         "verdicts": [{"verdict": v["verdict"], "note": v.get("note", "")} for v in it["verdicts"]]}
                     for i, it in sorted(s.items.items())}
            parts.append(f"REVIEW ITEMS WITH AUTHORS' RESPONSES:\n{json.dumps(items, indent=1)}")
            parts.append("ITEMS TO ASSESS: " + (", ".join(to_assess) or "(none)"))
        if prev and prev != s.draft_version:
            old = blobs.get_text(s.drafts[prev]["text_sha256"])
            diff = "".join(difflib.unified_diff(old.splitlines(True), draft.splitlines(True),
                                                f"version {prev}", f"version {s.draft_version}"))
            parts.append(f"CHANGES SINCE THE REVIEWER LAST SAW THE DRAFT:\n{diff[:DIFF_CHARS]}")

        inputs = {"draft_version": s.draft_version, "draft_sha256": d["text_sha256"], "protocol_sha256": pm["sha256"],
                  "plan_sha256": (s.plan_locked or {}).get("sha"),
                  "analyses": {k: {"outputs_sha256": v["outputs_sha256"], "code_sha256": v["code_sha256"]}
                               for k, v in ok.items()},
                  "curator_report_sha256": report_sha, "checks_sha256": d.get("checks_sha256"),
                  "previous_version": prev, "items": sorted(s.items)}
        return "\n\n".join(parts), inputs, to_assess, facts


# -- checks on what it returns ---------------------------------------------------
def check_points(raw: Any, draft: str) -> dict[str, Any]:
    points = []
    for p in [x for x in (raw if isinstance(raw, list) else []) if isinstance(x, dict)][:MAX_POINTS]:
        quote = str(p.get("quote") or "")
        points.append({"id": f"P{len(points) + 1}",
                       "category": p.get("category") if p.get("category") in CATEGORIES else "other",
                       "severity": p.get("severity") if p.get("severity") in ("major", "minor") else "minor",
                       "quote": quote, "quote_verified": bool(quote) and quote_found(quote, draft),
                       "question": str(p.get("question") or ""), "why_it_matters": str(p.get("why_it_matters") or "")})
    dropped = max(0, sum(isinstance(x, dict) for x in raw) - MAX_POINTS) if isinstance(raw, list) else 0
    return {"points_of_inquiry": points, "points_dropped_over_limit": dropped}


def check_assessments(raw: Any, to_assess: list[str], draft: str) -> dict[str, Any]:
    out, unknown = [], []
    for a in raw if isinstance(raw, list) else []:
        if not isinstance(a, dict):
            continue
        iid = str(a.get("item_id") or "")
        if iid not in to_assess:
            unknown.append(iid)
            continue
        quote = str(a.get("quote") or "")
        out.append({"item_id": iid,
                    "assessment": a.get("assessment") if a.get("assessment") in ASSESSMENTS else "cannot_tell",
                    "quote": quote, "quote_verified": bool(quote) and quote_found(quote, draft),
                    "note": str(a.get("note") or "")})
    seen = {a["item_id"] for a in out}
    return {"item_assessments": out, "items_not_assessed": [i for i in to_assess if i not in seen],
            "unknown_item_ids": unknown}


def _check_brief(checks: dict[str, Any] | None) -> dict[str, Any] | None:
    if checks is None:
        return None
    return {**summarise_checks(checks),
            "untraced_numbers": [{"number": u["number"], "sentence": u["sentence"][:200]}
                                 for u in checks.get("untraced", [])[:15]]}
