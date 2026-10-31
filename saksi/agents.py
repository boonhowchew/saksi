# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Agents that act on a project: Data Curator, Analysis Agent and Writing Agent.
The Reviewer Assistant is in reviewer.py.

Every agent action is requested by a named author and is written to the
event log by the workflow engine, with the model call that produced it.

What each model sees:
  Curator  - aggregate dataset profile (profiling.py) + protocol text
  Analyst  - protocol text + data dictionary/profile + one analysis spec
  Writer   - protocol text + data dictionary + computed results
None of them ever receives a data row. Raw data only enters the sandbox.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .documents import candidates, extract_text, locate, quote_found
from .llm import LLMError, ModelClient
from .profiling import profile, read_table
from .prompts import (ASSIST, CHECKLIST, CODE, CURATOR, CURATOR_UPDATE, DATA_PREP, OUTLINE, PLAN_ASSIST, SUGGEST_ANSWERS, CURATOR_COLUMNS, CURATOR_OVERVIEW, PLAN, QUOTE_REPAIR, REPAIR, RESULT_TEXT,
                      REWRITE, WRITER, prompt_ref)
from .sandbox import CodeRejected, Executor, check_code
from .canonical import sha256_text
from .figures import required_figure
from .tracing import interval_facts, trace
from . import standard
from .workflow import SYSTEM, Project, State, WorkflowError, current, included_order
from .workflow import answers_text, open_questions, preparation_in_use
from . import cleaning
from .workflow import summarise_checks as summarise
from . import tables as tables_mod
from . import guidelines

CURATOR_ID, ANALYST_ID, WRITER_ID = "agent:curator", "agent:analyst", "agent:writer"

# Reasoning effort and output budget per task. Reasoning tokens are billed, so
# effort is spent only where judgement matters.
TASKS = {
    "curate": {"reasoning_effort": "low", "max_tokens": 12000},
    "plan": {"reasoning_effort": "high", "max_tokens": 32000},
    "code": {"reasoning_effort": "medium", "max_tokens": 12000},
    "repair": {"reasoning_effort": "medium", "max_tokens": 12000},
    "quotes": {"reasoning_effort": "low", "max_tokens": 8000},
    "write": {"reasoning_effort": "medium", "max_tokens": 16000},
    "describe": {"reasoning_effort": "low", "max_tokens": 3000},
    "assist": {"reasoning_effort": "low", "max_tokens": 2500},
    "checklist": {"reasoning_effort": "low", "max_tokens": 6000},
    "prep": {"reasoning_effort": "medium", "max_tokens": 8000},
}
MAX_ATTEMPTS = 2
PROTOCOL_CHARS = 120_000
CURATE_BATCH = 40                  # columns per Data Curator call on wide datasets
CURATE_PROTOCOL_CHARS = 30_000
FOCUS_ABOVE = 60                   # wider than this, each analysis sees only the columns it names


class AgentError(RuntimeError):
    """The agent could not produce a usable result. Logged calls remain in the log."""


class CutOff(AgentError):
    """The model's reply was cut off at the length limit."""

    def __init__(self, message: str, seqs: list[int]):
        super().__init__(message)
        self.seqs = seqs


def parse_json(text: str) -> dict[str, Any]:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < start:
        raise AgentError("model reply contained no JSON object")
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise AgentError(f"model reply was not valid JSON: {e}") from e


def parse_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def analysis_seed(project_id: str, analysis_id: str) -> int:
    return int(sha256_text(f"{project_id}:{analysis_id}")[:8], 16)


def latest_material(s: State, kind: str, sha: str | None = None) -> dict[str, Any]:
    ms = [m for m in s.materials if m["kind"] == kind and not m.get("withdrawn") and (sha is None or m["sha256"] == sha)]
    if not ms:
        raise AgentError(f"no {kind} uploaded" + (f" with sha {sha[:12]}" if sha else ""))
    return ms[-1]


def material_bytes(project: Project, m: dict[str, Any]) -> bytes:
    """Read a stored material and refuse it if it no longer matches its recorded hash."""
    from .canonical import sha256_bytes
    data = (project.root / m["storage_ref"]).read_bytes()
    if sha256_bytes(data) != m["sha256"]:
        raise AgentError(f"stored {m['kind']} does not match its recorded hash")
    return data


def protocol_text(project: Project, s: State) -> tuple[str, dict[str, Any]]:
    if s.plan_locked is not None and not any(m["kind"] == "protocol" and not m.get("withdrawn") for m in s.materials):
        # Removed by the authors after the plan was confirmed: the confirmed plan keeps its checked quotations.
        return ("(The protocol file was removed by the authors after the plan was confirmed. The confirmed plan below "
                "holds the protocol's quotations, checked against it at the time.)", {"sha256": None, "filename": None})
    m = latest_material(s, "protocol")
    return extract_text(material_bytes(project, m), m["filename"])[:PROTOCOL_CHARS], m


@dataclass
class Agents:
    project: Project
    client: ModelClient
    executor: Executor

    # -- materials ---------------------------------------------------------
    def _latest(self, s: State, kind: str, sha: str | None = None) -> dict[str, Any]:
        return latest_material(s, kind, sha)

    def _bytes(self, m: dict[str, Any]) -> bytes:
        return material_bytes(self.project, m)

    def _protocol_text(self, s: State) -> tuple[str, dict[str, Any]]:
        return protocol_text(self.project, s)

    def _dictionary(self, s: State) -> dict[str, Any]:
        """Latest curator report for the current dataset, else a fresh profile."""
        ds = self._latest(s, "dataset")
        for r in reversed(s.curator_reports):
            if r.get("material_sha256") == ds["sha256"]:
                rep = self.project.blobs.get_json(r["report_sha256"])
                return {"profile": rep["profile"], "dictionary": rep.get("dictionary")}
        if ds.get("prepared_from"):
            # A prepared dataset: the Curator's dictionary of the original, for the columns kept, plus derived columns.
            prof = profile(read_table(self._bytes(ds), ds["filename"]))
            names = {c["name"] for c in prof["columns"]}
            base = next((self.project.blobs.get_json(r["report_sha256"]) for r in reversed(s.curator_reports)
                         if r.get("material_sha256") == ds["prepared_from"]), {})
            dic = [d for d in base.get("dictionary") or [] if d.get("column") in names]
            for st in ds["preparation"]["steps"]:
                if st.get("new") in names:
                    dic.append({"column": st["new"], "label": st.get("label") or st["new"], "type": "binary",
                                "units_or_coding": "0=No, 1=Yes", "role": "derived", "note": st["title"]})
            return {"profile": prof, "dictionary": dic or None}
        return {"profile": profile(read_table(self._bytes(ds), ds["filename"])), "dictionary": None}

    def _chat(self, task: str, system: str, user: str, json_mode: bool, purpose: str):
        cfg = TASKS[task]
        return self.client.chat("meta", [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                purpose=purpose, json_mode=json_mode, temperature=0.0, **cfg)

    # -- Data Curator --------------------------------------------------------
    def _json_call(self, actor: str, task: str, system: str, user: str, purpose: str):
        """Ask for a JSON object. A reply cut off at the length limit raises CutOff; an invalid reply
        gets one retry. Every failed call is logged. Returns (object, last call, logged call seqs)."""
        logged: list[int] = []
        for attempt in (1, 2):
            r = self._chat(task, system, user, json_mode=True, purpose=purpose if attempt == 1 else purpose + ":retry")
            call = r.record.to_payload()
            try:
                return parse_json(r.text), call, logged
            except AgentError as e:
                logged.append(self.project.record_model_call(actor, call))
                finish = ((r.raw.get("choices") or [{}])[0] or {}).get("finish_reason")
                if finish == "length":
                    raise CutOff(f"the reply was cut off at the length limit ({purpose})", logged) from e
                if attempt == 2:
                    raise AgentError(f"{e} (also after one retry)") from e
                user = f"{user}\n\nYOUR PREVIOUS REPLY WAS NOT VALID JSON ({e}). Reply again with only the JSON object."
        raise AssertionError("unreachable")

    def _curate_columns(self, prof: dict[str, Any], batch: list[dict[str, Any]], protocol: str):
        """Dictionary entries and flags for some columns; a cut-off reply splits the batch in two."""
        names = [c["name"] for c in prof["columns"]]
        user = (f"ROWS: {prof['rows']}\nALL COLUMN NAMES: {json.dumps(names)}\n\n"
                f"PROFILE OF THESE COLUMNS (aggregate only):\n{json.dumps(batch, separators=(',', ':'))}\n\n"
                f"PROTOCOL:\n{protocol}")
        try:
            out, call, logged = self._json_call(CURATOR_ID, "curate", CURATOR_COLUMNS, user,
                                                f"curate:columns:{batch[0]['name']}..{batch[-1]['name']}")
        except CutOff as e:
            if len(batch) <= 5:
                raise AgentError("the Data Curator's reply was cut off even for 5 columns") from e
            h = len(batch) // 2
            d1, f1, s1 = self._curate_columns(prof, batch[:h], protocol)
            d2, f2, s2 = self._curate_columns(prof, batch[h:], protocol)
            return d1 + d2, f1 + f2, e.seqs + s1 + s2
        logged.append(self.project.record_model_call(CURATOR_ID, call))
        return list(out.get("dictionary") or []), list(out.get("flags") or []), logged

    def curate(self, requested_by: str, material_sha256: str | None = None) -> State:
        s = self.project.state()
        ds = self._latest(s, "dataset", material_sha256)
        prof = profile(read_table(self._bytes(ds), ds["filename"]))
        try:
            protocol, _ = self._protocol_text(s)
        except AgentError:
            protocol = "(no protocol uploaded yet)"
        protocol = protocol[:CURATE_PROTOCOL_CHARS]
        # What has happened since: the Curator does not raise again what the authors have answered or prepared.
        prep = ds.get("preparation")
        if s.answers or prep:
            protocol += ("\n\nTHE AUTHORS' ANSWERS TO SAKSI'S QUESTIONS (treat these issues as settled):\n" + answers_text(s)
                         + ("\n\nTHIS FILE WAS PREPARED FROM THE ORIGINAL UPLOAD BY THESE STEPS:\n"
                            + "\n".join(f"- {x['title']}: {x['effect']}" for x in prep["log"]) if prep else ""))
        cols, seqs, out, call = prof["columns"], [], None, None
        if len(cols) <= CURATE_BATCH:                       # one call, as before
            user = (f"DATASET PROFILE (aggregate only):\n{json.dumps(prof, indent=1)}\n\nPROTOCOL:\n{protocol}")
            try:
                out, call, logged = self._json_call(CURATOR_ID, "curate", CURATOR, user, "curate")
                seqs += logged
                prompts = [prompt_ref("curator")]
            except CutOff as e:
                seqs += e.seqs                             # fall back to batches
        if out is None:                                     # wide dataset: batches of columns, then an overview
            dictionary, flags = [], []
            for i in range(0, len(cols), CURATE_BATCH):
                d, f, ss = self._curate_columns(prof, cols[i:i + CURATE_BATCH], protocol)
                dictionary, flags, seqs = dictionary + d, flags + f, seqs + ss
            brief = [{"name": c["name"], "kind": c.get("kind"), "missing": c.get("missing")} for c in cols]
            user = (f"ROWS: {prof['rows']}\nCOLUMNS (no values):\n{json.dumps(brief, separators=(',', ':'))}\n\n"
                    f"PROTOCOL:\n{protocol}")
            ov, call, logged = self._json_call(CURATOR_ID, "curate", CURATOR_OVERVIEW, user, "curate:overview")
            seqs += logged
            out = {"summary": ov.get("summary", ""), "dictionary": dictionary,
                   "flags": list(ov.get("flags") or []) + flags,
                   "protocol_variables_missing_from_data": ov.get("protocol_variables_missing_from_data", [])}
            prompts = [prompt_ref("curator_columns"), prompt_ref("curator_overview")]
        report = {"profile": prof, "dictionary": out.get("dictionary", []), "flags": out.get("flags", []),
                  "protocol_variables_missing_from_data": out.get("protocol_variables_missing_from_data", []),
                  "summary": out.get("summary", ""), "prompt": prompts[0], "prompts": prompts,
                  "columns": len(cols), "earlier_call_seqs": seqs}
        return self.project.record_curator_report(CURATOR_ID, report, call,
                                                  requested_by=requested_by, material_sha256=ds["sha256"])

    def update_curator_report(self, requested_by: str) -> State:
        """Bring the Curator's report up to date in one call: each earlier finding marked open or resolved (and how),
        with the protocol, the authors' answers and the data preparation in view. Quick, even on wide datasets."""
        s = self.project.state()
        ds = self._latest(s, "dataset")
        prev_r = next((r for r in reversed(s.curator_reports) if r.get("material_sha256") in (ds["sha256"], ds.get("prepared_from"))), None)
        if prev_r is None:
            return self.curate(requested_by, ds["sha256"])
        prev = self.project.blobs.get_json(prev_r["report_sha256"])
        try:
            protocol, _ = self._protocol_text(s)
        except AgentError:
            protocol = "(no protocol uploaded yet)"
        prof = prev["profile"] if prev_r.get("material_sha256") == ds["sha256"] else profile(read_table(self._bytes(ds), ds["filename"]))
        cols = [{"name": c["name"], "missing": c.get("missing")} for c in prof["columns"]]
        prep = ds.get("preparation")
        user = (f"EARLIER FINDINGS:\n{json.dumps({'summary': prev.get('summary'), 'flags': prev.get('flags')}, indent=1)[:30000]}\n\n"
                f"PROTOCOL:\n{protocol[:CURATE_PROTOCOL_CHARS]}\n\nTHE AUTHORS' ANSWERS:\n{answers_text(s)}\n\n"
                f"DATA PREPARATION LOG:\n{json.dumps(prep['log'] if prep else None, indent=1)}\n\n"
                f"COLUMNS NOW (dataset in use: {ds['filename']}):\n{json.dumps(cols, separators=(',', ':'))[:30000]}")
        obj, call, _ = self._json_call(CURATOR_ID, "curate", CURATOR_UPDATE, user, "curator_update")
        flags = [f for f in obj.get("flags") or [] if isinstance(f, dict) and f.get("issue")]
        dictionary = prev.get("dictionary") or []
        if prep:
            names = {c["name"] for c in prof["columns"]}
            dictionary = [d for d in dictionary if d.get("column") in names] + [
                {"column": st["new"], "label": st.get("label") or st["new"], "type": "binary", "units_or_coding": "0=No, 1=Yes",
                 "role": "derived"} for st in prep["steps"] if st.get("new") in names and not any(d.get("column") == st["new"] for d in dictionary)]
        report = {"profile": prof, "dictionary": dictionary, "flags": flags,
                  "protocol_variables_missing_from_data": prev.get("protocol_variables_missing_from_data", []),
                  "summary": str(obj.get("summary") or prev.get("summary") or ""), "prompt": prompt_ref("curator_update"),
                  "prompts": [prompt_ref("curator_update")], "columns": len(prof["columns"]), "updated_from_seq": prev_r["seq"]}
        return self.project.record_curator_report(CURATOR_ID, report, call, requested_by=requested_by,
                                                  material_sha256=ds["sha256"])

    # -- Analysis plan -------------------------------------------------------
    def propose_plan(self, requested_by: str, instructions: str | None = None) -> State:
        s = self.project.state()
        protocol, pm = self._protocol_text(s)
        d = self._dictionary(s)
        user = (f"PROTOCOL:\n{protocol}\n\nDATA DICTIONARY:\n{json.dumps(d['dictionary'], separators=(',', ':'))}\n\n"
                f"DATASET PROFILE (aggregate only):\n{json.dumps(d['profile'], separators=(',', ':'))}"
                + f"\n\nTHE AUTHORS' ANSWERS TO SAKSI'S QUESTIONS:\n{answers_text(s)}"
                + (f"\n\nTHE AUTHORS' INSTRUCTIONS FOR THIS PLAN (follow them; the protocol still governs what "
                   f"each analysis is):\n{instructions.strip()}" if instructions and instructions.strip() else ""))
        try:
            out, call, _ = self._json_call(ANALYST_ID, "plan", PLAN, user, "propose_plan")
        except CutOff as e:
            raise AgentError("the plan was too long for one reply. Ask for a new draft with instructions, "
                             "for example one objective at a time") from e
        analyses = out.get("analyses") or []
        if not analyses:
            self.project.record_model_call(ANALYST_ID, call)
            raise AgentError("the model proposed no analyses")
        analyses, repaired = self._repair_quotes(analyses, protocol)
        note = (f"Quotations not found at first: {repaired['failed']}; corrected automatically and found: "
                f"{repaired['fixed']}.") if repaired else None
        cols = {c["name"] for c in d["profile"]["columns"]}
        for a in analyses:
            a["protocol_quote_verified"] = quote_found(a.get("protocol_quote", ""), protocol)
            named = [a.get("outcome") or {}] + list(a.get("exposures") or []) + list(a.get("covariates") or [])
            a["columns_not_in_data"] = sorted({v["column"] for v in named
                                               if isinstance(v, dict) and v.get("column") and v["column"] not in cols})
            a["prompt"] = prompt_ref("plan")
        if instructions and instructions.strip():
            asked = f"Redrafted at the request of {requested_by}: {instructions.strip()}"
            note = f"{asked}. {note}" if note else asked
        return self.project.propose_plan(ANALYST_ID, analyses, protocol_ref=pm["sha256"],
                                         call=call, requested_by=requested_by, note=note,
                                         table1=_table1_in_data(out.get("table1"), cols), report=out.get("report"))

    def prepare_data(self, requested_by: str, instructions: str | None = None) -> State:
        """Propose how to clean the uploaded dataset, from the Curator's findings. The authors tick the steps; Saksi's
        own code applies them (Project.prepare_dataset)."""
        s = self.project.state()
        ds = next((m for m in reversed(s.materials) if m["kind"] == "dataset" and not m.get("withdrawn")
                   and not m.get("prepared_from")), None)
        if ds is None:
            raise WorkflowError("upload the dataset first")
        rep = next((self.project.blobs.get_json(r["report_sha256"]) for r in reversed(s.curator_reports)
                    if r.get("material_sha256") == ds["sha256"]), None)
        if rep is None:
            s = self.curate(requested_by, ds["sha256"])
            rep = self.project.blobs.get_json(s.curator_reports[-1]["report_sha256"])
        try:
            protocol, _ = self._protocol_text(s)
        except Exception:
            protocol = ""
        user = (f"DATA CURATOR'S FINDINGS:\n{json.dumps({'summary': rep.get('summary'), 'flags': rep.get('flags')}, indent=1)[:30000]}\n\n"
                f"DATASET PROFILE (aggregate only):\n{json.dumps(rep['profile'], separators=(',', ':'))[:60000]}\n\n"
                f"DATA DICTIONARY:\n{json.dumps(rep.get('dictionary'), separators=(',', ':'))[:30000]}\n\n"
                f"PROTOCOL:\n{protocol[:40000]}\n\nTHE AUTHORS' ANSWERS TO SAKSI'S QUESTIONS:\n{answers_text(s)}"
                + (f"\n\nTHE AUTHORS' INSTRUCTIONS: {instructions}" if instructions else ""))
        out, call, _ = self._json_call(ANALYST_ID, "prep", DATA_PREP, user, "prepare_data")
        steps, problems = cleaning.validate(out.get("steps") or [], [c["name"] for c in rep["profile"]["columns"]])
        summary = str(out.get("summary") or "")[:1500]
        qs = [str(q)[:400] for q in out.get("questions") or []][:5]
        if qs:
            summary += "" if not summary else " "
            summary += "Questions for the authors: " + " ".join(qs)
        return self.project.propose_preparation(ANALYST_ID, ds["sha256"], steps, summary, problems,
                                                requested_by=requested_by, call=call, instructions=instructions)

    def outline(self, requested_by: str) -> State:
        """What this paper reports and how, for the authors, with a Methods draft: from the protocol, the chosen
        analyses, the authors' answers and the data preparation. Updated as these change."""
        s = self.project.state()
        if s.plan is None:
            raise WorkflowError("there is no plan yet")
        try:
            protocol, _ = self._protocol_text(s)
        except Exception:
            protocol = ""
        d = self._dictionary(s)
        plan = s.plan_locked or s.plan
        chosen = {k: {f: a.get(f) for f in ("title", "objective", "role", "kind", "outcome", "exposures", "covariates",
                                            "method", "effect_measure", "missing_data", "sample_size_quote", "sample_size_target")}
                  for k, a in plan["analyses"].items() if a.get("included", True) is not False}
        left = {k: a.get("exclusion_reason") for k, a in plan["analyses"].items() if a.get("included", True) is False}
        user = (f"PROTOCOL:\n{protocol[:60000]}\n\nTHE ANALYSES THIS PAPER REPORTS (chosen by the authors):\n"
                f"{json.dumps(chosen, indent=1)[:25000]}\n\nPROTOCOL ANALYSES LEFT FOR OTHER PAPERS: {json.dumps(left)}\n\n"
                f"TABLE 1: {json.dumps(plan.get('table1'))}\n\nTHE PLAN'S AIM AND DESIGN SO FAR: {json.dumps(plan.get('report'))}\n\n"
                f"THE AUTHORS' ANSWERS TO SAKSI'S QUESTIONS:\n{answers_text(s)}\n\n"
                f"DATA PREPARATION: {json.dumps(preparation_in_use(s))[:8000]}\n\n"
                f"SUPPORTING MATERIALS THE AUTHORS UPLOADED:\n{self._supporting_text(s)[:20000]}\n\n"
                f"DATA DICTIONARY:\n{json.dumps(d.get('dictionary'), separators=(',', ':'))[:20000]}\n\n"
                f"REPORTING GUIDELINE ITEMS FOR METHODS:\n"
                f"{json.dumps([i for i in guidelines.item_list((plan.get('report') or {}).get('design')) if i['section'] == 'methods'])}")
        obj, call, _ = self._json_call(WRITER_ID, "checklist", OUTLINE, user, "outline")
        methods = str(obj.pop("methods", "") or "").strip()
        clean = {"open_items": [str(q)[:400] for q in obj.get("open_items") or []][:8],
                 "aim": str(obj.get("aim") or "")[:800], "questions": [str(q)[:400] for q in obj.get("questions") or []][:6],
                 "design": obj.get("design") if obj.get("design") in standard.REPORT_DESIGNS else "other",
                 "design_note": str(obj.get("design_note") or "")[:400], "population": str(obj.get("population") or "")[:800],
                 "outcomes": [str(x)[:300] for x in obj.get("outcomes") or []][:8],
                 "exposures": [str(x)[:300] for x in obj.get("exposures") or []][:20],
                 "approach": str(obj.get("approach") or "")[:1500]}
        if not methods:
            self.project.record_model_call(WRITER_ID, call)
            raise AgentError("the Writing Agent returned no Methods")
        return self.project.record_outline(WRITER_ID, clean, methods, requested_by=requested_by, call=call)

    def suggest_answers(self, requested_by: str) -> State:
        """Saksi proposes an answer to each of its open questions, for the authors to accept or change."""
        s = self.project.state()
        answered = {a["question"] for a in s.answers}
        qs = [q for q in open_questions(s) if q not in answered and q not in s.suggested_answers]
        if not qs:
            return s
        try:
            protocol, _ = self._protocol_text(s)
        except Exception:
            protocol = ""
        d = self._dictionary(s)
        rep = next((self.project.blobs.get_json(r["report_sha256"]) for r in reversed(s.curator_reports)), {})
        plan = {k: {f: a.get(f) for f in ("title", "objective", "outcome", "exposures", "covariates", "method", "unresolved")}
                for k, a in (s.plan or {}).get("analyses", {}).items()}
        user = (f"QUESTIONS:\n{json.dumps(qs, indent=1)}\n\nPROTOCOL:\n{protocol[:50000]}\n\n"
                f"DATA DICTIONARY:\n{json.dumps(d.get('dictionary'), separators=(',', ':'))[:30000]}\n\n"
                f"DATASET PROFILE (aggregate only):\n{json.dumps(d['profile'], separators=(',', ':'))[:40000]}\n\n"
                f"DATA CURATOR'S FINDINGS:\n{json.dumps(rep.get('flags'), separators=(',', ':'))[:15000]}\n\n"
                f"PLAN:\n{json.dumps(plan)[:20000]}\n\nANSWERS SO FAR:\n{answers_text(s)}")
        obj, call, _ = self._json_call(ANALYST_ID, "prep", SUGGEST_ANSWERS, user, "suggest_answers")
        got = {str(x.get("question")): x for x in obj.get("suggestions") or [] if isinstance(x, dict)}
        out = [{"question": q, "answer": str(got[q].get("answer") or "")[:1500], "basis": str(got[q].get("basis") or "")[:300]}
               for q in qs if q in got and str(got[q].get("answer") or "").strip()]
        if not out:
            self.project.record_model_call(ANALYST_ID, call)
            raise AgentError("Saksi could not propose answers this time")
        return self.project.suggest_answers(ANALYST_ID, out, requested_by=requested_by, call=call)

    def plan_assist(self, requested_by: str, message: str, history: list[dict[str, Any]] | None = None):
        """Saksi and the authors settle what this paper reports, before any analysis. Answers, asks, and when an
        answer settles something, proposes instructions to redraft the plan (the page sends them on)."""
        s = self.project.state()
        if s.plan is None:
            raise WorkflowError("there is no plan yet; prepare one first")
        if not (message or "").strip():
            raise WorkflowError("write a message")
        protocol, _ = self._protocol_text(s)
        d = self._dictionary(s)
        plan = {"report": s.plan.get("report"), "table1": s.plan.get("table1"),
                "analyses": {k: {f: a.get(f) for f in ("title", "objective", "role", "kind", "included", "exclusion_reason",
                                                         "outcome", "exposures", "method", "unresolved", "label")}
                             for k, a in s.plan["analyses"].items()}}
        convo = "\n".join(f"{'AUTHOR' if m.get('who') == 'you' else 'SAKSI'}: {str(m.get('text', ''))[:800]}"
                           for m in (history or [])[-12:])
        user = (f"PROTOCOL:\n{protocol[:60000]}\n\nDATA DICTIONARY:\n"
                f"{json.dumps([{'column': c.get('column'), 'label': c.get('label')} for c in d.get('dictionary') or []])[:15000]}\n\n"
                f"CURRENT PLAN:\n{json.dumps(plan, indent=1)[:30000]}\n\nCONVERSATION SO FAR:\n{convo or '(none)'}\n\n"
                f"AUTHOR'S NEW MESSAGE: {message}")
        obj, call, _ = self._json_call(ANALYST_ID, "assist", PLAN_ASSIST, user, purpose="plan_assist")
        self.project.record_model_call(ANALYST_ID, call)
        return self.project.state(), {"reply": str(obj.get("reply") or "")[:1500],
                                      "questions": [str(q)[:400] for q in (obj.get("questions") or [])][:3],
                                      "redraft": str(obj.get("redraft") or "").strip()[:3000]}

    def checklist(self, requested_by: str, version: int | None = None) -> State:
        """Check a draft against the reporting guideline for the study design; quotations verified by code."""
        s = self.project.state()
        v = version or s.draft_version
        if not v or v not in s.drafts:
            raise WorkflowError("there is no draft to check")
        design = ((s.plan_locked or {}).get("report") or {}).get("design")
        g, items = guidelines.for_design(design), guidelines.item_list(design)
        text = self.project.blobs.get_text(s.drafts[v]["text_sha256"])
        user = (f"GUIDELINE: {g['name']}\n\nITEMS:\n{json.dumps(items, indent=1)}\n\nDRAFT:\n{text[:60000]}")
        obj, call, _ = self._json_call(WRITER_ID, "checklist", CHECKLIST, user, purpose=f"checklist:{v}")
        got = {str(x.get("id")): x for x in obj.get("items") or [] if isinstance(x, dict)}
        out = []
        for it in items:
            x = got.get(it["id"], {})
            status = x.get("status") if x.get("status") in ("reported", "partly", "missing", "not_applicable") else "missing"
            quote = str(x.get("quote") or "")[:300]
            found = bool(quote) and quote_found(quote, text)
            if status in ("reported", "partly") and not found:
                status, note = "partly" if quote else "missing", "the quotation given was not found in the draft"
            else:
                note = str(x.get("note") or "")[:300]
            out.append({**it, "status": status, "quote": quote if found else "", "note": note})
        return self.project.record_checklist(WRITER_ID, v, g["name"], out, requested_by=requested_by, call=call)

    QUOTE_FIELDS = (("protocol_quote", "the sentence specifying the analysis"),
                    ("primary_quote", "the sentence naming it the primary analysis"),
                    ("sample_size_quote", "the sample-size justification covering it"))

    def _repair_quotes(self, analyses: list[dict[str, Any]], protocol: str):
        """Quotations the check cannot find go back to the agent once, with the protocol's most similar
        passages. Every replacement is checked again by code; nothing unfound is accepted."""
        failed = []
        for a in analyses:
            for field, what in self.QUOTE_FIELDS:
                q = str(a.get(field) or "")
                if q.strip() and locate(q, protocol)["status"] == "not_found":
                    failed.append({"id": a.get("id"), "field": field, "what": what, "quote": q,
                                   "passages": candidates(q, protocol)})
        if not failed:
            return analyses, None
        user = "QUOTATIONS NOT FOUND, WITH THE MOST SIMILAR PROTOCOL PASSAGES:\n" + json.dumps(failed, indent=1)
        try:
            out, call, _ = self._json_call(ANALYST_ID, "quotes", QUOTE_REPAIR, user, "repair_quotes")
        except AgentError:
            return analyses, {"failed": len(failed), "fixed": 0}
        self.project.record_model_call(ANALYST_ID, call)
        fixes = {(f.get("id"), f.get("field")): str(f.get("quote") or "") for f in out.get("fixes") or []
                 if isinstance(f, dict)}
        by_id, fixed = {a.get("id"): a for a in analyses}, 0
        for item in failed:
            new = fixes.get((item["id"], item["field"]))
            if new is None:
                continue
            if not new.strip() or locate(new, protocol)["status"] != "not_found":
                by_id[item["id"]][item["field"]] = new
                fixed += bool(new.strip())
        return analyses, {"failed": len(failed), "fixed": fixed}

    def prepare_plan(self, requested_by: str, instructions: str | None = None) -> State:
        """One step for the authors: check the dataset in use (if not yet checked), then draft the plan."""
        s = self.project.state()
        ds = self._latest(s, "dataset")
        self._latest(s, "protocol")
        if not any(r.get("material_sha256") == ds["sha256"] for r in s.curator_reports):
            self.curate(requested_by)
        return self.propose_plan(requested_by, instructions)

    # -- Running analyses ------------------------------------------------------
    def run_planned(self, requested_by: str, plan_analysis_id: str | None = None,
                    standard_step: str | None = None) -> State:
        """The standard sequence: data quality, Table 1, then the plan's analyses, descriptive before
        univariable before multivariable. One step at a time if one is named (keeps web requests short)."""
        s = self.project.state()
        if s.plan_locked is None:
            raise WorkflowError("the analysis plan has not been confirmed by every author")
        if standard_step:
            if any(a.get("standard") == standard_step and a["status"] == "ok" for a in s.analyses.values()):
                return s                                     # already done; never a second Table 1
            return self._run_standard(requested_by, standard_step)
        if not plan_analysis_id:
            for step in pending_standard(s):
                s = self._run_standard(requested_by, step)
        ids = [plan_analysis_id] if plan_analysis_id else sequence(s.plan_locked)
        for pid in ids:
            if pid not in s.plan_locked["analyses"]:
                raise WorkflowError(f"{pid!r} is not in the confirmed plan")
            done = any(a["plan_analysis_id"] == pid and a["status"] == "ok" for a in s.analyses.values())
            if not done:
                # Saksi completes what the plan requires: a new attempt learns from the last failed one.
                failed = [a for a in s.analyses.values() if a["plan_analysis_id"] == pid and a["status"] != "ok"]
                context = None
                if failed:
                    last = failed[-1]
                    code = self.project.blobs.get_text(last["code_sha256"]) if self.project.blobs.has(last.get("code_sha256") or "") else ""
                    context = (f"AN EARLIER ATTEMPT AT THIS ANALYSIS FAILED. Its error: {(last.get('error') or '')[:1500]}\n"
                               f"Its code:\n```python\n{code[:10000]}\n```\nWrite code that avoids this failure; keep to "
                               "the specification. If the specification truly cannot be followed with these data, return "
                               "{\"error\": \"reason\"} saying exactly what is missing.")
                self._run_one(requested_by, s.plan_locked["analyses"][pid], plan_analysis_id=pid,
                              analysis_id=self._next_id(pid), context=context)
                s = self.project.state()
        return s

    def _run_standard(self, requested_by: str, step: str, table1: dict[str, Any] | None = None,
                      supersedes: str | None = None, reason: str | None = None) -> State:
        """A standard step, with Saksi's own code, in the same sandbox and with the same record."""
        s = self.project.state()
        if step not in standard.STEPS:
            raise WorkflowError(f"no standard step {step!r}")
        t1 = table1 or (s.plan_locked or {}).get("table1")
        if step == "table1" and not t1:
            raise WorkflowError("the confirmed plan has no Table 1 specification")
        ds = self._latest(s, "dataset")
        aid = f"s_{step}"
        code = standard.code(step, t1)
        seed = analysis_seed(s.project_id, aid)
        result = self.executor.run(code, self._bytes(ds), seed=seed)
        ok = result.status == "ok" and isinstance(result.results, dict) and "error" not in result.results
        error = None
        if ok and result.meta.get("data_sha256") != ds["sha256"]:
            ok, error = False, "the sandbox reports a different data hash from the uploaded dataset"
        if not ok and error is None:
            error = result.error or (json.dumps(result.results)[:500] if result.results else "step failed")
        execution = {"backend": getattr(self.executor, "backend", "unknown"), "seed": seed,
                     "dataset_sha256": ds["sha256"], "code_by": "saksi (standard code, no model)",
                     "attempts": [{"attempt": 1, "outcome": "ok" if ok else result.status,
                                   "execution": result.to_payload()}], "provenance": result.meta}
        spec = {"id": aid, "title": standard.STEPS[step], "standard": step} | ({"table1": t1} if step == "table1" else {})
        return self.project.complete_analysis(
            SYSTEM, self._next_id(aid), plan_analysis_id=None, spec=spec,
            code_sha256=self.project.blobs.put_text(code),
            outputs_sha256=self.project.blobs.put_json(result.results) if ok else None,
            execution=execution, status="ok" if ok else "error", error=error, requested_by=requested_by,
            standard=step, supersedes=supersedes if ok else None, redo_reason=reason if ok else None)

    def change_table1(self, requested_by: str, table1: dict[str, Any], reason: str = "") -> State:
        """Authors change Table 1's variables or grouping. Saksi's own code builds it again; no model.
        The earlier table stays in the record, marked as replaced, with the reason."""
        s = self.project.state()
        old = next((k for k, v in current(s).items() if v.get("standard") == "table1"), None)
        if s.plan_locked is None:
            raise WorkflowError("the analysis plan has not been confirmed by every author")
        cols = list((self._dictionary(s).get("profile") or {}).get("columns") or [])
        t1 = standard.clean_table1(table1)
        if t1 is None:
            raise WorkflowError("give at least one characteristic for Table 1")
        names = {c.get("name") for c in cols if isinstance(c, dict)} or None
        if names:
            unknown = [v for v in t1["variables"] + [t1["stratify_by"] or ""] if v and v not in names]
            if unknown:
                raise WorkflowError(f"not columns in the dataset: {', '.join(unknown)}")
        if old is not None and not (reason or "").strip():
            raise WorkflowError("say why Table 1 is changed (kept in the record)")
        # Building it for the first time (a plan made before Table 1 was part of the standard sequence) needs no
        # reason; the specification is recorded with the table, by the author who asked.
        return self._run_standard(requested_by, "table1", table1=t1, supersedes=old, reason=reason if old else None)

    def redo(self, requested_by: str, analysis_id: str, reason: str) -> State:
        """An author asks the Analysis Agent to redo an analysis, for example because its code mis-coded a
        variable. The new run replaces it and keeps its label, which was fixed before any result; the
        earlier run stays in the record, marked as replaced with the reason, and the reviewer sees both."""
        s = self.project.state()
        old = current(s).get(analysis_id)
        if old is None or old.get("standard"):
            raise WorkflowError(f"{analysis_id!r} is not a current analysis the Analysis Agent ran")
        if not (reason or "").strip():
            raise WorkflowError("say what is wrong and should be redone")
        pid = old.get("plan_analysis_id")
        spec = dict(s.plan_locked["analyses"][pid]) if pid else dict(old["spec"])
        spec["redo"] = {"of": analysis_id, "reason": reason}
        context = (f"THE AUTHORS ASKED FOR THIS ANALYSIS TO BE REDONE. Their reason: {reason}\n"
                   "Keep to the ANALYSIS SPECIFICATION; change only what the reason points to. Its earlier code:\n"
                   f"```python\n{self.project.blobs.get_text(old['code_sha256'])[:12000]}\n```")
        base = pid or old["spec"].get("id") or analysis_id
        return self._run_one(requested_by, spec, plan_analysis_id=pid, analysis_id=self._next_id(base),
                             request_id=old.get("request_id"), context=context,
                             supersedes=analysis_id, redo_reason=reason)

    def run_request(self, requested_by: str, request_id: str, prompt: str, based_on: str | None = None) -> State:
        """An exploratory analysis an author asks for: new, or a variant of an existing analysis (which stays)."""
        s = self.project.state()
        parent = current(s).get(based_on) if based_on else None
        if based_on and (parent is None or parent.get("standard")):
            raise WorkflowError(f"{based_on!r} is not a current analysis the Analysis Agent ran")
        self.project.request_analysis(requested_by, request_id, prompt)
        who = (s.actors.get(requested_by) or {}).get("name") or requested_by
        spec = {"id": f"x_{request_id}", "title": prompt, "author_request": prompt,
                "note": f"Exploratory: requested by {who}, not specified in the protocol."}
        context = None
        if parent:
            ptitle = parent["spec"].get("title") or based_on
            spec.update({"title": f"{ptitle}: {prompt}"[:160], "kind": parent.get("kind") or "other",
                         "based_on": {"analysis_id": based_on, "title": ptitle},
                         "note": f"Exploratory: a variant of {based_on}, asked for by {who} after seeing its "
                                 "results. The original stays and is reported too."})
            context = (f"THIS IS A VARIANT OF AN EARLIER ANALYSIS ({based_on}). Its specification:\n"
                       f"{json.dumps(parent['spec'], indent=1)[:6000]}\nIts code:\n```python\n"
                       f"{self.project.blobs.get_text(parent['code_sha256'])[:12000]}\n```\n"
                       f"Change it as the author asks: {prompt}")
        return self._run_one(requested_by, spec, plan_analysis_id=None, analysis_id=self._next_id(spec["id"]),
                             request_id=request_id, context=context)

    def _next_id(self, base: str) -> str:
        s = self.project.state()
        n = sum(1 for k in s.analyses if k == base or k.startswith(base + ".r"))
        return base if n == 0 else f"{base}.r{n}"

    def _run_one(self, requested_by: str, spec: dict[str, Any], *, plan_analysis_id: str | None,
                 analysis_id: str, request_id: str | None = None, context: str | None = None,
                 supersedes: str | None = None, redo_reason: str | None = None) -> State:
        s = self.project.state()
        ds = self._latest(s, "dataset")
        data = self._bytes(ds)
        d = self._dictionary(s)
        seed = analysis_seed(s.project_id, analysis_id)
        rf = required_figure(spec)                  # chosen by rule from the plan, not by the model
        if rf:
            spec = {**spec, "required_figure": rf}
        dic, prof, all_names = _focus(d, [spec])
        base_user = (f"ANALYSIS SPECIFICATION:\n{json.dumps(spec, indent=1)}\n\n"
                     f"DATA DICTIONARY:\n{json.dumps(dic, indent=1)}\n\n"
                     f"DATASET PROFILE (aggregate only{'; only the columns this analysis names' if all_names else ''}):\n"
                     f"{json.dumps(prof, indent=1)}"
                     + (f"\n\nALL COLUMN NAMES IN THE CSV:\n{json.dumps(all_names)}" if all_names else "")
                     + (f"\n\nTHE AUTHORS' ANSWERS TO SAKSI'S QUESTIONS (definitions to follow):\n{answers_text(s)}" if s.answers else "")
                     + (f"\n\n{context}" if context else ""))
        messages_user, task, attempts, failed_call_seqs = base_user, "code", [], []
        code, result, last_call = "", None, None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            system = CODE if task == "code" else CODE + "\n\n" + REPAIR
            r = self._chat(task, system, messages_user, json_mode=False, purpose=f"{task}:{analysis_id}")
            last_call = r.record.to_payload()
            code = parse_code(r.text)
            code_sha = self.project.blobs.put_text(code)
            try:
                check_code(code)
            except CodeRejected as e:
                result = None
                attempts.append({"attempt": attempt, "code_sha256": code_sha, "outcome": "rejected", "reason": str(e)})
                problem = f"The code was rejected before running: {e}"
            else:
                result = self.executor.run(code, data, seed=seed)
                ok = result.status == "ok" and isinstance(result.results, dict) and "error" not in result.results
                outcome = result.status
                if result.status == "ok" and not ok:
                    # Ran without crashing but did not deliver: say so, don't call it "ok".
                    outcome = "returned_error" if isinstance(result.results, dict) else "bad_result"
                if ok and standard.kind_of(spec) == "multivariable" and not isinstance(result.results.get("diagnostics"), dict):
                    ok, outcome = False, "missing_diagnostics"   # the standard sequence requires them
                attempts.append({"attempt": attempt, "code_sha256": code_sha, "outcome": outcome,
                                 "execution": result.to_payload()})
                if ok:
                    break
                if result.status == "infrastructure_error":
                    break                       # not the code's fault; don't spend a repair on it
                problem = (MISSING_DIAGNOSTICS if outcome == "missing_diagnostics" else
                           f"Running it failed:\n{result.error}" if result.error
                           else f"analyse() returned: {json.dumps(result.results)[:1500]}")
            if attempt < MAX_ATTEMPTS:
                failed_call_seqs.append(self.project.record_model_call(ANALYST_ID, last_call))
                task = "repair"
                messages_user = f"{base_user}\n\nPREVIOUS CODE:\n```python\n{code}```\n\nPROBLEM:\n{problem}"

        final = attempts[-1]
        # Results without diagnostics after one repair are kept, but flagged in the record.
        diagnostics_missing = final["outcome"] == "missing_diagnostics"
        ok = final["outcome"] in ("ok", "missing_diagnostics") and result is not None \
            and "error" not in (result.results or {})
        error = None
        if ok and result.meta.get("data_sha256") != ds["sha256"]:
            ok, error = False, "the sandbox reports a different data hash from the uploaded dataset"
        if not ok and error is None:
            error = final.get("reason") or (result.error if result else None) or \
                (json.dumps(result.results)[:500] if result and result.results else "analysis failed")
        figs = [{"name": f["name"], "caption": f.get("caption", ""), "svg_sha256": self.project.blobs.put_text(f["svg"])}
                for f in (result.figures if ok and result else [])]
        execution = {"backend": getattr(self.executor, "backend", "unknown"), "seed": seed,
                     "dataset_sha256": ds["sha256"], "attempts": attempts,
                     "provenance": result.meta if result else {}, "prompt": prompt_ref("code"),
                     "required_figure": rf["name"] if rf else None, "diagnostics_missing": diagnostics_missing,
                     "required_figure_drawn": bool(rf) and any(f["name"] == rf["name"] for f in figs),
                     "figure_error": result.figure_error if result else None}
        return self.project.complete_analysis(
            ANALYST_ID, analysis_id, plan_analysis_id=plan_analysis_id, spec=spec,
            code_sha256=self.project.blobs.put_text(code),
            outputs_sha256=self.project.blobs.put_json(result.results) if ok else None,
            execution=execution, status="ok" if ok else "error", error=error, request_id=request_id,
            call=last_call, requested_by=requested_by, call_seqs=failed_call_seqs, figures=figs,
            supersedes=supersedes if ok else None, redo_reason=redo_reason if ok else None)

    # -- The assistant on the Analyses page ----------------------------------
    ACTIONS = {"variant", "new_analysis", "redo", "rewrite_text", "table1", "merge_into_table1", "leave_out",
               "remove_figure", "none"}

    def assist(self, requested_by: str, message: str, analysis_id: str | None = None):
        """An author writes to the assistant about one result or the whole Results. The assistant answers and
        proposes at most one action; the page carries it out through the usual, recorded routes."""
        s = self.project.state()
        if not (message or "").strip():
            raise WorkflowError("write a message")
        if s.plan_locked is None:
            raise WorkflowError("the analysis plan has not been confirmed by every author")
        tables = tables_mod.build(s, self.project.blobs)
        cur = current(s)
        brief = {}
        for k, v in sorted(cur.items(), key=lambda kv: _position(kv[1])):
            out = self.project.blobs.get_json(v["outputs_sha256"])
            t = next((x for x in tables if x["analysis_id"] == k), None)
            d = out.get("diagnostics") if isinstance(out.get("diagnostics"), dict) else {}
            text = ""
            if v.get("description") and self.project.blobs.has(v["description"]["text_sha256"]):
                text = tables_mod.resolve(self.project.blobs.get_text(v["description"]["text_sha256"]), k, tables, s.figures)
            brief[k] = {"title": v["spec"].get("title"), "label": v["label"], "basis": v.get("basis"),
                        "kind": v.get("kind"), "standard": v.get("standard"), "table": t["number"] if t else None,
                        "figures": [f["number"] for f in s.figures if f["analysis_id"] == k and f.get("number")],
                        "n_analysed": out.get("n_analysed"), "estimates": (out.get("estimates") or [])[:8],
                        "diagnostic_flags": d.get("flags"), "text": text[:1500]}
        dic = [{"column": c.get("column"), "label": c.get("label")} for c in self._dictionary(s).get("dictionary") or []][:300]
        user = (f"AUTHOR'S MESSAGE: {message}\n\nABOUT: {analysis_id or 'the Results as a whole'}\n\n"
                f"RESULTS SO FAR:\n{json.dumps(brief, indent=1)[:40000]}\n\nDATA DICTIONARY:\n{json.dumps(dic)}")
        obj, call, _ = self._json_call(ANALYST_ID, "assist", ASSIST, user, purpose="assist")
        self.project.record_model_call(ANALYST_ID, call)
        reply = str(obj.get("reply") or "").strip()[:1200]
        act = obj.get("action") if isinstance(obj.get("action"), dict) else {}
        kind = act.get("type") if act.get("type") in self.ACTIONS else "none"
        target = act.get("analysis_id") or analysis_id
        fig = None
        if kind == "remove_figure":
            live = [f for f in s.figures if f.get("number") and not f.get("omitted")]
            want = re.sub(r"\D", "", str(act.get("figure") or ""))
            fig = next((f for f in live if str(f["number"]) == want), None)
            if fig is None:
                own = [f for f in live if f["analysis_id"] == target]
                fig = own[0] if len(own) == 1 else None
            if fig is None:
                kind, reply = "none", reply + " (Tell me which figure, by its number.)"
            else:
                target = fig["analysis_id"]
        merge = None
        if kind == "merge_into_table1":
            merge = self._merge_into_table1(s, cur, tables, target)
            if merge is None:
                kind, reply = "none", reply + " (I could not find a table describing the participants to fold into Table 1.)"
            else:
                target = merge["analysis_id"]
        if kind in {"variant", "redo", "rewrite_text", "leave_out"}:
            if target not in cur or cur[target].get("standard") and kind != "rewrite_text":
                kind, reply = "none", reply + " (I could not find a result I can change that way.)"
            elif kind == "leave_out" and cur[target].get("basis") != "post hoc":
                kind, reply = "none", reply + " (A planned analysis is reported; only an exploratory one added after the plan can be left out.)"
        proposal = {"reply": reply or "Done.", "action": {"type": kind, "analysis_id": target if kind != "none" else None,
                                                          "instruction": str(act.get("instruction") or message)[:2000]}}
        if merge is not None and kind == "merge_into_table1":
            proposal["action"].update(merge)
        if fig is not None and kind == "remove_figure":
            proposal["action"].update({"figure_name": fig["name"], "figure_number": fig["number"]})
        if kind == "table1":
            proposal["action"].update({"stratify_by": act.get("stratify_by"), "variables": act.get("variables") or []})
        return self.project.state(), proposal

    def _merge_into_table1(self, s: State, cur: dict[str, Any], tables: list[dict[str, Any]], target: str | None):
        """What folding a characteristics table into Table 1 means: Table 1's grouping, its characteristics plus those
        the other table adds (columns that exist in the data), and whether the other table can be left out."""
        known = {d.get("column") for d in self._dictionary(s).get("dictionary") or []}
        return tables_mod.merge_plan(s, self.project.blobs, target, known)

    def _supporting_text(self, s: State) -> str:
        parts, total = [], 0
        for m in s.materials:
            if m["kind"] in ("dataset", "protocol") or m.get("withdrawn") or not m.get("storage_ref"):
                continue
            try:
                text = extract_text(material_bytes(self.project, m), m["filename"])[:8000]
            except Exception:
                continue
            if total + len(text) > 30000:
                break
            total += len(text)
            parts.append(f"[{m['kind'].replace('_', ' ')}: {m['filename']}]\n{text}")
        return "\n\n".join(parts) or "(none)"

    # -- Text for each analysis -------------------------------------------------
    def describe(self, requested_by: str, analysis_id: str, instruction: str | None = None) -> State:
        """The Writing Agent writes the text that reports one analysis: Results text that points to its table
        and figures and says what matters without repeating them, or, for the data checks, a Methods passage.
        Every number is checked against that analysis; one rewrite if the check fails."""
        s = self.project.state()
        a = current(s).get(analysis_id)
        if a is None:
            raise WorkflowError(f"{analysis_id!r} is not a current analysis")
        out = self.project.blobs.get_json(a["outputs_sha256"])
        if a.get("sample_check"):
            out = {**out, "sample_check": a["sample_check"]}
        section = "methods" if a.get("standard") == "data_quality" else "results"
        tbl = next((t for t in tables_mod.build(s, self.project.blobs) if t["analysis_id"] == analysis_id), None)
        figs = [{"token": f"[[Figure:{f['name']}]]", "caption": f["caption"]}
                for f in s.figures if f["analysis_id"] == analysis_id and f.get("number")]
        shown = _for_writer(a, out, current(s)) if a.get("standard") == "data_quality" else out
        info = {"id": analysis_id, "title": a["spec"].get("title"), "label": a["label"], "basis": a.get("basis"),
                "kind": a.get("kind"), "standard": a.get("standard"), "sample_check": a.get("sample_check"),
                "protocol_departure": a.get("protocol_deviation"), "downgraded": a.get("downgraded")}
        if a.get("standard") == "data_quality":
            info["data_preparation"] = preparation_in_use(s)
        user = (f"ANALYSIS:\n{json.dumps(info, indent=1)}\n\nRESULTS:\n{json.dumps(shown, indent=1)[:30000]}\n\n"
                f"TABLE:\n{json.dumps({k: tbl[k] for k in ('title', 'columns', 'rows', 'footnotes')} if tbl else None)[:20000]}\n\n"
                f"FIGURES:\n{json.dumps(figs, indent=1)}\n\n"
                f"INTERVAL FACTS (computed by code):\n{json.dumps(interval_facts({analysis_id: out}), indent=1)}"
                + (f"\n\nAUTHOR'S INSTRUCTION: {instruction}" if instruction else ""))
        counts = []
        if s.curator_reports:
            rows = self.project.blobs.get_json(s.curator_reports[-1]["report_sha256"]).get("profile", {}).get("rows")
            counts = [rows] if rows else []
        attempts, messages, text, call = [], user, "", None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            r = self._chat("describe", RESULT_TEXT, messages, json_mode=False, purpose=f"describe:{analysis_id}")
            call, text = r.record.to_payload(), r.text.strip()
            text = re.sub(r"^#+\s.*\n", "", text).strip()
            m = re.match(r"^\s*\**heading\**\s*:\s*(.+?)\s*(?:\n|$)", text, re.I)
            heading = m.group(1).strip(" *#")[:120] if m else ""
            text = text[m.end():].strip() if m else text
            check = trace(text, {analysis_id: {"label": a["label"]}}, {analysis_id: out}, counts,
                          required_refs=[analysis_id])
            attempts.append({"attempt": attempt, "passed_check": check["ok"],
                             "findings_sha256": self.project.blobs.put_json(check)})
            if check["ok"] or attempt == MAX_ATTEMPTS:
                break
            attempts[-1]["llm_call_seq"] = self.project.record_model_call(WRITER_ID, call)
            messages = f"{user}\n\nYOUR PREVIOUS TEXT:\n{text}\n\nCHECK FINDINGS:\n{_brief(check)}"
        if not text:
            self.project.record_model_call(WRITER_ID, call)
            raise AgentError("the Writing Agent returned no text")
        return self.project.describe_analysis(WRITER_ID, analysis_id, text, section, summarise(check),
                                              requested_by=requested_by, call=call, instruction=instruction,
                                              attempts=attempts, heading=heading)

    # -- Writing -------------------------------------------------------------
    def write_draft(self, requested_by: str) -> State:
        """First draft (or an author-requested redraft) from computed results only."""
        s = self.project.state()
        protocol, _ = self._protocol_text(s)
        d = self._dictionary(s)
        ok = current(s)
        failed = {k: v for k, v in s.analyses.items() if v["status"] != "ok"}
        if not ok:
            raise AgentError("there are no successful analyses to write up")
        results = {k: {"label": v["label"], "basis": v.get("basis"), "kind": v.get("kind"),
                       "plan_analysis_id": v["plan_analysis_id"], "spec": v["spec"],
                       "sample_check": v.get("sample_check"), "protocol_departure": v.get("protocol_deviation"),
                       "downgraded": v.get("downgraded"),
                       "results": _for_writer(v, self.project.blobs.get_json(v["outputs_sha256"]), ok)}
                   for k, v in sorted(ok.items(), key=lambda kv: _position(kv[1]))}
        failed_brief = {k: {"label": v["label"], "plan_analysis_id": v["plan_analysis_id"],
                            "title": v["spec"].get("title"), "error": (v.get("error") or "")[:300]}
                        for k, v in failed.items()
                        # a failure later superseded by a successful rerun is not a failed analysis
                        if not any(o["plan_analysis_id"] == v["plan_analysis_id"] and v["plan_analysis_id"]
                                   for o in ok.values())}
        rows = d["profile"].get("rows")
        dic, _, _ = _focus(d, [v["spec"] for v in ok.values()])
        user = (f"PROTOCOL (for design and objectives only):\n{protocol[:40000]}\n\n"
                f"DATASET ROWS: {rows}\n\nDATA DICTIONARY:\n{json.dumps(dic, indent=1)}\n\n"
                f"RESULTS (cite as [A:<key>]):\n{json.dumps(results, indent=1)}\n\n"
                f"INTERVAL FACTS (computed by code):\n"
                f"{json.dumps(interval_facts({k: v['results'] for k, v in results.items()}), indent=1)}\n\n"
                f"METHODS DRAFTED WITH THE AUTHORS BEFORE THE ANALYSIS (use it as the basis of Methods; keep what the "
                f"authors wrote):\n{self.project.blobs.get_text(s.outline['methods_sha256']) if s.outline else '(none)'}\n\n"
                f"SOFTWARE (from the sandbox's recorded provenance): {json.dumps(_software(s))}\n\n"
                f"SUPPORTING MATERIALS THE AUTHORS UPLOADED (use them for Methods details such as ethics approval, "
                f"registration, setting and definitions; never for results):\n{self._supporting_text(s)}\n\n"
                f"DATA PREPARATION (applied by Saksi's code before analysis; describe it in Methods):\n"
                f"{json.dumps(preparation_in_use(s), indent=1)}\n\n"
                f"THE AUTHORS' ANSWERS TO SAKSI'S QUESTIONS:\n{answers_text(s)}\n\n"
                f"REPORT (what this paper answers, confirmed with the plan):\n"
                f"{json.dumps((s.plan_locked or {}).get('report'), indent=1)}\n\n"
                f"REPORTING GUIDELINE: {guidelines.for_design(((s.plan_locked or {}).get('report') or {}).get('design'))['name']}\n"
                f"{json.dumps(guidelines.item_list(((s.plan_locked or {}).get('report') or {}).get('design')), indent=1)}\n\n"
                f"PLANNED ANALYSES THAT FAILED:\n{json.dumps(failed_brief, indent=1)}\n\n"
                f"PROTOCOL ANALYSES NOT PERFORMED IN THIS PAPER (authors' decision, with reasons):\n"
                f"{json.dumps(not_performed(s), indent=1)}\n\n"
                f"TEXT ALREADY WRITTEN FOR EACH ANALYSIS (number-checked; keep its numbers and citations):\n"
                f"{_texts(s, ok, self.project.blobs)}\n\n"
                f"FIGURES (refer to them only by these numbers):\n{_figure_list(s, ok)}\n\n"
                f"TABLES (built by Saksi from RESULTS; refer to them only by these numbers):\n"
                f"{tables_mod.listing(s, self.project.blobs)}")
        if s.draft_version and s.drafts.get(s.draft_version):
            d = s.drafts[s.draft_version]
            cur_text = self.project.blobs.get_text(d["text_sha256"]) if d.get("text_sha256") and self.project.blobs.has(d["text_sha256"]) else ""
            if cur_text:
                user += (f"\n\nTHE CURRENT DRAFT (version {s.draft_version}; the results have changed since it was written):\n"
                         f"{cur_text[:60000]}")
        # Write, check, and give the agent one chance to fix what the check found.
        # Whatever still fails goes to the authors with the findings attached.
        attempts, text, last_call = [], "", None
        messages_user, system = user, WRITER
        for attempt in range(1, MAX_ATTEMPTS + 1):
            r = self._chat("write", system, messages_user, json_mode=False, purpose=f"write_draft:{attempt}")
            last_call, text = r.record.to_payload(), r.text.strip()
            if "## Results" not in text:
                findings = {"structural": "the draft has no '## Results' section"}
                passed = False
            else:
                findings = self.project.check_text(text)
                passed = findings["ok"]
            attempts.append({"attempt": attempt, "text_sha256": self.project.blobs.put_text(text),
                             "passed_check": passed, "findings_sha256": self.project.blobs.put_json(findings),
                             "prompt": prompt_ref("writer" if attempt == 1 else "rewrite")})
            if passed or attempt == MAX_ATTEMPTS:
                break
            attempts[-1]["llm_call_seq"] = self.project.record_model_call(WRITER_ID, last_call)
            system = WRITER + "\n\n" + REWRITE
            messages_user = f"{user}\n\nYOUR PREVIOUS DRAFT:\n{text}\n\nCHECK FINDINGS:\n{_brief(findings)}"
        if "## Results" not in text:
            self.project.record_model_call(WRITER_ID, last_call)
            raise AgentError("the draft has no Results section after a rewrite")
        sections = [ln[3:].strip() for ln in text.splitlines() if ln.startswith("## ")]
        return self.project.create_draft(WRITER_ID, text, sections, sorted(ok), last_call,
                                         requested_by=requested_by, attempts=attempts)

    # -- Review ----------------------------------------------------------------
    def brief_reviewer(self, requested_by: str) -> State:
        """Reviewer Assistant briefing. A different model; it gets no executor and cannot edit."""
        from .reviewer import ReviewerAssistant
        return ReviewerAssistant(self.project, self.client).brief(requested_by)


MISSING_DIAGNOSTICS = (
    "This is a multivariable model, so the standard sequence requires its diagnostics. After fitting the model, "
    "add: from saksi_checks import logistic_checks, linear_checks, and return "
    "\"diagnostics\": logistic_checks(model) for a binary outcome (linear_checks(model) for a continuous one), "
    "passing clustered_by=\"<column>\" if observations are clustered. Keep everything else the same.")


def _table1_in_data(raw: Any, cols: set[str]) -> dict[str, Any] | None:
    """The agent's Table 1 proposal, keeping only columns that exist."""
    t1 = standard.clean_table1(raw)
    if not t1:
        return None
    vs = [v for v in t1["variables"] if v in cols]
    by = t1["stratify_by"] if t1["stratify_by"] in cols else None
    return {**t1, "variables": vs, "stratify_by": by} if vs else None


_position = standard.position


def _for_writer(a: dict[str, Any], out: dict[str, Any], ok: dict[str, Any]) -> dict[str, Any]:
    """The data-quality step reports every column; the writer needs only the analysed ones."""
    if a.get("standard") != "data_quality":
        return out
    used = set()
    for v in ok.values():
        sp = v.get("spec") or {}
        for x in [sp.get("outcome") or {}] + list(sp.get("exposures") or []) + list(sp.get("covariates") or []):
            if isinstance(x, dict) and x.get("column"):
                used.add(x["column"])
        used.update((sp.get("table1") or {}).get("variables") or [])
    return {**out, "columns": [c for c in out.get("columns", []) if c["column"] in used] if used else out.get("columns", [])}


def sequence(plan: dict[str, Any]) -> list[str]:
    """Included analyses in the standard order: descriptive, univariable, multivariable, other;
    protocol order within each."""
    ids = included_order(plan)
    return sorted(ids, key=lambda i: (standard.KIND_ORDER.get(plan["analyses"][i].get("kind"), 3), ids.index(i)))


def pending_standard(s: State) -> list[str]:
    steps = ["data_quality"] + (["table1"] if (s.plan_locked or {}).get("table1") else [])
    done = {a.get("standard") for a in s.analyses.values() if a.get("standard") and a["status"] == "ok"}
    return [x for x in steps if x not in done]


def not_performed(s: State) -> list[dict[str, Any]]:
    plan = s.plan_locked or {}
    return [{"id": i, "title": a.get("title"), "reason": a.get("exclusion_reason")}
            for i, a in (plan.get("analyses") or {}).items() if not a.get("included", True)]


def _focus(d: dict[str, Any], specs: list[dict[str, Any]]):
    """On a wide dataset, only the columns these analyses name (plus every column name).
    Returns (dictionary, profile, all column names or None when nothing was narrowed)."""
    cols = d["profile"]["columns"]
    named = set()
    for spec in specs:
        for v in [spec.get("outcome") or {}] + list(spec.get("exposures") or []) + list(spec.get("covariates") or []):
            if isinstance(v, dict) and v.get("column"):
                named.add(v["column"])
    if len(cols) <= FOCUS_ABOVE or not named:
        return d["dictionary"], d["profile"], None
    return ([x for x in (d["dictionary"] or []) if x.get("column") in named],
            {**d["profile"], "columns": [c for c in cols if c["name"] in named]}, [c["name"] for c in cols])


def _figure_list(s: State, ok: dict[str, Any]) -> str:
    rows = [f"Figure {f['number']} [A:{f['analysis_id']}]: {f['caption']}"
            for f in sorted(s.figures, key=lambda f: f["number"] or 0) if f["analysis_id"] in ok and f["number"]]
    return "\n".join(rows) or "(none)"


def _software(s: State) -> dict[str, str]:
    for a in s.analyses.values():
        pk = ((a.get("execution") or {}).get("provenance") or {}).get("packages")
        if pk:
            return {"python": ((a.get("execution") or {}).get("provenance") or {}).get("python") or "3", **pk}
    return {}


def _texts(s: State, ok: dict[str, Any], blobs) -> str:
    tables = tables_mod.build(s, blobs)
    rows = []
    for k, v in sorted(ok.items(), key=lambda kv: _position(kv[1])):
        d = v.get("description")
        if d and blobs.has(d["text_sha256"]):
            text = tables_mod.resolve(blobs.get_text(d["text_sha256"]), k, tables, s.figures)
            rows.append(f"[{d['section'].upper()}] {k}: {text}")
    return "\n\n".join(rows) or "(none)"


def _brief(findings: dict[str, Any]) -> str:
    """Findings in a compact form the writer can act on."""
    if "structural" in findings:
        return findings["structural"]
    lines = []
    for u in findings.get("untraced", [])[:20]:
        lines.append(f"- UNTRACED {u['number']} in: {u['sentence'][:240]}")
    for x in findings.get("exploratory_not_labelled", [])[:10]:
        lines.append(f"- EXPLORATORY NOT LABELLED ({', '.join(x['analyses'])}): {x['sentence'][:240]}")
    for ref in findings.get("unknown_analysis_refs", []):
        lines.append(f"- UNKNOWN ANALYSIS TAG [A:{ref}]: use only keys from RESULTS")
    for ref in findings.get("analyses_not_reported", []):
        lines.append(f"- ANALYSIS NOT REPORTED: [A:{ref}] must be reported")
    for n in findings.get("tables_not_referred", []):
        lines.append(f"- TABLE {n} NOT REFERRED TO: give it a sentence or paragraph in Results that points to it and "
                     "states its main finding in words, without repeating the table")
    for n in findings.get("figures_not_referred", []):
        lines.append(f"- FIGURE {n} NOT REFERRED TO: refer to it where its result is reported")
    for x in findings.get("references_unknown", []):
        lines.append(f"- {x.upper()} DOES NOT EXIST: use only the numbers in TABLES and FIGURES")
    for x in findings.get("references_out_of_order", []):
        lines.append(f"- {x.upper()} REFERRED TO BEFORE A LOWER-NUMBERED ONE: in Results, refer to tables and figures "
                     "first in number order (reorder the paragraphs or the sentences)")
    for x in findings.get("referred_before_results", []):
        lines.append(f"- {x.upper()} CITED BEFORE THE RESULTS: Methods and earlier sections do not cite tables or "
                     "figures by number; refer to it first in Results")
    for ref in findings.get("departure_not_disclosed", []):
        lines.append(f"- DEPARTURE FROM THE PROTOCOL NOT DISCLOSED for [A:{ref}]: in a sentence citing it, say that "
                     "its primary role differs from the protocol, and give the authors' reason from protocol_departure")
    for ref in findings.get("shortfall_not_reported", []):
        lines.append(f"- SAMPLE SHORTFALL NOT STATED for [A:{ref}]: state the planned and achieved numbers "
                     "from its sample_check, citing it")
    if findings.get("nothing_checked_but_numbers_present"):
        lines.append("- Numbers present but none could be checked: tag result sentences with [A:<key>]")
    return "\n".join(lines) or "- (no specific findings)"
