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

from .documents import extract_text, quote_found
from .llm import LLMError, ModelClient
from .profiling import profile, read_table
from .prompts import CODE, CURATOR, CURATOR_COLUMNS, CURATOR_OVERVIEW, PLAN, REPAIR, REWRITE, WRITER, prompt_ref
from .sandbox import CodeRejected, Executor, check_code
from .canonical import sha256_text
from .figures import required_figure
from .tracing import interval_facts
from .workflow import Project, State, WorkflowError, included_order

CURATOR_ID, ANALYST_ID, WRITER_ID = "agent:curator", "agent:analyst", "agent:writer"

# Reasoning effort and output budget per task. Reasoning tokens are billed, so
# effort is spent only where judgement matters.
TASKS = {
    "curate": {"reasoning_effort": "low", "max_tokens": 12000},
    "plan": {"reasoning_effort": "high", "max_tokens": 32000},
    "code": {"reasoning_effort": "medium", "max_tokens": 12000},
    "repair": {"reasoning_effort": "medium", "max_tokens": 12000},
    "write": {"reasoning_effort": "medium", "max_tokens": 16000},
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
    ms = [m for m in s.materials if m["kind"] == kind and (sha is None or m["sha256"] == sha)]
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

    # -- Analysis plan -------------------------------------------------------
    def propose_plan(self, requested_by: str, instructions: str | None = None) -> State:
        s = self.project.state()
        protocol, pm = self._protocol_text(s)
        d = self._dictionary(s)
        user = (f"PROTOCOL:\n{protocol}\n\nDATA DICTIONARY:\n{json.dumps(d['dictionary'], separators=(',', ':'))}\n\n"
                f"DATASET PROFILE (aggregate only):\n{json.dumps(d['profile'], separators=(',', ':'))}"
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
        cols = {c["name"] for c in d["profile"]["columns"]}
        for a in analyses:
            a["protocol_quote_verified"] = quote_found(a.get("protocol_quote", ""), protocol)
            named = [a.get("outcome") or {}] + list(a.get("exposures") or []) + list(a.get("covariates") or [])
            a["columns_not_in_data"] = sorted({v["column"] for v in named
                                               if isinstance(v, dict) and v.get("column") and v["column"] not in cols})
            a["prompt"] = prompt_ref("plan")
        note = f"Redrafted at the request of {requested_by}: {instructions.strip()}" if instructions and \
            instructions.strip() else None
        return self.project.propose_plan(ANALYST_ID, analyses, protocol_ref=pm["sha256"],
                                         call=call, requested_by=requested_by, note=note)

    # -- Running analyses ------------------------------------------------------
    def run_planned(self, requested_by: str, plan_analysis_id: str | None = None) -> State:
        s = self.project.state()
        if s.plan_locked is None:
            raise WorkflowError("the analysis plan has not been confirmed by every author")
        ids = [plan_analysis_id] if plan_analysis_id else included_order(s.plan_locked)
        for pid in ids:
            if pid not in s.plan_locked["analyses"]:
                raise WorkflowError(f"{pid!r} is not in the confirmed plan")
            done = any(a["plan_analysis_id"] == pid and a["status"] == "ok" for a in s.analyses.values())
            if not done:
                self._run_one(requested_by, s.plan_locked["analyses"][pid], plan_analysis_id=pid,
                              analysis_id=self._next_id(pid))
                s = self.project.state()
        return s

    def run_request(self, requested_by: str, request_id: str, prompt: str) -> State:
        self.project.request_analysis(requested_by, request_id, prompt)
        spec = {"id": f"x_{request_id}", "title": prompt, "author_request": prompt,
                "note": "Exploratory: requested by an author, not specified in the protocol."}
        return self._run_one(requested_by, spec, plan_analysis_id=None, analysis_id=self._next_id(spec["id"]),
                             request_id=request_id)

    def _next_id(self, base: str) -> str:
        s = self.project.state()
        n = sum(1 for k in s.analyses if k == base or k.startswith(base + ".r"))
        return base if n == 0 else f"{base}.r{n}"

    def _run_one(self, requested_by: str, spec: dict[str, Any], *, plan_analysis_id: str | None,
                 analysis_id: str, request_id: str | None = None) -> State:
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
                     + (f"\n\nALL COLUMN NAMES IN THE CSV:\n{json.dumps(all_names)}" if all_names else ""))
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
                attempts.append({"attempt": attempt, "code_sha256": code_sha, "outcome": outcome,
                                 "execution": result.to_payload()})
                if ok:
                    break
                if result.status == "infrastructure_error":
                    break                       # not the code's fault; don't spend a repair on it
                problem = (f"Running it failed:\n{result.error}" if result.error
                           else f"analyse() returned: {json.dumps(result.results)[:1500]}")
            if attempt < MAX_ATTEMPTS:
                failed_call_seqs.append(self.project.record_model_call(ANALYST_ID, last_call))
                task = "repair"
                messages_user = f"{base_user}\n\nPREVIOUS CODE:\n```python\n{code}```\n\nPROBLEM:\n{problem}"

        final = attempts[-1]
        ok = final["outcome"] == "ok" and result is not None and "error" not in (result.results or {})
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
                     "required_figure": rf["name"] if rf else None,
                     "required_figure_drawn": bool(rf) and any(f["name"] == rf["name"] for f in figs),
                     "figure_error": result.figure_error if result else None}
        return self.project.complete_analysis(
            ANALYST_ID, analysis_id, plan_analysis_id=plan_analysis_id, spec=spec,
            code_sha256=self.project.blobs.put_text(code),
            outputs_sha256=self.project.blobs.put_json(result.results) if ok else None,
            execution=execution, status="ok" if ok else "error", error=error, request_id=request_id,
            call=last_call, requested_by=requested_by, call_seqs=failed_call_seqs, figures=figs)

    # -- Writing -------------------------------------------------------------
    def write_draft(self, requested_by: str) -> State:
        """First draft (or an author-requested redraft) from computed results only."""
        s = self.project.state()
        protocol, _ = self._protocol_text(s)
        d = self._dictionary(s)
        ok = {k: v for k, v in s.analyses.items() if v["status"] == "ok"}
        failed = {k: v for k, v in s.analyses.items() if v["status"] != "ok"}
        if not ok:
            raise AgentError("there are no successful analyses to write up")
        results = {k: {"label": v["label"], "basis": v.get("basis"), "plan_analysis_id": v["plan_analysis_id"],
                       "spec": v["spec"], "sample_check": v.get("sample_check"),
                       "results": self.project.blobs.get_json(v["outputs_sha256"])} for k, v in ok.items()}
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
                f"PLANNED ANALYSES THAT FAILED:\n{json.dumps(failed_brief, indent=1)}\n\n"
                f"PROTOCOL ANALYSES NOT PERFORMED IN THIS PAPER (authors' decision, with reasons):\n"
                f"{json.dumps(not_performed(s), indent=1)}\n\n"
                f"FIGURES (refer to them only by these numbers):\n{_figure_list(s, ok)}")
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
    rows = [f"Figure {f['number']} [A:{f['analysis_id']}]: {f['caption']}" for f in s.figures if f["analysis_id"] in ok]
    return "\n".join(rows) or "(none)"


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
    for ref in findings.get("shortfall_not_reported", []):
        lines.append(f"- SAMPLE SHORTFALL NOT STATED for [A:{ref}]: state the planned and achieved numbers "
                     "from its sample_check, citing it")
    if findings.get("nothing_checked_but_numbers_present"):
        lines.append("- Numbers present but none could be checked: tag result sentences with [A:<key>]")
    return "\n".join(lines) or "- (no specific findings)"
