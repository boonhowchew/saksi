# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Agents that act on a project: Data Curator and Analysis Agent.

Every agent action is requested by a named author and is written to the
event log by the workflow engine, with the model call that produced it.

What each model sees:
  Curator  - aggregate dataset profile (profiling.py) + protocol text
  Analyst  - protocol text + data dictionary/profile + one analysis spec
Neither ever receives a data row. Raw data only enters the sandbox.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .documents import extract_text, quote_found
from .llm import LLMError, ModelClient
from .profiling import profile, read_table
from .prompts import CODE, CURATOR, PLAN, REPAIR, REWRITE, WRITER, prompt_ref
from .sandbox import CodeRejected, Executor, check_code
from .canonical import sha256_text
from .workflow import Project, State, WorkflowError

CURATOR_ID, ANALYST_ID, WRITER_ID = "agent:curator", "agent:analyst", "agent:writer"

# Reasoning effort and output budget per task. Reasoning tokens are billed, so
# effort is spent only where judgement matters.
TASKS = {
    "curate": {"reasoning_effort": "low", "max_tokens": 8000},
    "plan": {"reasoning_effort": "high", "max_tokens": 16000},
    "code": {"reasoning_effort": "medium", "max_tokens": 12000},
    "repair": {"reasoning_effort": "medium", "max_tokens": 12000},
    "write": {"reasoning_effort": "medium", "max_tokens": 16000},
}
MAX_ATTEMPTS = 2
PROTOCOL_CHARS = 120_000


class AgentError(RuntimeError):
    """The agent could not produce a usable result. Logged calls remain in the log."""


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


@dataclass
class Agents:
    project: Project
    client: ModelClient
    executor: Executor

    # -- materials ---------------------------------------------------------
    def _latest(self, s: State, kind: str, sha: str | None = None) -> dict[str, Any]:
        ms = [m for m in s.materials if m["kind"] == kind and (sha is None or m["sha256"] == sha)]
        if not ms:
            raise AgentError(f"no {kind} uploaded" + (f" with sha {sha[:12]}" if sha else ""))
        return ms[-1]

    def _bytes(self, m: dict[str, Any]) -> bytes:
        data = (self.project.root / m["storage_ref"]).read_bytes()
        from .canonical import sha256_bytes
        if sha256_bytes(data) != m["sha256"]:
            raise AgentError(f"stored {m['kind']} does not match its recorded hash")
        return data

    def _protocol_text(self, s: State) -> tuple[str, dict[str, Any]]:
        m = self._latest(s, "protocol")
        return extract_text(self._bytes(m), m["filename"])[:PROTOCOL_CHARS], m

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
    def curate(self, requested_by: str, material_sha256: str | None = None) -> State:
        s = self.project.state()
        ds = self._latest(s, "dataset", material_sha256)
        prof = profile(read_table(self._bytes(ds), ds["filename"]))
        try:
            protocol, _ = self._protocol_text(s)
        except AgentError:
            protocol = "(no protocol uploaded yet)"
        user = (f"DATASET PROFILE (aggregate only):\n{json.dumps(prof, indent=1)}\n\n"
                f"PROTOCOL:\n{protocol}")
        r = self._chat("curate", CURATOR, user, json_mode=True, purpose="curate")
        try:
            out = parse_json(r.text)
        except AgentError:
            self.project.record_model_call(CURATOR_ID, r.record.to_payload())
            raise
        report = {"profile": prof, "dictionary": out.get("dictionary", []), "flags": out.get("flags", []),
                  "protocol_variables_missing_from_data": out.get("protocol_variables_missing_from_data", []),
                  "summary": out.get("summary", ""), "prompt": prompt_ref("curator")}
        return self.project.record_curator_report(CURATOR_ID, report, r.record.to_payload(),
                                                  requested_by=requested_by, material_sha256=ds["sha256"])

    # -- Analysis plan -------------------------------------------------------
    def propose_plan(self, requested_by: str) -> State:
        s = self.project.state()
        protocol, pm = self._protocol_text(s)
        d = self._dictionary(s)
        user = (f"PROTOCOL:\n{protocol}\n\nDATA DICTIONARY:\n{json.dumps(d['dictionary'], indent=1)}\n\n"
                f"DATASET PROFILE (aggregate only):\n{json.dumps(d['profile'], indent=1)}")
        r = self._chat("plan", PLAN, user, json_mode=True, purpose="propose_plan")
        try:
            out = parse_json(r.text)
            analyses = out.get("analyses") or []
            if not analyses:
                raise AgentError("the model proposed no analyses")
        except AgentError:
            self.project.record_model_call(ANALYST_ID, r.record.to_payload())
            raise
        cols = {c["name"] for c in d["profile"]["columns"]}
        for a in analyses:
            a["protocol_quote_verified"] = quote_found(a.get("protocol_quote", ""), protocol)
            named = [a.get("outcome") or {}] + list(a.get("exposures") or []) + list(a.get("covariates") or [])
            a["columns_not_in_data"] = sorted({v["column"] for v in named
                                               if isinstance(v, dict) and v.get("column") and v["column"] not in cols})
            a["prompt"] = prompt_ref("plan")
        return self.project.propose_plan(ANALYST_ID, analyses, protocol_ref=pm["sha256"],
                                         call=r.record.to_payload(), requested_by=requested_by)

    # -- Running analyses ------------------------------------------------------
    def run_planned(self, requested_by: str, plan_analysis_id: str | None = None) -> State:
        s = self.project.state()
        if s.plan_locked is None:
            raise WorkflowError("the analysis plan has not been confirmed by every author")
        ids = [plan_analysis_id] if plan_analysis_id else list(s.plan_locked["order"])
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
        base_user = (f"ANALYSIS SPECIFICATION:\n{json.dumps(spec, indent=1)}\n\n"
                     f"DATA DICTIONARY:\n{json.dumps(d['dictionary'], indent=1)}\n\n"
                     f"DATASET PROFILE (aggregate only; the CSV has these columns):\n{json.dumps(d['profile'], indent=1)}")
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
        execution = {"backend": getattr(self.executor, "backend", "unknown"), "seed": seed,
                     "dataset_sha256": ds["sha256"], "attempts": attempts,
                     "provenance": result.meta if result else {}, "prompt": prompt_ref("code")}
        return self.project.complete_analysis(
            ANALYST_ID, analysis_id, plan_analysis_id=plan_analysis_id, spec=spec,
            code_sha256=self.project.blobs.put_text(code),
            outputs_sha256=self.project.blobs.put_json(result.results) if ok else None,
            execution=execution, status="ok" if ok else "error", error=error, request_id=request_id,
            call=last_call, requested_by=requested_by, call_seqs=failed_call_seqs)

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
        results = {k: {"label": v["label"], "plan_analysis_id": v["plan_analysis_id"], "spec": v["spec"],
                       "results": self.project.blobs.get_json(v["outputs_sha256"])} for k, v in ok.items()}
        failed_brief = {k: {"label": v["label"], "plan_analysis_id": v["plan_analysis_id"],
                            "title": v["spec"].get("title"), "error": (v.get("error") or "")[:300]}
                        for k, v in failed.items()
                        # a failure later superseded by a successful rerun is not a failed analysis
                        if not any(o["plan_analysis_id"] == v["plan_analysis_id"] and v["plan_analysis_id"]
                                   for o in ok.values())}
        rows = d["profile"].get("rows")
        user = (f"PROTOCOL (for design and objectives only):\n{protocol[:40000]}\n\n"
                f"DATASET ROWS: {rows}\n\nDATA DICTIONARY:\n{json.dumps(d['dictionary'], indent=1)}\n\n"
                f"RESULTS (cite as [A:<key>]):\n{json.dumps(results, indent=1)}\n\n"
                f"PLANNED ANALYSES THAT FAILED:\n{json.dumps(failed_brief, indent=1)}")
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
    if findings.get("nothing_checked_but_numbers_present"):
        lines.append("- Numbers present but none could be checked: tag result sentences with [A:<key>]")
    return "\n".join(lines) or "- (no specific findings)"
