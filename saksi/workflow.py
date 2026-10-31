# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Saksi workflow engine.

Stages (OSAV white paper §2.2–§4):

  INGEST           authors upload dataset + protocol; curator agent profiles them;
                   analysis plan proposed and confirmed by every author
  ANALYSIS         analyst agent runs the confirmed (confirmatory) plan, then authors
                   iterate; anything outside the plan is labelled EXPLORATORY by the
                   engine and the label cannot be removed; every author accepts
  DRAFTING         writer agent drafts from computed outputs only
  AUTHOR_REVISION  authors edit (directly or via agents, always attributed); every
                   author signs off through structured verification
  REVIEW           human reviewer, assisted by an independent reviewer-assist agent,
                   writes item-by-item comments and gives verdicts on responses
  RESPONSE         authors revise / clarify / rebut each open item, sign off again,
                   and send it back to REVIEW (next round)
  ATTESTATION      reviewer has accepted with every item settled; each author attests
  FINAL            all authors have attested the final version

Design rules
  * State is never stored: it is rebuilt by replaying the hash-chained event log.
  * Stage changes are themselves events ("stage.advanced"), so the log reads as
    the full story of the project.
  * Commands check the rules before writing; a refused command writes nothing.
  * Sign-offs and acceptances bind to an exact version/hash; any later change
    voids them automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .blobs import BlobStore
from .canonical import sha256_bytes, sha256_obj
from .design import assess, sample_check
from .figures import saksi_figures
from .standard import KIND_ORDER, STEPS, clean_report, clean_table1, infer_kind, kind_of, position
from .documents import extract_text
from .events import Event, EventLog
from .tracing import CITE, SENTENCE, trace
from . import tables as tables_mod

STAGES = ("INGEST", "ANALYSIS", "DRAFTING", "AUTHOR_REVISION", "REVIEW", "RESPONSE", "ATTESTATION", "FINAL")
HUMAN_ROLES = {"author", "corresponding_author", "reviewer"}
AGENT_ROLES = {"curator", "analyst", "writer", "reviewer_assist"}
MATERIAL_KINDS = {"dataset", "protocol", "codebook", "ethics_approval", "registration", "other"}
RESPONSES = {"revised", "clarified", "rebutted"}
VERDICTS = {"satisfied", "still_concerned", "agree_to_disagree"}
SETTLED = {"satisfied", "agree_to_disagree"}
SYSTEM = "system"

# Structured sign-off (white paper §2.2: no bare "Approve" button).
SIGNOFF_QUESTIONS = (
    "Which result in this draft would you be least comfortable defending, and why?",
    "Is every confirmatory claim supported by a protocol-specified analysis?",
    "Are all exploratory analyses described as exploratory in the text?",
)
MIN_RESTATEMENT_WORDS = 8
DEFAULT_MAX_ROUNDS = 5

# Default agent roster. The reviewer-assist agent runs on a different model
# deployment (open weights, via Baseten) and never sees author-side history.
DEFAULT_AGENTS = (
    {"actor_id": "agent:curator", "name": "Data Curator", "role": "curator", "provider": "meta", "model": "muse-spark-1.3"},
    {"actor_id": "agent:analyst", "name": "Analysis Agent", "role": "analyst", "provider": "meta", "model": "muse-spark-1.3"},
    {"actor_id": "agent:writer", "name": "Writing Agent", "role": "writer", "provider": "meta", "model": "muse-spark-1.3"},
    {"actor_id": "agent:reviewer_assist", "name": "Reviewer Assistant", "role": "reviewer_assist",
     "provider": "baseten", "model": "meta-models/muse-spark-open-1026"},
)

ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")


class WorkflowError(Exception):
    """A command was refused. Nothing was written."""


# ---------------------------------------------------------------------------
# State (rebuilt from events)
# ---------------------------------------------------------------------------

@dataclass
class State:
    project_id: str = ""
    title: str = ""
    study_type: str = ""
    max_rounds: int = DEFAULT_MAX_ROUNDS
    stage: str = "INGEST"
    round: int = 0
    actors: dict[str, dict[str, Any]] = field(default_factory=dict)
    materials: list[dict[str, Any]] = field(default_factory=list)
    curator_reports: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] | None = None              # {"sha": ..., "analyses": {id: spec}}
    plan_confirmed_by: set[str] = field(default_factory=set)
    plan_locked: dict[str, Any] | None = None       # plan as confirmed by all authors
    analyses: dict[str, dict[str, Any]] = field(default_factory=dict)
    analyses_accepted_by: dict[str, set[str]] = field(default_factory=dict)  # set-hash -> authors
    draft_version: int = 0
    drafts: dict[int, dict[str, Any]] = field(default_factory=dict)
    opened: dict[tuple[str, int], str] = field(default_factory=dict)
    signoffs: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict)
    briefings: list[dict[str, Any]] = field(default_factory=list)
    reviewed_versions: dict[int, int] = field(default_factory=dict)
    figures: list[dict[str, Any]] = field(default_factory=list)
    answers: list[dict[str, Any]] = field(default_factory=list)          # authors' answers to Saksi's questions
    suggested_answers: dict[str, Any] = field(default_factory=dict)      # question -> Saksi's suggested answer
    outline: dict[str, Any] | None = None                                # what the paper reports, and its Methods draft
    preparations: list[dict[str, Any]] = field(default_factory=list)     # data preparation proposals
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    review_decisions: dict[int, str] = field(default_factory=dict)
    attestations: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict)
    head_seq: int = 0

    # -- derived views -----------------------------------------------------
    def humans(self, role: str) -> list[str]:
        return sorted(a for a, v in self.actors.items() if v["kind"] == "human" and role in v["roles"])

    @property
    def authors(self) -> list[str]:
        return self.humans("author")

    @property
    def reviewers(self) -> list[str]:
        return self.humans("reviewer")

    def analysis_set_sha(self) -> str:
        return sha256_obj({k: {"outputs_sha256": v.get("outputs_sha256"), "label": v["label"], "status": v["status"]}
                           | ({"withdrawn": True} if v.get("withdrawn") else {})
                           for k, v in sorted(self.analyses.items())})

    def open_items(self) -> list[str]:
        return sorted(i for i, v in self.items.items() if v["status"] == "open")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot for the API and UI."""
        return {
            "project_id": self.project_id, "title": self.title, "study_type": self.study_type,
            "stage": self.stage, "round": self.round, "max_rounds": self.max_rounds,
            "actors": self.actors, "authors": self.authors, "reviewers": self.reviewers,
            "materials": self.materials, "curator_reports": self.curator_reports,
            "plan": self.plan, "plan_confirmed_by": sorted(self.plan_confirmed_by),
            "plan_locked": self.plan_locked, "analyses": self.analyses,
            "analysis_set_sha256": self.analysis_set_sha() if self.analyses else None,
            "analyses_accepted_by": sorted(self.analyses_accepted_by.get(self.analysis_set_sha(), set())),
            "draft_version": self.draft_version, "drafts": {str(k): v for k, v in self.drafts.items()},
            "signoffs": {str(k): v for k, v in self.signoffs.items()},
            "opened": {f"{a}|{v}": ts for (a, v), ts in self.opened.items()},
            "briefings": self.briefings, "items": self.items,
            "review_decisions": {str(k): v for k, v in self.review_decisions.items()},
            "reviewed_versions": {str(k): v for k, v in self.reviewed_versions.items()},
            "figures": self.figures, "answers": self.answers, "suggested_answers": self.suggested_answers, "outline": self.outline, "preparations": self.preparations,
            "attestations": {str(k): v for k, v in self.attestations.items()},
            "signoff_questions": list(SIGNOFF_QUESTIONS), "head_seq": self.head_seq,
        }


def apply(s: State, ev: Event) -> State:
    """Pure reducer: fold one event into the state."""
    p, t = ev.payload, ev.type
    s.head_seq = ev.seq
    if t == "project.created":
        s.project_id, s.title, s.study_type = ev.project_id, p["title"], p["study_type"]
        s.max_rounds = p.get("max_rounds", DEFAULT_MAX_ROUNDS)
    elif t == "actor.registered":
        s.actors[p["actor_id"]] = {k: v for k, v in p.items() if k != "actor_id"} | {"registered_seq": ev.seq}
    elif t == "stage.advanced":
        s.stage = p["to"]
        if "round" in p:
            s.round = p["round"]
    elif t == "material.uploaded":
        s.materials.append(p | {"by": ev.actor_id, "seq": ev.seq, "ts": ev.ts})
    elif t == "material.withdrawn":
        for m in s.materials:
            if m["sha256"] == p["sha256"] and not m.get("withdrawn"):
                m["withdrawn"] = {"by": ev.actor_id, "reason": p["reason"], "seq": ev.seq}
    elif t == "curator.reported":
        s.curator_reports.append(p | {"seq": ev.seq, "ts": ev.ts})
    elif t == "plan.proposed":
        analyses = {}
        for k, a in p["analyses"].items():
            a = dict(a)
            if a.get("kind") not in KIND_ORDER:      # a plan from before the standard sequence: kind by rule
                a["kind"], a["kind_inferred"] = infer_kind(a), True
            analyses[k] = a
        s.plan = {"sha": p["plan_sha256"], "analyses": analyses, "protocol_ref": p.get("protocol_ref"),
                  "order": p.get("order") or sorted(p["analyses"]), "by": ev.actor_id, "seq": ev.seq,
                  "note": p.get("note"), "changes": p.get("changes"), "requested_by": p.get("requested_by"),
                  "table1": p.get("table1"), "report": p.get("report")}
        s.plan_confirmed_by = set()
    elif t == "plan.confirmed":
        s.plan_confirmed_by.add(ev.actor_id)
    elif t == "plan.locked":
        s.plan_locked = {"sha": p["plan_sha256"], "analyses": s.plan["analyses"], "order": s.plan["order"],
                         "table1": s.plan.get("table1"), "report": s.plan.get("report"), "seq": ev.seq}
    elif t == "analysis.requested":
        pass  # kept in the log for attribution; results arrive as analysis.completed
    elif t == "analysis.completed":
        s.analyses[p["analysis_id"]] = p | {"by": ev.actor_id, "seq": ev.seq}
        planned = (s.plan_locked or {}).get("analyses", {}).get(p.get("plan_analysis_id") or "")
        if planned and planned.get("kind_inferred"):
            s.analyses[p["analysis_id"]]["kind"] = planned["kind"]
        if p.get("supersedes"):
            old = s.analyses[p["supersedes"]]
            old["superseded_by"] = p["analysis_id"]
            s.analyses[p["analysis_id"]]["order_seq"] = old.get("order_seq", old["seq"])
        s.figures.extend(f | {"analysis_id": p["analysis_id"], "seq": ev.seq} for f in p.get("figures") or [])
        _number_figures(s)
    elif t == "question.answered":
        s.answers.append(p | {"by": ev.actor_id, "seq": ev.seq, "ts": ev.ts})
    elif t == "plan.outlined":
        s.outline = p["outline"] | {k: p.get(k) for k in ("methods_sha256", "plan_seq", "answers", "requested_by")} \
            | {"by": ev.actor_id, "seq": ev.seq, "ts": ev.ts, "edited_by": []}
    elif t == "plan.outline_edited":
        if s.outline:
            s.outline = s.outline | {"methods_sha256": p["methods_sha256"], "seq": ev.seq,
                                     "edited_by": s.outline.get("edited_by", []) + [ev.actor_id]}
    elif t == "questions.suggested":
        for x in p["suggestions"]:
            s.suggested_answers[x["question"]] = x | {"seq": ev.seq}
    elif t == "data.preparation_proposed":
        s.preparations.append(p | {"by": ev.actor_id, "seq": ev.seq})
    elif t == "checklist.assessed":
        if p["version"] in s.drafts:
            s.drafts[p["version"]]["checklist"] = p | {"seq": ev.seq}
    elif t == "figure.omitted":
        for f in s.figures:
            if f["analysis_id"] == p["analysis_id"] and f["name"] == p["name"]:
                f["omitted"] = {"by": ev.actor_id, "reason": p["reason"], "seq": ev.seq}
        _number_figures(s)
    elif t == "figure.restored":
        for f in s.figures:
            if f["analysis_id"] == p["analysis_id"] and f["name"] == p["name"]:
                f.pop("omitted", None)
        _number_figures(s)
    elif t == "analysis.withdrawn":
        s.analyses[p["analysis_id"]]["withdrawn"] = {"reason": p["reason"], "by": ev.actor_id, "seq": ev.seq}
        _number_figures(s)
    elif t == "analysis.described":
        if p["analysis_id"] in s.analyses:
            s.analyses[p["analysis_id"]]["description"] = p | {"by": ev.actor_id, "seq": ev.seq}
    elif t == "analyses.accepted":
        s.analyses_accepted_by.setdefault(p["analysis_set_sha256"], set()).add(ev.actor_id)
    elif t in ("draft.created", "draft.edited"):
        s.draft_version = p["version"]
        s.drafts[p["version"]] = p | {"by": ev.actor_id, "type": t, "seq": ev.seq, "ts": ev.ts}
    elif t == "draft.opened":
        s.opened.setdefault((ev.actor_id, p["version"]), ev.ts)
    elif t == "draft.signed_off":
        s.signoffs.setdefault(p["version"], {})[ev.actor_id] = p | {"seq": ev.seq, "ts": ev.ts}
    elif t == "review.briefing":
        s.briefings.append(p | {"seq": ev.seq})
    elif t == "review.item_added":
        s.items[p["item_id"]] = {**p, "by": ev.actor_id, "status": "open", "responses": [], "verdicts": [],
                                 "seq": ev.seq, "draft_version_at_creation": s.draft_version}
    elif t == "review.item_responded":
        it = s.items[p["item_id"]]
        it["responses"].append(p | {"by": ev.actor_id, "round": s.round, "seq": ev.seq})
        it["status"] = "responded"
    elif t == "review.item_verdict":
        it = s.items[p["item_id"]]
        it["verdicts"].append(p | {"by": ev.actor_id, "round": s.round, "seq": ev.seq})
        it["status"] = "open" if p["verdict"] == "still_concerned" else p["verdict"]
    elif t == "review.submitted":
        s.review_decisions[p["round"]] = p["decision"]
        if "draft_version" in p:
            s.reviewed_versions[p["round"]] = p["draft_version"]
    elif t == "response.submitted":
        pass
    elif t == "authorship.attested":
        s.attestations.setdefault(p["version"], {})[ev.actor_id] = p | {"seq": ev.seq, "ts": ev.ts}
    elif t == "llm.called":
        pass
    return s


def replay(events: list[Event]) -> State:
    s = State()
    for ev in events:
        apply(s, ev)
    return s


# ---------------------------------------------------------------------------
# Project: commands that check rules, then append events
# ---------------------------------------------------------------------------

class Project:
    def __init__(self, root: str | Path, project_id: str, clock=None):
        self.root = Path(root) / project_id
        self.project_id = project_id
        kw = {"clock": clock} if clock else {}
        self.log = EventLog(self.root / "events.jsonl", project_id, **kw)
        self.blobs = BlobStore(Path(root) / "blobs")

    # -- infrastructure ----------------------------------------------------
    def state(self) -> State:
        return replay(self.log.events())

    def _emit(self, actor_id: str, type: str, payload: dict[str, Any]) -> Event:
        return self.log.append(actor_id, type, payload)

    def _actor(self, s: State, actor_id: str) -> dict[str, Any]:
        if actor_id not in s.actors:
            raise WorkflowError(f"unknown actor {actor_id!r}")
        return s.actors[actor_id]

    def _require_role(self, s: State, actor_id: str, *roles: str) -> dict[str, Any]:
        a = self._actor(s, actor_id)
        if not set(roles) & set(a["roles"]):
            raise WorkflowError(f"{actor_id} ({', '.join(a['roles'])}) may not do this; needs {' or '.join(roles)}")
        return a

    def _require_stage(self, s: State, *stages: str) -> None:
        if s.stage not in stages:
            raise WorkflowError(f"not allowed in stage {s.stage}; allowed in {', '.join(stages)}")

    def _advance(self, s: State, to: str, reason: str, **extra: Any) -> None:
        self._emit(SYSTEM, "stage.advanced", {"from": s.stage, "to": to, "reason": reason, **extra})

    def _requested_by(self, s: State, agent: dict[str, Any], requested_by: str | None) -> None:
        """Agent actions taken for a human must name that human."""
        if agent["kind"] == "agent" and requested_by is not None:
            self._require_role(s, requested_by, "author", "reviewer")

    def _log_call(self, actor_id: str, call: dict[str, Any] | None) -> int | None:
        if call is None:
            return None
        return self._emit(actor_id, "llm.called", call).seq

    def record_model_call(self, by: str, call: dict[str, Any]) -> int:
        """Log a model call by an agent (e.g. a failed attempt) and return its event number."""
        s = self.state()
        if self._actor(s, by)["kind"] != "agent":
            raise WorkflowError("only agents make model calls")
        return self._emit(by, "llm.called", call).seq

    # -- setup -------------------------------------------------------------
    def create(self, title: str, study_type: str, creator: dict[str, Any],
               max_rounds: int = DEFAULT_MAX_ROUNDS, agents=DEFAULT_AGENTS) -> State:
        if self.log.events():
            raise WorkflowError("project already exists")
        if study_type not in {"primary", "evidence_synthesis"}:
            raise WorkflowError("study_type must be 'primary' or 'evidence_synthesis'")
        self._emit(creator["actor_id"], "project.created",
                   {"title": title, "study_type": study_type, "max_rounds": max_rounds})
        self._emit(SYSTEM, "actor.registered", self._human_payload(
            creator, roles=["author", "corresponding_author"]))
        for a in agents:
            self._emit(SYSTEM, "actor.registered", {"actor_id": a["actor_id"], "kind": "agent", "name": a["name"],
                                                    "roles": [a["role"]], "provider": a["provider"], "model": a["model"]})
        return self.state()

    @staticmethod
    def _human_payload(actor: dict[str, Any], roles: list[str]) -> dict[str, Any]:
        aid, name = actor.get("actor_id", ""), actor.get("name", "").strip()
        if not aid or aid.startswith(("agent:", SYSTEM)):
            raise WorkflowError("human actor_id required (and may not start with 'agent:' or 'system')")
        if not name:
            raise WorkflowError("name required")
        orcid = actor.get("orcid")
        if orcid and not ORCID_RE.match(orcid):
            raise WorkflowError(f"ORCID iD {orcid!r} is not in the form 0000-0000-0000-0000")
        return {"actor_id": aid, "kind": "human", "name": name, "orcid": orcid,
                "affiliation": actor.get("affiliation"), "roles": roles}

    def register_human(self, by: str, actor: dict[str, Any], role: str) -> State:
        s = self.state()
        self._require_role(s, by, "corresponding_author")
        if s.stage == "FINAL":
            raise WorkflowError("project is final")
        if role not in {"author", "reviewer"}:
            raise WorkflowError("role must be 'author' or 'reviewer'")
        if actor.get("actor_id") in s.actors:
            raise WorkflowError(f"{actor['actor_id']} is already registered")
        if role == "author" and s.stage not in {"INGEST", "ANALYSIS"}:
            raise WorkflowError("authors join before drafting starts, so every author can verify the analysis")
        orcid = actor.get("orcid")
        if orcid and any(v.get("orcid") == orcid for v in s.actors.values()):
            raise WorkflowError(f"ORCID iD {orcid} already belongs to a registered actor")
        self._emit(SYSTEM, "actor.registered", self._human_payload(actor, roles=[role]) | {"invited_by": by})
        return self.state()

    # -- INGEST ------------------------------------------------------------
    def upload_material(self, by: str, kind: str, filename: str, sha256: str, size: int,
                        storage_ref: str | None = None, note: str | None = None,
                        inspection: dict[str, Any] | None = None, prepared: dict[str, Any] | None = None) -> State:
        s = self.state()
        self._require_role(s, by, "author")
        if kind not in MATERIAL_KINDS:
            raise WorkflowError(f"kind must be one of {sorted(MATERIAL_KINDS)}")
        if s.stage == "FINAL":
            raise WorkflowError("the paper is final")
        if kind in ("dataset", "protocol") and s.stage != "INGEST":
            raise WorkflowError("the dataset and the protocol are fixed once every author has confirmed the plan, because "
                                "the results depend on them; other materials (codebook, ethics approval, registration, "
                                "other documents) can be added at any time")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise WorkflowError("sha256 must be 64 lowercase hex characters")
        self._emit(by, "material.uploaded", {"kind": kind, "filename": filename, "sha256": sha256,
                                             "bytes": size, "storage_ref": storage_ref, "note": note,
                                             "inspection": inspection}
                   | ({"prepared_from": prepared["source_sha256"], "preparation": prepared} if prepared else {}))
        return self.state()

    # -- Saksi's questions and the authors' answers ---------------------------
    def record_outline(self, by: str, outline: dict[str, Any], methods: str, requested_by: str,
                       call: dict[str, Any] | None = None) -> State:
        """The Writing Agent's account of what this paper reports and how, with a Methods draft."""
        s = self.state()
        a = self._require_role(s, by, "writer")
        self._requested_by(s, a, requested_by)
        call_seq = self._log_call(by, call)
        self._emit(by, "plan.outlined", {"outline": outline, "methods_sha256": self.blobs.put_text(methods),
                                          "plan_seq": (s.plan or {}).get("seq"), "answers": len(s.answers),
                                          "requested_by": requested_by, "llm_call_seq": call_seq})
        return self.state()

    def edit_outline_methods(self, by: str, text: str) -> State:
        """An author edits the Methods draft on the Plan page; recorded under their name."""
        s = self.state()
        self._require_role(s, by, "author")
        if not s.outline:
            raise WorkflowError("there is no Methods draft yet")
        if not (text or "").strip():
            raise WorkflowError("the Methods cannot be empty")
        if self.blobs.get_text(s.outline["methods_sha256"]) == text:
            raise WorkflowError("nothing changed")
        self._emit(by, "plan.outline_edited", {"methods_sha256": self.blobs.put_text(text)})
        return self.state()

    def suggest_answers(self, by: str, suggestions: list[dict[str, Any]], requested_by: str,
                        call: dict[str, Any] | None = None) -> State:
        """Saksi's suggested answers to its own questions, for the authors to accept or change."""
        s = self.state()
        a = self._require_role(s, by, "analyst")
        self._requested_by(s, a, requested_by)
        call_seq = self._log_call(by, call)
        self._emit(by, "questions.suggested", {"suggestions": suggestions, "requested_by": requested_by,
                                                "llm_call_seq": call_seq})
        return self.state()

    def answer_question(self, by: str, question: str, answer: str, accepted_suggestion: bool = False) -> State:
        """An author answers one of Saksi's questions about this paper. Recorded under their name and given to the
        agents from then on (plan, data preparation, analysis, writing)."""
        s = self.state()
        self._require_role(s, by, "author")
        if s.stage == "FINAL":
            raise WorkflowError("the paper is final")
        if not (question or "").strip() or not (answer or "").strip():
            raise WorkflowError("give the question and your answer")
        self._emit(by, "question.answered", {"question": question.strip()[:600], "answer": answer.strip()[:2000],
                                              "stage": s.stage, "accepted_suggestion": bool(accepted_suggestion)})
        return self.state()

    # -- Data preparation ------------------------------------------------------
    def propose_preparation(self, by: str, source_sha256: str, steps: list[dict[str, Any]], summary: str,
                            problems: list[str], requested_by: str, call: dict[str, Any] | None = None,
                            instructions: str | None = None) -> State:
        s = self.state()
        a = self._require_role(s, by, "analyst")
        self._require_stage(s, "INGEST")
        self._requested_by(s, a, requested_by)
        call_seq = self._log_call(by, call)
        self._emit(by, "data.preparation_proposed", {"source_sha256": source_sha256, "steps": steps, "summary": summary,
                                                      "problems": problems, "instructions": instructions,
                                                      "requested_by": requested_by, "llm_call_seq": call_seq})
        return self.state()

    def prepare_dataset(self, by: str, step_ids: list[str], note: str = "") -> State:
        """Apply the steps an author chose from the latest proposal, by Saksi's own code, to the original upload.
        The prepared file is stored beside the original (which is kept) and is the one analysed from then on."""
        from .cleaning import apply, validate
        from .inspection import inspect_material
        from .profiling import read_table
        s = self.state()
        self._require_role(s, by, "author")
        self._require_stage(s, "INGEST")
        if not s.preparations:
            raise WorkflowError("there is no proposal to apply; ask Saksi to prepare the data first")
        prop = s.preparations[-1]
        src = next((m for m in s.materials if m["sha256"] == prop["source_sha256"] and not m.get("withdrawn")), None)
        if src is None or not src.get("storage_ref") or not (self.root / src["storage_ref"]).exists():
            raise WorkflowError("the dataset this proposal was made for is no longer available")
        chosen = [st for st in prop["steps"] if st["id"] in set(step_ids)]
        if not chosen:
            raise WorkflowError("tick at least one step")
        raw = (self.root / src["storage_ref"]).read_bytes()
        df = read_table(raw, src["filename"])
        steps, problems = validate(chosen, [str(c) for c in df.columns])
        if problems:
            raise WorkflowError("; ".join(problems))
        out, log = apply(df, steps)
        from .cleaning import compare
        comparison = compare(df, out)
        data = out.to_csv(index=False).encode()
        sha = sha256_bytes(data)
        (self.root / "materials").mkdir(parents=True, exist_ok=True)
        (self.root / "materials" / sha).write_bytes(data)
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", src["filename"])
        prepared = {"source_sha256": src["sha256"], "source_filename": src["filename"], "proposal_seq": prop["seq"],
                    "steps": steps, "log": log, "skipped": [st["id"] for st in prop["steps"] if st["id"] not in set(step_ids)],
                    "rows_before": int(len(df)), "columns_before": int(df.shape[1]),
                    "rows_after": int(len(out)), "columns_after": int(out.shape[1]), "note": note,
                    "comparison": comparison}
        return self.upload_material(by, "dataset", f"{stem}-prepared.csv", sha, len(data), storage_ref=f"materials/{sha}",
                                    note=note or "Prepared by Saksi's code from the steps the authors chose",
                                    inspection=inspect_material("dataset", f"{stem}-prepared.csv", data), prepared=prepared)

    def withdraw_material(self, by: str, sha256: str, reason: str) -> State:
        """Remove an uploaded file, at any stage before the paper is final. The stored copy is deleted; the record keeps
        its name, hash, who removed it and why. Removing the dataset or protocol after the plan is confirmed is recorded
        as such: the results stay in the record, but they can no longer be re-run or checked against the file."""
        s = self.state()
        self._require_role(s, by, "author")
        if s.stage == "FINAL":
            raise WorkflowError("the paper is final")
        hits = [m for m in s.materials if m["sha256"] == sha256 and not m.get("withdrawn")]
        if not hits:
            raise WorkflowError("no file with that hash is in this project")
        if not (reason or "").strip():
            raise WorkflowError("say why you are removing it; the record keeps the reason")
        deleted = False
        for m in hits:
            if m.get("storage_ref"):
                path = self.root / m["storage_ref"]
                if path.exists():
                    path.unlink()
                    deleted = True
        self._emit(by, "material.withdrawn", {"sha256": sha256, "kind": hits[-1]["kind"],
                                              "filename": hits[-1]["filename"], "reason": reason.strip(),
                                              "stored_copy_deleted": deleted,
                                              "after_plan_locked": s.plan_locked is not None
                                              and hits[-1]["kind"] in ("dataset", "protocol")})
        return self.state()

    def record_curator_report(self, by: str, report: dict[str, Any], call: dict[str, Any] | None,
                              requested_by: str | None = None, material_sha256: str | None = None) -> State:
        s = self.state()
        a = self._require_role(s, by, "curator")
        if s.stage == "FINAL":        # the report is kept up to date as the work goes on; it changes no result
            raise WorkflowError("the paper is final")
        self._requested_by(s, a, requested_by)
        call_seq = self._log_call(by, call)
        self._emit(by, "curator.reported", {"report_sha256": self.blobs.put_json(report),
                                            "summary": report.get("summary"), "material_sha256": material_sha256,
                                            "requested_by": requested_by, "llm_call_seq": call_seq})
        return self.state()

    def propose_plan(self, by: str, analyses: list[dict[str, Any]], protocol_ref: str | None,
                     call: dict[str, Any] | None = None, requested_by: str | None = None,
                     note: str | None = None, table1: dict[str, Any] | None = None,
                     report: dict[str, Any] | None = None) -> State:
        """A plan from the Analysis Agent, or a revision by an author. Authors decide which protocol
        analyses this paper includes; leaving one out needs a reason, which the record and the draft keep."""
        s = self.state()
        a = self._require_role(s, by, "analyst", "author")
        self._require_stage(s, "INGEST")
        self._requested_by(s, a, requested_by)
        ids = [x.get("id") for x in analyses]
        if not analyses or any(not i for i in ids) or len(set(ids)) != len(ids):
            raise WorkflowError("plan needs at least one analysis, each with a unique id")
        for x in analyses:
            x["included"] = x.get("included", True) is not False
            if not x["included"] and not str(x.get("exclusion_reason") or "").strip():
                raise WorkflowError(f"say why {x['id']!r} is left out of this paper")
        if not any(x["included"] for x in analyses):
            raise WorkflowError("include at least one analysis")
        if a["kind"] == "human" and s.plan and not (note or "").strip():
            raise WorkflowError("say what you changed in the plan and why")
        # Confirmatory or exploratory is decided here, by code, against the stored protocol,
        # whoever proposed the plan. Anything the proposer claimed about it is overwritten.
        text = self._protocol_text(s, protocol_ref)
        old = (s.plan or {}).get("analyses") or {}
        for x in analyses:
            if a["kind"] != "human":
                # Naming a primary the protocol does not, or changing the protocol's, is the authors' call.
                x.pop("primary_reason", None); x.pop("primary_reason_by", None)
            elif str(x.get("primary_reason") or "").strip():
                same = str((old.get(x["id"]) or {}).get("primary_reason") or "").strip() == str(x["primary_reason"]).strip()
                x["primary_reason_by"] = (old.get(x["id"]) or {}).get("primary_reason_by") if same else by
            else:
                x.pop("primary_reason_by", None)
        plan = {}
        for x in analyses:
            d = assess(x, text)
            if a["kind"] == "human" and d["deviation"] and not d["deviation"]["reason"] and x["included"]:
                raise WorkflowError(f"the protocol names {x['id']!r} as primary; say why it is now a {d['role']} "
                                    "analysis (the draft will disclose it)")
            plan[x["id"]] = {**x, "design": d, "label": d["label"],
                             "kind": kind_of(x),
                             "protocol_quote_verified": d["checks"]["protocol_quote_verified"]}
        order = [x["id"] for x in analyses]          # protocol order; dict keys are sorted when hashed
        changes = _plan_changes(s.plan, plan) if s.plan else None
        t1 = clean_table1(table1)
        rep = clean_report(report)
        hashed = {"analyses": plan, "order": order} | ({"table1": t1} if t1 else {}) | ({"report": rep} if rep else {})
        call_seq = self._log_call(by, call)
        self._emit(by, "plan.proposed", {"analyses": plan, "order": order, "note": note, "changes": changes,
                                         "table1": t1, "report": rep, "plan_sha256": sha256_obj(hashed),
                                         "protocol_ref": protocol_ref, "llm_call_seq": call_seq,
                                         "requested_by": requested_by})
        return self.state()

    def _protocol_text(self, s: State, ref: str | None) -> str | None:
        """The stored protocol's text, only if the file is present and still matches its hash."""
        m = next((m for m in reversed(s.materials) if m["kind"] == "protocol" and m["sha256"] == ref
                  and m.get("storage_ref") and not m.get("withdrawn")), None)
        if m is None:
            return None
        path = self.root / m["storage_ref"]
        if not path.exists():
            return None
        data = path.read_bytes()
        if sha256_bytes(data) != m["sha256"]:
            return None
        return extract_text(data, m["filename"])

    def confirm_plan(self, by: str, plan_sha256: str) -> State:
        s = self.state()
        self._require_role(s, by, "author")
        self._require_stage(s, "INGEST")
        if not s.plan:
            raise WorkflowError("no plan has been proposed")
        if plan_sha256 != s.plan["sha"]:
            raise WorkflowError("this plan has changed since you read it; review the current version")
        live = [m for m in s.materials if not m.get("withdrawn")]
        if not any(m["kind"] == "dataset" for m in live) or not any(m["kind"] == "protocol" for m in live):
            raise WorkflowError("a dataset and a protocol must be uploaded before the plan can be confirmed")
        ref = s.plan.get("protocol_ref")
        removed = {m["sha256"] for m in s.materials if m.get("withdrawn")} - {m["sha256"] for m in live}
        if ref in removed:
            raise WorkflowError("this plan was drafted from a protocol that has since been removed; ask for a new draft")
        self._emit(by, "plan.confirmed", {"plan_sha256": plan_sha256})
        s = self.state()
        if set(s.authors) <= s.plan_confirmed_by:
            self._emit(SYSTEM, "plan.locked", {"plan_sha256": s.plan["sha"]})
            self._advance(s, "ANALYSIS", "every author confirmed the analysis plan")
        return self.state()

    # -- ANALYSIS ----------------------------------------------------------
    def request_analysis(self, by: str, request_id: str, prompt: str) -> State:
        s = self.state()
        self._require_role(s, by, "author")
        self._require_stage(s, "ANALYSIS", "RESPONSE")
        if not prompt.strip():
            raise WorkflowError("prompt is empty")
        self._emit(by, "analysis.requested", {"request_id": request_id, "prompt": prompt})
        return self.state()

    def complete_analysis(self, by: str, analysis_id: str, *, plan_analysis_id: str | None,
                          spec: dict[str, Any], code_sha256: str, outputs_sha256: str | None,
                          execution: dict[str, Any], status: str, error: str | None = None,
                          request_id: str | None = None, call: dict[str, Any] | None = None,
                          requested_by: str | None = None, call_seqs: list[int] | None = None,
                          figures: list[dict[str, Any]] | None = None, standard: str | None = None,
                          supersedes: str | None = None, redo_reason: str | None = None) -> State:
        s = self.state()
        if supersedes is not None:
            # A redo: the new run replaces an earlier one, which stays in the record, marked as replaced.
            old = s.analyses.get(supersedes)
            if old is None or old.get("superseded_by"):
                raise WorkflowError(f"{supersedes!r} is not a current analysis")
            if old.get("plan_analysis_id") != plan_analysis_id or old.get("standard") != standard:
                raise WorkflowError("a redo must be of the same analysis")
            if not (redo_reason or "").strip() or requested_by is None:
                raise WorkflowError("a redo needs the author who asked for it and the reason")
        if standard is not None:
            # A step of the standard sequence, run with Saksi's own code (see standard.py).
            if standard not in STEPS or plan_analysis_id is not None:
                raise WorkflowError(f"no standard step {standard!r}")
            if by != SYSTEM:
                raise WorkflowError("standard steps are run by Saksi itself")
            if requested_by is not None:
                self._require_role(s, requested_by, "author")
            a = {"kind": "system"}
        else:
            a = self._require_role(s, by, "analyst")
        self._require_stage(s, "ANALYSIS", "RESPONSE")
        self._requested_by(s, a, requested_by)
        if status not in {"ok", "error"}:
            raise WorkflowError("status must be 'ok' or 'error'")
        if plan_analysis_id is not None and plan_analysis_id not in s.plan_locked["analyses"]:
            raise WorkflowError(f"{plan_analysis_id!r} is not in the confirmed plan")
        if plan_analysis_id is not None and not s.plan_locked["analyses"][plan_analysis_id].get("included", True):
            raise WorkflowError(f"{plan_analysis_id!r} was left out of this paper when the plan was confirmed")
        # The label comes from the confirmed plan, fixed before any result; never from the caller.
        planned = s.plan_locked["analyses"][plan_analysis_id] if plan_analysis_id is not None else None
        label = "descriptive" if standard else planned.get("label", "exploratory") if planned else "exploratory"
        outputs = self.blobs.get_json(outputs_sha256) if status == "ok" and outputs_sha256 \
            and self.blobs.has(outputs_sha256) else None
        check = sample_check(planned.get("design") if planned else None, outputs)
        # Author's definition (31 Oct): a pre-specified primary analysis is confirmatory only if the completed study
        # reached the sample its protocol planned. Decided by code from the counts analysed, never the estimates.
        downgraded = None
        if label == "confirmatory" and check and check.get("shortfall"):
            label, downgraded = "exploratory", ("the achieved sample fell short of the sample the protocol planned, so the "
                                                "study could not answer its question with the power it planned")
        basis = "standard approach" if standard else "pre-specified" if planned else "post hoc"
        figs = self._figures(s, analysis_id, spec, outputs, label, basis, figures or [], replacing=supersedes) \
            if outputs is not None and not standard else []
        call_seq = self._log_call(by, call)
        self._emit(by, "analysis.completed", {
            "analysis_id": analysis_id, "plan_analysis_id": plan_analysis_id, "label": label,
            "basis": basis, "standard": standard, "kind": standard or (planned or {}).get("kind") or (spec.get("kind") if spec.get("kind") in KIND_ORDER else "other"),
            "primary_basis": ((planned or {}).get("design") or {}).get("primary_basis"),
            "supersedes": supersedes, "redo_reason": redo_reason, "downgraded": downgraded,
            "protocol_deviation": ((planned or {}).get("design") or {}).get("deviation"),
            "sample_check": check, "figures": figs,
            "spec": spec, "code_sha256": code_sha256, "outputs_sha256": outputs_sha256,
            "execution": execution, "status": status, "error": error, "request_id": request_id,
            "added_in_stage": s.stage, "added_in_round": s.round, "llm_call_seq": call_seq,
            "llm_call_seqs": (call_seqs or []) + ([call_seq] if call_seq else []), "requested_by": requested_by,
        })
        return self.state()

    def _figures(self, s: State, analysis_id: str, spec: dict[str, Any], outputs: dict[str, Any], label: str,
                 basis: str, from_code: list[dict[str, Any]], replacing: str | None = None) -> list[dict[str, Any]]:
        """Saksi's own figures plus those drawn by the analysis code. Numbered in order of creation;
        every caption is number-checked against this analysis's results."""
        rows = None
        if s.curator_reports:
            rows = self.blobs.get_json(s.curator_reports[-1]["report_sha256"]).get("profile", {}).get("rows")
        made = [{**f, "svg_sha256": self.blobs.put_text(f.pop("svg")), "drawn_by": "saksi"}
                for f in saksi_figures(spec, outputs, rows, label, basis)]
        # One flow diagram per distinct set of counts, however each analysis worded its exclusions.
        live = [f for f in s.figures if f["analysis_id"] != replacing
                and not s.analyses.get(f["analysis_id"], {}).get("superseded_by")]
        seen_svg = {f["svg_sha256"] for f in live}
        seen_flow = {tuple(f["flow_counts"]) for f in live if f.get("flow_counts")}
        made = [f for f in made if f["svg_sha256"] not in seen_svg
                and not (f.get("flow_counts") and tuple(f["flow_counts"]) in seen_flow)]
        # The label is Saksi's fact, so Saksi states it on captions written by analysis code.
        prefix = ""      # the label belongs in the text, not in captions (author's decision, 31 Oct)
        for f in from_code:
            if not self.blobs.has(f.get("svg_sha256", "")):
                continue
            cap = str(f.get("caption", ""))
            made.append({"name": str(f["name"]), "svg_sha256": f["svg_sha256"], "drawn_by": "analysis code",
                         "caption": cap if not prefix or "exploratory" in cap.lower() else prefix + cap,
                         "caption_from_code": cap})
        out = []
        for i, f in enumerate(made, start=len(s.figures) + 1):
            cited = " ".join(f"{x.strip().rstrip('.')} [A:{analysis_id}]."      # cite inside every sentence
                             for x in SENTENCE.split(f["caption"]) if x.strip())
            r = trace(cited, {analysis_id: {"label": "descriptive"}}, {analysis_id: outputs},
                      [rows] if rows else [])
            out.append({**f, "number": i, "label": label, "caption_check": summarise_checks(r),
                        "warning": f.get("warning")})
        return out

    def omit_figure(self, by: str, analysis_id: str, name: str, reason: str) -> State:
        """Authors leave a figure out of the paper (its information goes in the text). Recorded with the reason;
        the figure stays in the record and can be put back."""
        s = self.state()
        self._require_role(s, by, "author")
        self._require_stage(s, "ANALYSIS", "AUTHOR_REVISION", "RESPONSE")
        if not any(f["analysis_id"] == analysis_id and f["name"] == name and not f.get("omitted") for f in s.figures):
            raise WorkflowError("no such figure in the paper")
        if not (reason or "").strip():
            raise WorkflowError("say why the figure is left out")
        self._emit(by, "figure.omitted", {"analysis_id": analysis_id, "name": name, "reason": reason.strip()})
        return self.state()

    def restore_figure(self, by: str, analysis_id: str, name: str) -> State:
        s = self.state()
        self._require_role(s, by, "author")
        if not any(f["analysis_id"] == analysis_id and f["name"] == name and f.get("omitted") for f in s.figures):
            raise WorkflowError("that figure was not left out")
        self._require_stage(s, "ANALYSIS", "AUTHOR_REVISION", "RESPONSE")
        self._emit(by, "figure.restored", {"analysis_id": analysis_id, "name": name})
        return self.state()

    def withdraw_analysis(self, by: str, analysis_id: str, reason: str) -> State:
        """Leave an exploratory analysis that was asked for after the plan out of the paper. It stays in the record,
        with the reason, and the reviewer is told. Pre-specified analyses cannot be left out this way."""
        s = self.state()
        self._require_role(s, by, "author")
        self._require_stage(s, "ANALYSIS")
        a = s.analyses.get(analysis_id)
        if a is None or a.get("superseded_by") or a.get("withdrawn"):
            raise WorkflowError(f"{analysis_id!r} is not a current analysis")
        if a.get("basis") != "post hoc":
            raise WorkflowError("only an exploratory analysis added after the plan can be left out; a planned analysis "
                                "is reported, or left out by revising the plan before any result")
        if not (reason or "").strip():
            raise WorkflowError("say why it is left out (kept in the record and shown to the reviewer)")
        self._emit(by, "analysis.withdrawn", {"analysis_id": analysis_id, "reason": reason})
        return self.state()

    def describe_analysis(self, by: str, analysis_id: str, text: str, section: str, checks: dict[str, Any],
                          requested_by: str, call: dict[str, Any] | None = None,
                          instruction: str | None = None, attempts: list[dict[str, Any]] | None = None,
                          heading: str = "") -> State:
        """The Writing Agent's text for one analysis (Results, or Methods for the data checks), number-checked."""
        s = self.state()
        a = self._require_role(s, by, "writer")
        self._requested_by(s, a, requested_by)
        if analysis_id not in current(s):
            raise WorkflowError(f"{analysis_id!r} is not a current analysis")
        if section not in ("methods", "results") or not text.strip():
            raise WorkflowError("a description needs text and a section")
        call_seq = self._log_call(by, call)
        self._emit(by, "analysis.described", {"analysis_id": analysis_id, "section": section,
                                               "text_sha256": self.blobs.put_text(text), "checks": checks,
                                               "instruction": instruction, "attempts": attempts or [],
                                               "heading": heading, "requested_by": requested_by,
                                               "llm_call_seq": call_seq})
        return self.state()

    def edit_description(self, by: str, analysis_id: str, text: str, heading: str | None = None) -> State:
        """An author edits the text (and subheading) that reports a result, on the Analyses page. Recorded under
        their name and number-checked like any text; the earlier wording stays in the record."""
        s = self.state()
        self._require_role(s, by, "author")
        self._require_stage(s, "ANALYSIS", "RESPONSE")
        a = current(s).get(analysis_id)
        if a is None:
            raise WorkflowError(f"{analysis_id!r} is not a current analysis")
        if not (text or "").strip():
            raise WorkflowError("the text cannot be empty")
        old = a.get("description") or {}
        heading = (old.get("heading") or "") if heading is None else heading.strip()[:120]
        if old and self.blobs.has(old.get("text_sha256", "")) and self.blobs.get_text(old["text_sha256"]) == text \
                and heading == (old.get("heading") or ""):
            raise WorkflowError("nothing changed")
        out = self.blobs.get_json(a["outputs_sha256"]) | ({"sample_check": a["sample_check"]} if a.get("sample_check") else {})
        rows = []
        if s.curator_reports:
            r = self.blobs.get_json(s.curator_reports[-1]["report_sha256"]).get("profile", {}).get("rows")
            rows = [r] if r else []
        check = trace(text, {analysis_id: {"label": a["label"]}}, {analysis_id: out}, rows, required_refs=[analysis_id])
        section = "methods" if a.get("standard") == "data_quality" else "results"
        self._emit(by, "analysis.described", {"analysis_id": analysis_id, "section": section,
                                               "text_sha256": self.blobs.put_text(text),
                                               "checks": summarise_checks(check), "heading": heading,
                                               "edited_by_author": True, "previous_by": old.get("by"),
                                               "requested_by": None})
        return self.state()

    def record_checklist(self, by: str, version: int, guideline: str, items: list[dict[str, Any]],
                         requested_by: str, call: dict[str, Any] | None = None) -> State:
        """The reporting-guideline check of one draft version, by the Writing Agent; quotations verified by code."""
        s = self.state()
        a = self._require_role(s, by, "writer")
        self._requested_by(s, a, requested_by)
        if version not in s.drafts:
            raise WorkflowError(f"no draft version {version}")
        call_seq = self._log_call(by, call)
        self._emit(by, "checklist.assessed", {"version": version, "guideline": guideline,
                                               "items_sha256": self.blobs.put_json(items), "requested_by": requested_by,
                                               "summary": {k: sum(1 for i in items if i["status"] == k)
                                                           for k in ("reported", "partly", "missing", "not_applicable")},
                                               "llm_call_seq": call_seq})
        return self.state()

    def accept_analyses(self, by: str, analysis_set_sha256: str) -> State:
        s = self.state()
        self._require_role(s, by, "author")
        self._require_stage(s, "ANALYSIS")
        if analysis_set_sha256 != s.analysis_set_sha():
            raise WorkflowError("the analyses have changed since you reviewed them")
        # Only the analyses this paper includes; those the authors left out of the plan are reported as not performed.
        missing = [k for k in included_order(s.plan_locked)
                   if not any(a["plan_analysis_id"] == k and a["status"] == "ok" for a in s.analyses.values())]
        if missing:
            raise WorkflowError("some planned analyses have not run successfully yet: " + ", ".join(missing)
                                + ". Generate the Results again to complete them.")
        self._emit(by, "analyses.accepted", {"analysis_set_sha256": analysis_set_sha256})
        s = self.state()
        if set(s.authors) <= s.analyses_accepted_by.get(analysis_set_sha256, set()):
            self._advance(s, "DRAFTING", "every author accepted the analyses")
        return self.state()

    # -- DRAFTING / AUTHOR_REVISION -----------------------------------------
    def _trace(self, s: State, text: str) -> dict[str, Any]:
        """Check every number in a draft against computed results. Runs on every version."""
        r = self.check_text(text, s)
        return {"checks_sha256": self.blobs.put_json(r), "checks": summarise_checks(r)}

    def check_text(self, text: str, s: State | None = None) -> dict[str, Any]:
        """Full number/reporting check of a text against this project's results. Writes nothing."""
        s = s or self.state()
        ok = {k: v for k, v in current(s).items() if self.blobs.has(v["outputs_sha256"])}
        # A sample-size shortfall's numbers are facts about the analysis, so they can be cited.
        outputs = {k: {**self.blobs.get_json(v["outputs_sha256"]), **({"sample_check": v["sample_check"]}
                                                                      if v.get("sample_check") else {})}
                   for k, v in ok.items()}
        must_state = {k: [r["target"] for r in v["sample_check"]["targets"].values() if r.get("short")]
                      for k, v in ok.items() if (v.get("sample_check") or {}).get("shortfall")}
        must_disclose = sorted(k for k, v in ok.items() if v.get("protocol_deviation"))
        counts = []
        if s.curator_reports:
            rep = self.blobs.get_json(s.curator_reports[-1]["report_sha256"])
            counts.append(rep.get("profile", {}).get("rows"))
        r = trace(text, s.analyses, outputs, [c for c in counts if c is not None], required_refs=sorted(ok),
                  must_state=must_state, must_disclose=must_disclose)
        # Every table and figure must be referred to in the text, and only those that exist.
        tables = [t["number"] for t in tables_mod.build(s, self.blobs)]
        figures = [f["number"] for f in s.figures if f.get("number") and f["analysis_id"] in ok]
        refs = tables_mod.references(CITE.sub("", text))
        r["tables_not_referred"] = sorted(set(tables) - refs["table"])
        r["figures_not_referred"] = sorted(set(figures) - refs["figure"])
        r["references_unknown"] = [f"Table {n}" for n in sorted(refs["table"] - set(tables))] + \
                                  [f"Figure {n}" for n in sorted(refs["figure"] - set(figures))]
        r["ok"] = r["ok"] and not (r["tables_not_referred"] or r["figures_not_referred"] or r["references_unknown"])
        return r

    def create_draft(self, by: str, text: str, sections: list[str], source_analysis_ids: list[str],
                     call: dict[str, Any] | None, requested_by: str | None = None,
                     attempts: list[dict[str, Any]] | None = None) -> State:
        s = self.state()
        a = self._require_role(s, by, "writer")
        self._require_stage(s, "DRAFTING", "AUTHOR_REVISION", "RESPONSE")
        if s.stage != "DRAFTING" and requested_by is None:
            raise WorkflowError("after the first draft, a redraft must be requested by a named author")
        self._requested_by(s, a, requested_by)
        unknown = [i for i in source_analysis_ids if i not in s.analyses]
        if unknown:
            raise WorkflowError(f"draft cites unknown analyses: {', '.join(unknown)}")
        call_seq = self._log_call(by, call)
        self._emit(by, "draft.created", {
            "version": s.draft_version + 1, "base_version": s.draft_version or None,
            "text_sha256": self.blobs.put_text(text), "sections": sections,
            "source_analysis_ids": source_analysis_ids, "requested_by": requested_by, "llm_call_seq": call_seq,
            "writer_attempts": attempts or [],
            **self._trace(s, text),
        })
        if s.stage == "DRAFTING":
            self._advance(self.state(), "AUTHOR_REVISION", "first draft produced")
        return self.state()

    def edit_draft(self, by: str, base_version: int, text: str, summary: str,
                   requested_by: str | None = None, call: dict[str, Any] | None = None,
                   table_edits: dict[str, Any] | None = None) -> State:
        """A new version of the draft: its text, and optionally the wording of its tables (never their numbers)."""
        s = self.state()
        a = self._require_role(s, by, "author", "writer")
        self._require_stage(s, "AUTHOR_REVISION", "RESPONSE")
        if a["kind"] == "agent" and requested_by is None:
            raise WorkflowError("an agent edit must name the author who requested it")
        self._requested_by(s, a, requested_by)
        if base_version != s.draft_version:
            raise WorkflowError(f"draft has moved on to version {s.draft_version}; reload before editing")
        old = self.blobs.get_text(s.drafts[base_version]["text_sha256"])
        te_sha = s.drafts[base_version].get("table_edits_sha256")
        if table_edits is not None:
            try:
                clean = tables_mod.check_edits(tables_mod.build(s, self.blobs), table_edits)
            except ValueError as e:
                raise WorkflowError(str(e))
            te_sha = self.blobs.put_json(clean) if clean else None
        if not (summary or "").strip():
            raise WorkflowError("say what you changed and why")
        if text == old and te_sha == s.drafts[base_version].get("table_edits_sha256"):
            raise WorkflowError("nothing changed")
        call_seq = self._log_call(by, call)
        self._emit(by, "draft.edited", {
            "version": base_version + 1, "base_version": base_version,
            "text_sha256": self.blobs.put_text(text), "summary": summary,
            "diff": _diff_stats(old, text), "requested_by": requested_by, "llm_call_seq": call_seq,
            "table_edits_sha256": te_sha,
            **self._trace(s, text),
        })
        return self.state()

    def open_draft(self, by: str, version: int) -> State:
        s = self.state()
        self._require_role(s, by, "author", "reviewer")
        if version not in s.drafts:
            raise WorkflowError(f"no draft version {version}")
        self._emit(by, "draft.opened", {"version": version})
        return self.state()

    def sign_off(self, by: str, version: int, claim_restatement: str, answers: dict[str, str]) -> State:
        s = self.state()
        self._require_role(s, by, "author")
        self._require_stage(s, "AUTHOR_REVISION", "RESPONSE")
        if version != s.draft_version:
            raise WorkflowError(f"you can only sign off the current version ({s.draft_version})")
        opened = s.opened.get((by, version))
        if opened is None:
            raise WorkflowError("open this version before signing it off")
        if len(claim_restatement.split()) < MIN_RESTATEMENT_WORDS:
            raise WorkflowError(f"restate the primary claim in your own words (at least {MIN_RESTATEMENT_WORDS} words)")
        blank = [q for q in SIGNOFF_QUESTIONS if not answers.get(q, "").strip()]
        if blank:
            raise WorkflowError(f"{len(blank)} verification question(s) unanswered")
        self._emit(by, "draft.signed_off", {
            "version": version, "claim_restatement": claim_restatement,
            "answers": {q: answers[q] for q in SIGNOFF_QUESTIONS},
            "opened_at": opened,
            # What the author accepted: the number check of the exact version signed.
            "checks_at_signoff": s.drafts[version].get("checks"),
        })
        s = self.state()
        if s.stage == "AUTHOR_REVISION" and set(s.authors) <= set(s.signoffs.get(version, {})):
            if not s.reviewers:
                return s  # waits here until a reviewer is registered
            self._advance(s, "REVIEW", "every author signed off the draft", round=1)
        return self.state()

    def start_review(self, by: str) -> State:
        """Used when all authors signed off before any reviewer was registered."""
        s = self.state()
        self._require_role(s, by, "corresponding_author")
        self._require_stage(s, "AUTHOR_REVISION")
        if not s.reviewers:
            raise WorkflowError("register a reviewer first")
        if not set(s.authors) <= set(s.signoffs.get(s.draft_version, {})):
            raise WorkflowError("every author must sign off the current version first")
        self._advance(s, "REVIEW", "every author signed off the draft", round=1)
        return self.state()

    # -- REVIEW ------------------------------------------------------------
    def add_briefing(self, by: str, briefing: dict[str, Any], call: dict[str, Any] | None,
                     requested_by: str | None = None) -> State:
        """The Reviewer Assistant's briefing. It works for the reviewer, so only a reviewer may ask for one."""
        s = self.state()
        self._require_role(s, by, "reviewer_assist")
        self._require_stage(s, "REVIEW")
        if requested_by is None:
            raise WorkflowError("a briefing must name the reviewer who asked for it")
        self._require_role(s, requested_by, "reviewer")
        if briefing.get("draft_version") != s.draft_version:
            raise WorkflowError(f"briefing is about version {briefing.get('draft_version')}; "
                                f"the draft under review is version {s.draft_version}")
        call_seq = self._log_call(by, call)
        self._emit(by, "review.briefing", {"round": s.round, "draft_version": s.draft_version,
                                           "briefing_sha256": self.blobs.put_json(briefing),
                                           "inputs": briefing.get("inputs", {}),
                                           "points_of_inquiry": briefing.get("points_of_inquiry", []),
                                           "item_assessments": briefing.get("item_assessments", []),
                                           "requested_by": requested_by, "llm_call_seq": call_seq})
        return self.state()

    def add_item(self, by: str, item_id: str, comment: str, location: str | None = None,
                 assisted_by_seq: int | None = None, from_point: str | None = None) -> State:
        """A review comment is always the reviewer's own. If it adopts a point from a briefing,
        say which one, so the record shows what the assistant contributed."""
        s = self.state()
        self._require_role(s, by, "reviewer")
        self._require_stage(s, "REVIEW")
        if item_id in s.items:
            raise WorkflowError(f"item {item_id} already exists")
        if not comment.strip():
            raise WorkflowError("comment is empty")
        if from_point is not None and assisted_by_seq is None:
            raise WorkflowError("from_point needs assisted_by_seq (the briefing it comes from)")
        if assisted_by_seq is not None:
            b = next((b for b in s.briefings if b["seq"] == assisted_by_seq), None)
            if b is None:
                raise WorkflowError(f"event {assisted_by_seq} is not a review briefing")
            if from_point is not None and from_point not in {pt.get("id") for pt in b["points_of_inquiry"]}:
                raise WorkflowError(f"briefing {assisted_by_seq} has no point {from_point}")
        self._emit(by, "review.item_added", {"item_id": item_id, "round": s.round, "comment": comment,
                                             "location": location, "assisted_by_seq": assisted_by_seq,
                                             "from_point": from_point})
        return self.state()

    def give_verdict(self, by: str, item_id: str, verdict: str, note: str = "") -> State:
        s = self.state()
        self._require_role(s, by, "reviewer")
        self._require_stage(s, "REVIEW")
        it = s.items.get(item_id)
        if it is None:
            raise WorkflowError(f"no item {item_id}")
        if it["status"] != "responded":
            raise WorkflowError(f"item {item_id} is {it['status']}; a verdict needs an author response first")
        if verdict not in VERDICTS:
            raise WorkflowError(f"verdict must be one of {sorted(VERDICTS)}")
        if verdict == "agree_to_disagree" and not note.strip():
            raise WorkflowError("say what remains in disagreement; it will appear in the Reviewer's Commentary")
        self._emit(by, "review.item_verdict", {"item_id": item_id, "verdict": verdict, "note": note})
        return self.state()

    def submit_review(self, by: str, decision: str, summary: str = "") -> State:
        s = self.state()
        self._require_role(s, by, "reviewer")
        self._require_stage(s, "REVIEW")
        pending = sorted(i for i, v in s.items.items() if v["status"] == "responded")
        if pending:
            raise WorkflowError(f"give a verdict on every author response first: {', '.join(pending)}")
        if decision == "accept":
            unsettled = s.open_items()
            if unsettled:
                raise WorkflowError(f"cannot accept while items are open: {', '.join(unsettled)}")
            self._emit(by, "review.submitted", {"round": s.round, "decision": "accept", "summary": summary,
                                                "draft_version": s.draft_version})
            self._advance(self.state(), "ATTESTATION", "reviewer accepted; every item settled")
        elif decision == "revise":
            if not s.open_items():
                raise WorkflowError("'revise' needs at least one open item")
            if s.round >= s.max_rounds:
                raise WorkflowError(f"round limit ({s.max_rounds}) reached: accept, or settle remaining items "
                                    "as agree-to-disagree so they are published in the commentary")
            self._emit(by, "review.submitted", {"round": s.round, "decision": "revise", "summary": summary,
                                                "draft_version": s.draft_version})
            self._advance(self.state(), "RESPONSE", "reviewer requested revision")
        else:
            raise WorkflowError("decision must be 'accept' or 'revise'")
        return self.state()

    # -- RESPONSE ----------------------------------------------------------
    def respond(self, by: str, item_id: str, response: str, note: str = "") -> State:
        s = self.state()
        self._require_role(s, by, "author")
        self._require_stage(s, "RESPONSE")
        it = s.items.get(item_id)
        if it is None:
            raise WorkflowError(f"no item {item_id}")
        if it["status"] != "open":
            raise WorkflowError(f"item {item_id} is {it['status']}, not open")
        if response not in RESPONSES:
            raise WorkflowError(f"response must be one of {sorted(RESPONSES)}")
        last_seen = max([it["draft_version_at_creation"]] + [r["draft_version"] for r in it["responses"]])
        if response == "revised" and s.draft_version <= last_seen:
            raise WorkflowError("'revised' needs a new draft version; edit the draft first")
        if response in {"clarified", "rebutted"} and not note.strip():
            raise WorkflowError("explain your clarification or rebuttal")
        self._emit(by, "review.item_responded", {"item_id": item_id, "response": response, "note": note,
                                                 "draft_version": s.draft_version})
        return self.state()

    def submit_response(self, by: str) -> State:
        s = self.state()
        self._require_role(s, by, "corresponding_author")
        self._require_stage(s, "RESPONSE")
        if s.open_items():
            raise WorkflowError(f"respond to every open item first: {', '.join(s.open_items())}")
        unsigned = sorted(set(s.authors) - set(s.signoffs.get(s.draft_version, {})))
        if unsigned:
            raise WorkflowError(f"every author must sign off version {s.draft_version}: {', '.join(unsigned)}")
        self._emit(by, "response.submitted", {"round": s.round, "draft_version": s.draft_version})
        self._advance(self.state(), "REVIEW", "authors responded to every item", round=s.round + 1)
        return self.state()

    # -- ATTESTATION -------------------------------------------------------
    def attest(self, by: str, version: int, statement: str) -> State:
        s = self.state()
        self._require_role(s, by, "author")
        self._require_stage(s, "ATTESTATION")
        if version != s.draft_version:
            raise WorkflowError(f"attest the final version ({s.draft_version})")
        if len(statement.split()) < 5:
            raise WorkflowError("write a short attestation statement in your own words")
        self._emit(by, "authorship.attested", {"version": version, "statement": statement,
                                               "contributions_ledger_sha256": sha256_obj(ledger(self.log.events()))})
        s = self.state()
        if set(s.authors) <= set(s.attestations.get(version, {})):
            self._advance(s, "FINAL", "every author attested the final version",
                          final_version=version, log_head=self.log.head()[1])
        return self.state()


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def _diff_stats(old: str, new: str) -> dict[str, int]:
    import difflib
    sm = difflib.SequenceMatcher(a=old.split(), b=new.split(), autojunk=False)
    added = removed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return {"words_added": added, "words_removed": removed, "words_before": len(old.split()),
            "words_after": len(new.split())}


def preparation_in_use(s: State) -> dict[str, Any] | None:
    """How the dataset in use was prepared from the original upload, and by whom; None if it is the original."""
    ds = next((m for m in reversed(s.materials) if m["kind"] == "dataset" and not m.get("withdrawn")), None)
    if not ds or not ds.get("preparation"):
        return None
    who = (s.actors.get(ds["by"]) or {}).get("name") or ds["by"]
    return {**ds["preparation"], "applied_by": who, "dataset": ds["filename"]}


def open_questions(s: State) -> list[str]:
    """Saksi's questions about this paper: the plan's own, then what each included analysis left unresolved
    (the same wording the page shows)."""
    p = s.plan or {}
    qs = list(((p.get("report") or {}).get("questions")) or []) + list((s.outline or {}).get("open_items") or [])
    for i in p.get("order") or []:
        a = p["analyses"][i]
        if a.get("included", True) is not False:
            qs += [f"{a.get('title') or i}: {u}" for u in a.get("unresolved") or []]
    return qs


def answers_text(s: State) -> str:
    """The authors' answers to Saksi's questions, for the agents."""
    rows = [f"- Q: {a['question']}\n  A ({(s.actors.get(a['by']) or {}).get('name') or a['by']}): {a['answer']}" for a in s.answers]
    return "\n".join(rows) or "(none yet)"


def current(s: State) -> dict[str, dict[str, Any]]:
    """The analyses that count: those that ran, minus any an author had redone (kept in the record)."""
    return {k: v for k, v in s.analyses.items()
            if v["status"] == "ok" and v.get("outputs_sha256") and not v.get("superseded_by") and not v.get("withdrawn")}


def _number_figures(s: State) -> None:
    """Figures are numbered in reporting order; a replaced analysis's figures lose their number."""
    live = [f for f in s.figures if not s.analyses.get(f["analysis_id"], {}).get("superseded_by")
            and not s.analyses.get(f["analysis_id"], {}).get("withdrawn") and not f.get("omitted")]
    live.sort(key=lambda f: (position(s.analyses[f["analysis_id"]]), f["seq"]))
    for f in s.figures:
        f["number"] = None
    for i, f in enumerate(live, start=1):
        f["number"] = i


def included_order(plan: dict[str, Any] | None) -> list[str]:
    """The plan's analyses that this paper includes, in protocol order."""
    if not plan:
        return []
    return [i for i in plan["order"] if plan["analyses"][i].get("included", True)]


def _plan_changes(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    o = old["analyses"]
    inc = lambda d, i: d[i].get("included", True)
    return {"added": sorted(set(new) - set(o)), "removed": sorted(set(o) - set(new)),
            "left_out": sorted(i for i in new if not inc(new, i) and (i not in o or inc(o, i))),
            "put_back": sorted(i for i in new if inc(new, i) and i in o and not inc(o, i)),
            "role_changed": sorted(i for i in new if i in o and new[i].get("role") != o[i].get("role")),
            "label_changed": sorted(i for i in new if i in o and new[i].get("label") != o[i].get("label"))}


def summarise_checks(r: dict[str, Any]) -> dict[str, Any]:
    return {"numbers_checked": r["numbers_checked"], "untraced": len(r["untraced"]),
            "exploratory_not_labelled": len(r["exploratory_not_labelled"]),
            "unknown_analysis_refs": r["unknown_analysis_refs"],
            "analyses_not_reported": r.get("analyses_not_reported", []),
            "nothing_checked_but_numbers_present": r.get("nothing_checked_but_numbers_present", False),
            "shortfall_not_reported": r.get("shortfall_not_reported", []),
            "departure_not_disclosed": r.get("departure_not_disclosed", []),
            "tables_not_referred": r.get("tables_not_referred", []),
            "figures_not_referred": r.get("figures_not_referred", []),
            "references_unknown": r.get("references_unknown", []),
            "ok": r["ok"]}


# What each event type counts as in the Contribution Ledger (white paper §3.1).
LEDGER_CATEGORIES = {
    "material.uploaded": "materials", "material.withdrawn": "materials_removed", "plan.proposed": "plan_proposals", "plan.confirmed": "verification_decisions",
    "analysis.requested": "analysis_requests", "analysis.completed": "analyses_run", "analysis.described": "result_texts", "analysis.withdrawn": "analyses_left_out", "figure.omitted": "edits", "figure.restored": "edits", "checklist.assessed": "checklists", "question.answered": "clarifications", "questions.suggested": "suggested_answers", "plan.outlined": "plan_outlines",
    "plan.outline_edited": "edits",
    "data.preparation_proposed": "data_preparation_proposals",
    "analyses.accepted": "verification_decisions", "draft.created": "drafts", "draft.edited": "edits",
    "draft.signed_off": "verification_decisions", "review.briefing": "review_briefings",
    "review.item_added": "review_comments", "review.item_verdict": "review_verdicts",
    "review.submitted": "review_decisions", "review.item_responded": "responses_to_review",
    "authorship.attested": "attestations", "curator.reported": "curation_reports", "llm.called": "model_calls",
}


def ledger(events: list[Event]) -> dict[str, Any]:
    """Who did what, from the log alone. Descriptive only: no ranking."""
    out: dict[str, dict[str, Any]] = {}
    times: dict[tuple[str, int], str] = {}
    for ev in events:
        if ev.type == "draft.opened":
            times.setdefault((ev.actor_id, ev.payload["version"]), ev.ts)
        cat = LEDGER_CATEGORIES.get(ev.type)
        if cat is None:
            continue
        row = out.setdefault(ev.actor_id, {"counts": {}, "words_added": 0, "words_removed": 0,
                                           "verification_seconds": 0, "tokens": 0, "on_behalf_of": {}})
        row["counts"][cat] = row["counts"].get(cat, 0) + 1
        p = ev.payload
        if ev.type == "draft.edited":
            row["words_added"] += p["diff"]["words_added"]
            row["words_removed"] += p["diff"]["words_removed"]
        if ev.type == "draft.signed_off":
            start = datetime.fromisoformat(p["opened_at"])
            row["verification_seconds"] += int((datetime.fromisoformat(ev.ts) - start).total_seconds())
        if ev.type == "llm.called":
            row["tokens"] += (p.get("prompt_tokens") or 0) + (p.get("completion_tokens") or 0)
        if ev.type == "review.item_added" and p.get("from_point"):
            row["counts"]["review_comments_from_assistant"] = row["counts"].get("review_comments_from_assistant", 0) + 1
        if p.get("requested_by"):
            row["on_behalf_of"][p["requested_by"]] = row["on_behalf_of"].get(p["requested_by"], 0) + 1
    return out
