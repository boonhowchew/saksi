# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""HTTP API for the Saksi workflow.

Identity: each human actor receives a personal access token when registered.
The token is shown once and stored only as a SHA-256 hash. Every request is
attributed to the actor who owns the token — the request body cannot choose
who it acts as. (ORCID sign-in is on the roadmap; see README.)

Agent actions are not exposed here as raw commands: agents run server-side
and write their own events.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .canonical import sha256_bytes, sha256_text
from .inspection import inspect_material
from .workflow import Project, WorkflowError, ledger

log = logging.getLogger("saksi")

# Agent actions: name -> (Agents method, allowed argument names, role that may ask)
# The authors' agents work for the authors; the Reviewer Assistant works for the reviewer.
AGENT_ACTIONS: dict[str, tuple[str, set[str], str]] = {
    "curate": ("curate", {"material_sha256"}, "author"),
    "propose_plan": ("propose_plan", {"instructions"}, "author"),
    "prepare_plan": ("prepare_plan", {"instructions"}, "author"),
    "run_planned": ("run_planned", {"plan_analysis_id"}, "author"),
    "run_request": ("run_request", {"request_id", "prompt"}, "author"),
    "write_draft": ("write_draft", set(), "author"),
    "brief_reviewer": ("brief_reviewer", set(), "reviewer"),
}

PROJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")
# The web page. On Modal it is copied to SAKSI_WEB_DIR (Modal ships only .py files with the package).
WEB_DIR = Path(os.environ.get("SAKSI_WEB_DIR") or Path(__file__).parent / "static")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Human commands callable over HTTP: name -> (Project method, allowed argument names)
HUMAN_COMMANDS: dict[str, tuple[str, set[str]]] = {
    "register_human": ("register_human", {"actor", "role"}),
    "withdraw_material": ("withdraw_material", {"sha256", "reason"}),
    "confirm_plan": ("confirm_plan", {"plan_sha256"}),
    "propose_plan": ("propose_plan", {"analyses", "protocol_ref", "note"}),
    "request_analysis": ("request_analysis", {"request_id", "prompt"}),
    "accept_analyses": ("accept_analyses", {"analysis_set_sha256"}),
    "edit_draft": ("edit_draft", {"base_version", "text", "summary"}),
    "open_draft": ("open_draft", {"version"}),
    "sign_off": ("sign_off", {"version", "claim_restatement", "answers"}),
    "start_review": ("start_review", set()),
    "add_item": ("add_item", {"item_id", "comment", "location", "assisted_by_seq", "from_point"}),
    "give_verdict": ("give_verdict", {"item_id", "verdict", "note"}),
    "submit_review": ("submit_review", {"decision", "summary"}),
    "respond": ("respond", {"item_id", "response", "note"}),
    "submit_response": ("submit_response", set()),
    "attest": ("attest", {"version", "statement"}),
}


# The page talks only to this server: no third-party scripts, fonts or images.
CSP = ("default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
       "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


class TokenStore:
    """actor tokens per project, kept outside the event log (they are secrets)."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, pid: str) -> Path:
        return self.root / pid / "tokens.json"

    def _load(self, pid: str) -> dict[str, str]:
        p = self._path(pid)
        return json.loads(p.read_text()) if p.exists() else {}

    def issue(self, pid: str, actor_id: str) -> str:
        token = "sk_saksi_" + secrets.token_urlsafe(24)
        data = self._load(pid)
        data[sha256_text(token)] = actor_id
        p = self._path(pid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
        p.chmod(0o600)
        return token

    def find(self, token: str) -> tuple[str, str] | None:
        """Which project and person a token belongs to, so signing in needs only the token."""
        h = sha256_text(token.strip())
        for f in sorted(self.root.glob("*/tokens.json")):
            actor = json.loads(f.read_text()).get(h)
            if actor:
                return f.parent.name, actor
        return None

    def resolve(self, pid: str, token: str | None) -> str:
        if not token:
            raise HTTPException(401, "missing X-Saksi-Token header")
        actor = self._load(pid).get(sha256_text(token))
        if not actor:
            raise HTTPException(401, "token not recognised for this project")
        return actor


class CreateProject(BaseModel):
    title: str
    creator: dict[str, Any]
    project_id: str | None = None         # made from the title when not given
    study_type: str = "primary"
    max_rounds: int = 5


class SignIn(BaseModel):
    token: str


def slug(text: str, n: int = 24) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:n].strip("-")
    return s or "x"


def new_actor_id(name: str, taken: set[str]) -> str:
    base = slug(name, 20)
    base = base if not base.startswith(("agent", "system")) else f"p-{base}"
    aid, n = base, 2
    while aid in taken:
        aid, n = f"{base}-{n}", n + 1
    return aid


class Command(BaseModel):
    args: dict[str, Any] = {}


def create_app(data_root: str | Path, on_write: Callable[[], None] | None = None,
               allowed_origins: list[str] | None = None,
               agents_factory: Callable[[Project], Any] | None = None) -> FastAPI:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    tokens = TokenStore(root)
    persist = on_write or (lambda: None)

    app = FastAPI(title="Saksi", version="0.1.0",
                  description="Research that is witnessed, not reconstructed.")
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins or ["*"],
                       allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-Saksi-Token", "X-Saksi-Create-Token"])

    def project(pid: str) -> Project:
        if not PROJECT_ID_RE.match(pid):
            raise HTTPException(400, "project id: 3–41 lowercase letters, digits or hyphens")
        return Project(root, pid)

    def existing(pid: str) -> Project:
        p = project(pid)
        if not p.log.events():
            raise HTTPException(404, f"no project {pid}")
        return p

    def actor(pid: str, x_saksi_token: str | None = Header(default=None)) -> str:
        return tokens.resolve(pid, x_saksi_token)

    def refused(e: WorkflowError):
        return HTTPException(409, str(e))

    source_url = os.environ.get("SAKSI_SOURCE_URL", "https://github.com/boonhowchew/saksi")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page():
        f = WEB_DIR / "index.html"
        if not f.exists():
            raise HTTPException(404, f"web page not found at {f}")
        return HTMLResponse(f.read_text(encoding="utf-8"),
                            headers={"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff",
                                     "Referrer-Policy": "no-referrer"})

    @app.get("/health")
    def health():
        return {"ok": True, "source": source_url}

    @app.get("/source")
    def source():
        """AGPL-3.0 section 13: users of the service can get its source."""
        return {"source": source_url, "licence": "AGPL-3.0-only"}

    create_token = os.environ.get("SAKSI_CREATE_TOKEN")

    @app.post("/projects")
    def create_project(body: CreateProject, x_saksi_create_token: str | None = Header(default=None)):
        # On a public deployment, only someone holding the creation token may open projects.
        if create_token and not secrets.compare_digest(x_saksi_create_token or "", create_token):
            raise HTTPException(403, "project creation needs a valid X-Saksi-Create-Token")
        pid = body.project_id or f"{slug(body.title)}-{secrets.token_hex(2)}"
        creator = {**body.creator, "actor_id": body.creator.get("actor_id") or new_actor_id(body.creator.get("name", ""), set())}
        p = project(pid)
        try:
            s = p.create(body.title, body.study_type, creator, max_rounds=body.max_rounds)
        except WorkflowError as e:
            raise refused(e)
        token = tokens.issue(pid, creator["actor_id"])
        persist()
        return {"state": s.to_dict(), "token": token, "project_id": pid,
                "note": "Keep this token; it is shown only once."}

    @app.post("/signin")
    def signin(body: SignIn):
        """Find the project a personal token belongs to."""
        hit = tokens.find(body.token)
        if not hit:
            raise HTTPException(401, "That token is not recognised. Copy it in full: it starts with sk_saksi_.")
        return {"project_id": hit[0], "actor_id": hit[1]}

    @app.get("/projects/{pid}")
    def get_state(pid: str, who: str = Depends(actor)):
        return {"you": who, "state": existing(pid).state().to_dict()}

    @app.get("/projects/{pid}/events")
    def get_events(pid: str, who: str = Depends(actor)):
        return [e.__dict__ for e in existing(pid).log.events()]

    @app.get("/projects/{pid}/verify")
    def verify(pid: str):
        r = existing(pid).log.verify()
        return r.__dict__

    @app.get("/projects/{pid}/ledger")
    def get_ledger(pid: str, who: str = Depends(actor)):
        return ledger(existing(pid).log.events())

    @app.get("/projects/{pid}/blobs/{digest}")
    def get_blob(pid: str, digest: str, who: str = Depends(actor)):
        p = existing(pid)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or not p.blobs.has(digest):
            raise HTTPException(404, "no such content")
        return {"sha256": digest, "text": p.blobs.get_text(digest)}

    @app.post("/projects/{pid}/commands/{name}")
    def command(pid: str, name: str, body: Command, who: str = Depends(actor)):
        if name not in HUMAN_COMMANDS:
            raise HTTPException(404, f"unknown command {name}")
        method, allowed = HUMAN_COMMANDS[name]
        extra = set(body.args) - allowed
        if extra:
            raise HTTPException(422, f"unexpected arguments: {', '.join(sorted(extra))}")
        p = existing(pid)
        if name == "register_human" and isinstance(body.args.get("actor"), dict) and not body.args["actor"].get("actor_id"):
            body.args["actor"]["actor_id"] = new_actor_id(body.args["actor"].get("name", ""), set(p.state().actors))
        try:
            s = getattr(p, method)(who, **body.args)
        except WorkflowError as e:
            raise refused(e)
        except TypeError as e:
            raise HTTPException(422, f"bad arguments: {e}")
        out: dict[str, Any] = {"state": s.to_dict()}
        if name == "register_human":
            out["token_for_new_actor"] = tokens.issue(pid, body.args["actor"]["actor_id"])
            out["note"] = "Send this token privately to the person you registered; it is shown only once."
        persist()
        return out

    @app.post("/projects/{pid}/agents/{action}")
    def agent(pid: str, action: str, body: Command, who: str = Depends(actor)):
        """Ask an agent to act. The request is attributed to the person who sent it."""
        from .agents import AgentError
        from .llm import LLMError
        if agents_factory is None:
            raise HTTPException(503, "agents are not configured on this server")
        if action not in AGENT_ACTIONS:
            raise HTTPException(404, f"unknown agent action {action}")
        method, allowed, role = AGENT_ACTIONS[action]
        extra = set(body.args) - allowed
        if extra:
            raise HTTPException(422, f"unexpected arguments: {', '.join(sorted(extra))}")
        p = existing(pid)
        if role not in p.state().actors[who]["roles"]:
            raise HTTPException(403, "only authors can ask the authors' agents to act" if role == "author"
                                else "only a reviewer can ask the Reviewer Assistant")
        try:
            s = getattr(agents_factory(p), method)(who, **body.args)
        except WorkflowError as e:
            raise refused(e)
        except AgentError as e:
            persist()
            raise HTTPException(422, f"the agent could not produce a usable result: {e}")
        except LLMError as e:
            persist()
            raise HTTPException(502, f"model service error: {e}")
        except Exception as e:  # a bug, not a refusal: say which action failed, keep details in the server log
            log.exception("agent action %s failed", action)
            persist()
            raise HTTPException(500, f"Saksi hit an unexpected error during '{action}' ({type(e).__name__}). "
                                     "Anything already done is in the record; the details are in the server log.")
        persist()
        return {"state": s.to_dict()}

    @app.post("/projects/{pid}/materials")
    async def upload(pid: str, kind: str = Form(...), note: str | None = Form(None),
                     file: UploadFile = File(...), who: str = Depends(actor)):
        p = existing(pid)
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file larger than 200 MB")
        digest = sha256_bytes(data)
        # Study materials live in a private per-project folder, never in the shared blob store.
        dest = p.root / "materials" / digest
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        fname = Path(file.filename or "upload").name
        try:
            s = p.upload_material(who, kind, fname, digest, len(data), storage_ref=f"materials/{digest}", note=note,
                                  inspection=inspect_material(kind, fname, data))
        except WorkflowError as e:
            raise refused(e)
        persist()
        return {"state": s.to_dict(), "sha256": digest}

    return app
