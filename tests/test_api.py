# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

from fastapi.testclient import TestClient

from saksi.api import create_app
from tests.conftest import ANSWERS, AUTHOR_A, AUTHOR_B, RESTATE


def setup(tmp_path):
    c = TestClient(create_app(tmp_path))
    r = c.post("/projects", json={"project_id": "gdm-demo", "title": "Demo", "study_type": "primary",
                                  "creator": AUTHOR_A})
    assert r.status_code == 200, r.text
    return c, r.json()["token"]


def cmd(c, token, name, **args):
    return c.post(f"/projects/gdm-demo/commands/{name}", json={"args": args}, headers={"X-Saksi-Token": token})


def test_token_decides_identity(tmp_path):
    c, tok_a = setup(tmp_path)
    r = cmd(c, tok_a, "register_human", actor=AUTHOR_B, role="author")
    assert r.status_code == 200
    tok_b = r.json()["token_for_new_actor"]
    assert c.get("/projects/gdm-demo", headers={"X-Saksi-Token": tok_b}).json()["you"] == "coauthor"
    # The co-author cannot register people: only the corresponding author can.
    r = cmd(c, tok_b, "register_human", actor={"actor_id": "x", "name": "X"}, role="author")
    assert r.status_code == 409


def test_no_token_no_access(tmp_path):
    c, _ = setup(tmp_path)
    assert c.get("/projects/gdm-demo").status_code == 401
    assert c.get("/projects/gdm-demo", headers={"X-Saksi-Token": "sk_saksi_wrong"}).status_code == 401
    assert c.get("/projects/gdm-demo/verify").json()["ok"] is True   # anyone may check integrity


def test_body_cannot_pick_actor(tmp_path):
    c, tok = setup(tmp_path)
    r = c.post("/projects/gdm-demo/commands/open_draft", json={"args": {"version": 1, "by": "someone"}},
               headers={"X-Saksi-Token": tok})
    assert r.status_code == 422 and "unexpected" in r.text


def test_upload_is_hashed_and_private(tmp_path):
    c, tok = setup(tmp_path)
    r = c.post("/projects/gdm-demo/materials", data={"kind": "dataset"},
               files={"file": ("../../etc/data.csv", b"id,bmi\n1,24\n")}, headers={"X-Saksi-Token": tok})
    assert r.status_code == 200, r.text
    m = r.json()["state"]["materials"][0]
    assert m["filename"] == "data.csv" and m["bytes"] == 12
    assert (tmp_path / "gdm-demo" / "materials" / m["sha256"]).exists()
    assert not (tmp_path / "blobs" / m["sha256"][:2] / m["sha256"]).exists()


def test_refusal_is_409_with_reason(tmp_path):
    c, tok = setup(tmp_path)
    r = cmd(c, tok, "sign_off", version=1, claim_restatement=RESTATE, answers=ANSWERS)
    assert r.status_code == 409 and "not allowed in stage INGEST" in r.json()["detail"]


def test_bad_project_id(tmp_path):
    c = TestClient(create_app(tmp_path))
    r = c.post("/projects", json={"project_id": "../x", "title": "t", "study_type": "primary", "creator": AUTHOR_A})
    assert r.status_code == 400


def test_creation_token_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("SAKSI_CREATE_TOKEN", "letmein-long-random")
    c = TestClient(create_app(tmp_path))
    body = {"project_id": "gated", "title": "t", "study_type": "primary", "creator": AUTHOR_A}
    assert c.post("/projects", json=body).status_code == 403
    assert c.post("/projects", json=body, headers={"X-Saksi-Create-Token": "wrong"}).status_code == 403
    assert c.post("/projects", json=body, headers={"X-Saksi-Create-Token": "letmein-long-random"}).status_code == 200


def test_source_offered_to_users(tmp_path):
    c = TestClient(create_app(tmp_path))
    assert c.get("/source").json() == {"source": "https://github.com/boonhowchew/saksi", "licence": "AGPL-3.0-only"}
    assert c.get("/health").json()["source"].endswith("/saksi")


def test_agents_over_http(tmp_path, monkeypatch):
    from saksi import demo_data
    from saksi.agents import Agents
    from saksi.llm import ModelClient
    from saksi.sandbox import LocalExecutor
    from tests.test_agents import CURATOR_REPLY, GOOD, PLAN_REPLY, ScriptedModel, fence

    monkeypatch.setenv("META_API_KEY", "test-key")
    model = ScriptedModel({"curate": [CURATOR_REPLY], "plan": [{"analyses": [PLAN_REPLY["analyses"][0]]}],
                           "code": [fence(GOOD)]})
    app = create_app(tmp_path, agents_factory=lambda p: Agents(
        p, ModelClient(p.blobs, transport=model, mode="live"), LocalExecutor()))
    c = TestClient(app)
    tok = c.post("/projects", json={"project_id": "gdm-demo", "title": "Demo", "study_type": "primary",
                                    "creator": AUTHOR_A}).json()["token"]
    h = {"X-Saksi-Token": tok}
    rev = cmd(c, tok, "register_human", actor={"actor_id": "rev1", "name": "R"}, role="reviewer")
    rev_h = {"X-Saksi-Token": rev.json()["token_for_new_actor"]}
    c.post("/projects/gdm-demo/materials", data={"kind": "dataset"}, headers=h,
           files={"file": ("gdm.csv", demo_data.make().to_csv(index=False).encode())})
    c.post("/projects/gdm-demo/materials", data={"kind": "protocol"}, headers=h,
           files={"file": ("protocol.txt", demo_data.PROTOCOL.encode())})

    assert c.post("/projects/gdm-demo/agents/curate", json={"args": {}}, headers=rev_h).status_code == 403
    assert c.post("/projects/gdm-demo/agents/curate", json={"args": {}}, headers=h).status_code == 200
    st = c.post("/projects/gdm-demo/agents/propose_plan", json={"args": {}}, headers=h).json()["state"]
    assert st["plan"]["analyses"]["primary"]["protocol_quote_verified"] is True
    assert cmd(c, tok, "confirm_plan", plan_sha256=st["plan"]["sha"]).json()["state"]["stage"] == "ANALYSIS"
    st = c.post("/projects/gdm-demo/agents/run_planned", json={"args": {}}, headers=h).json()["state"]
    assert st["analyses"]["primary"]["status"] == "ok"
    # Running again before the plan changes does nothing new.
    r = c.post("/projects/gdm-demo/agents/run_planned", json={"args": {}}, headers=h)
    assert r.status_code == 200 and list(r.json()["state"]["analyses"]) == ["primary"]
    assert c.post("/projects/gdm-demo/agents/nope", json={"args": {}}, headers=h).status_code == 404
    assert c.get("/projects/gdm-demo/verify").json()["ok"] is True
