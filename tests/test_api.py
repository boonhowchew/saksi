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


def test_review_loop_over_http(tmp_path, monkeypatch):
    """Draft to FINAL over the API, two review rounds, with the Reviewer Assistant."""
    from saksi import demo_data
    from saksi.agents import Agents
    from saksi.llm import ModelClient
    from saksi.sandbox import LocalExecutor
    from tests.test_agents import CURATOR_REPLY, GOOD, PLAN_REPLY, ScriptedModel, fence

    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("BASETEN_API_KEY", "test-key-2")
    draft = ("## Methods\nLogistic regression as planned.\n\n## Results\n"
             "Booking BMI was associated with GDM [A:primary]. 388 women were analysed against 400 planned, "
             "with 18 GDM events against 40 planned [A:primary].\n\n## Discussion\nThe sample was small.\n")
    model = ScriptedModel({
        "curate": [CURATOR_REPLY], "plan": [{"analyses": [PLAN_REPLY["analyses"][0]]}], "code": [fence(GOOD)],
        "write": [draft],
        "review": [{"primary_claim": "BMI and GDM.", "summary": "s",
                    "points_of_inquiry": [{"category": "limitations", "severity": "major",
                                           "quote": "The sample was small", "question": "How small?",
                                           "why_it_matters": "Precision."}]},
                   {"primary_claim": "BMI and GDM.", "summary": "s", "points_of_inquiry": [],
                    "item_assessments": [{"item_id": "R1", "assessment": "addressed",
                                          "quote": "388 women were analysed", "note": "ok"}]}]})
    c = TestClient(create_app(tmp_path, agents_factory=lambda p: Agents(
        p, ModelClient(p.blobs, transport=model, mode="live"), LocalExecutor())))
    tok = c.post("/projects", json={"project_id": "gdm-demo", "title": "Demo", "study_type": "primary",
                                    "creator": AUTHOR_A}).json()["token"]
    h = {"X-Saksi-Token": tok}
    rev = cmd(c, tok, "register_human", actor={"actor_id": "rev1", "name": "R"}, role="reviewer")
    rtok = rev.json()["token_for_new_actor"]
    rh = {"X-Saksi-Token": rtok}
    for kind, name, data in (("dataset", "gdm.csv", demo_data.make().to_csv(index=False).encode()),
                             ("protocol", "protocol.txt", demo_data.PROTOCOL.encode())):
        c.post("/projects/gdm-demo/materials", data={"kind": kind}, headers=h, files={"file": (name, data)})

    def agent(action, headers):
        r = c.post(f"/projects/gdm-demo/agents/{action}", json={"args": {}}, headers=headers)
        assert r.status_code == 200, r.text
        return r.json()["state"]

    agent("curate", h)
    st = agent("propose_plan", h)
    cmd(c, tok, "confirm_plan", plan_sha256=st["plan"]["sha"])
    st = agent("run_planned", h)
    cmd(c, tok, "accept_analyses", analysis_set_sha256=st["analysis_set_sha256"])
    st = agent("write_draft", h)
    assert st["stage"] == "AUTHOR_REVISION"
    cmd(c, tok, "open_draft", version=1)
    st = cmd(c, tok, "sign_off", version=1, claim_restatement=RESTATE, answers=ANSWERS).json()["state"]
    assert st["stage"] == "REVIEW"

    # Only the reviewer can ask the Reviewer Assistant; only authors can ask the authors' agents.
    assert c.post("/projects/gdm-demo/agents/brief_reviewer", json={"args": {}}, headers=h).status_code == 403
    assert c.post("/projects/gdm-demo/agents/write_draft", json={"args": {}}, headers=rh).status_code == 403
    st = agent("brief_reviewer", rh)
    b = st["briefings"][-1]
    assert b["points_of_inquiry"][0]["quote_verified"] is True
    assert cmd(c, rtok, "add_item", item_id="R1", comment="State the number analysed.",
               assisted_by_seq=b["seq"], from_point="P1").status_code == 200
    assert cmd(c, rtok, "submit_review", decision="revise").json()["state"]["stage"] == "RESPONSE"

    cmd(c, tok, "edit_draft", base_version=1, summary="R1",
        text=draft.replace("The sample was small.", "The sample was small: 388 women were analysed [A:primary]."))
    cmd(c, tok, "respond", item_id="R1", response="revised")
    cmd(c, tok, "open_draft", version=2)
    cmd(c, tok, "sign_off", version=2, claim_restatement=RESTATE, answers=ANSWERS)
    st = cmd(c, tok, "submit_response").json()["state"]
    assert st["stage"] == "REVIEW" and st["round"] == 2

    st = agent("brief_reviewer", rh)
    assert st["briefings"][-1]["item_assessments"][0]["assessment"] == "addressed"
    cmd(c, rtok, "give_verdict", item_id="R1", verdict="satisfied")
    assert cmd(c, rtok, "submit_review", decision="accept").json()["state"]["stage"] == "ATTESTATION"
    st = cmd(c, tok, "attest", version=2, statement="I stand behind this final version.").json()["state"]
    assert st["stage"] == "FINAL"
    assert c.get("/projects/gdm-demo/verify").json()["ok"] is True


def test_web_page_is_served_with_a_strict_policy(tmp_path):
    c = TestClient(create_app(tmp_path))
    r = c.get("/")
    assert r.status_code == 200 and "<title>Saksi</title>" in r.text
    csp = r.headers["content-security-policy"]
    assert "default-src 'none'" in csp and "connect-src 'self'" in csp     # the page talks only to Saksi
    assert "https://" not in r.text.split("<script>")[0]                    # no third-party fonts or styles
