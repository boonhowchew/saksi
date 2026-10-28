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
