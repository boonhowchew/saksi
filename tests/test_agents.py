# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Agents end to end with a scripted model and the local executor."""

import json
from pathlib import Path

import pytest

from saksi import demo_data
from saksi.agents import AgentError, Agents
from saksi.blobs import BlobStore
from saksi.canonical import sha256_bytes
from saksi.llm import ModelClient
from saksi.sandbox import LocalExecutor
from saksi.workflow import ledger
from tests.test_sandbox import GOOD

AUC_BROKEN = '''
import pandas as pd
def analyse(data_path, seed):
    df = pd.read_csv(data_path)
    return {"x": float(df["no_such_column"].mean())}
'''
AUC_FIXED = '''
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

def analyse(data_path, seed):
    df = pd.read_csv(data_path).dropna(subset=["gdm", "booking_bmi", "age", "parity", "previous_gdm"])
    m = smf.logit("gdm ~ booking_bmi + age + parity + previous_gdm", data=df).fit(disp=0)
    p = m.predict(df)
    rng = np.random.default_rng(seed)
    boots = []
    y = df["gdm"].to_numpy(); pv = p.to_numpy()
    for _ in range(200):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            boots.append(roc_auc_score(y[i], pv[i]))
    return {"n_analysed": int(len(df)), "n_excluded": 0, "exclusion_reason": "",
            "estimates": [{"name": "AUC", "measure": "AUC", "estimate": float(roc_auc_score(y, pv)),
                           "ci_lower": float(np.percentile(boots, 2.5)), "ci_upper": float(np.percentile(boots, 97.5)),
                           "p_value": None}],
            "model": {"formula_or_description": "same as primary", "converged": True}, "notes": []}


def figures(data_path, seed):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    df = pd.read_csv(data_path).dropna(subset=["gdm", "booking_bmi", "age", "parity", "previous_gdm"])
    m = smf.logit("gdm ~ booking_bmi + age + parity + previous_gdm", data=df).fit(disp=0)
    fpr, tpr, _ = roc_curve(df["gdm"], m.predict(df))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    return {"roc_curve": {"figure": fig, "caption": "ROC curve of the primary model in 388 women."}}
'''
CURATOR_REPLY = {
    "summary": "400 records, 10 columns. Two direct identifiers should be removed.",
    "dictionary": [{"column": "gdm", "label": "Gestational diabetes", "type": "binary", "units_or_coding": "0/1",
                    "role": "outcome"},
                   {"column": "booking_bmi", "label": "BMI at booking", "type": "numeric", "units_or_coding": "kg/m2",
                    "role": "exposure"}],
    "flags": [{"column": "patient_name", "severity": "high", "issue": "direct identifier",
               "suggested_check": "remove before analysis"}],
    "protocol_variables_missing_from_data": [],
}
PLAN_REPLY = {"analyses": [
    {"id": "primary", "title": "BMI and GDM", "objective": "association", "outcome": {"name": "GDM", "column": "gdm"},
     "exposures": [{"name": "BMI", "column": "booking_bmi"}],
     "covariates": [{"name": "age", "column": "age"}, {"name": "parity", "column": "parity"},
                    {"name": "previous GDM", "column": "previous_gdm"}],
     "method": "multivariable logistic regression", "effect_measure": "adjusted odds ratio",
     "missing_data": "complete case",
     "protocol_quote": "Multivariable logistic regression with GDM as the outcome and booking BMI as the exposure",
     "role": "primary",
     "primary_quote": "Primary objective: To estimate the association between booking body mass index (BMI) and "
                      "gestational diabetes mellitus (GDM)",
     "sample_size_quote": "400 women (about 40 GDM events) give 80% power to detect an odds ratio of 1.10 per kg/m2",
     "sample_size_target": {"n": 400, "events": 40},
     "unresolved": []},
    {"id": "auc", "title": "Discrimination", "objective": "AUC", "outcome": {"name": "GDM", "column": "gdm"},
     "exposures": [], "covariates": [{"name": "waist", "column": "waist_cm"}], "method": "AUC with 95% CI",
     "effect_measure": "AUC", "missing_data": "complete case",
     "protocol_quote": "The model was externally validated in three cohorts", "role": "secondary",
     "primary_quote": "", "sample_size_quote": "", "sample_size_target": {"n": None, "events": None},
     "unresolved": []},
]}


class ScriptedModel:
    """Answers by task, in order. Records every request body it receives."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.bodies = []

    def post(self, url, headers, json, timeout):
        self.bodies.append(json)
        system = json["messages"][0]["content"]
        purpose = ("review" if "Reviewer Assistant" in system else "curate" if "Data Curator" in system
                   else "plan" if "structured analysis plan" in system
                   else "write" if "Writing Agent" in system else "code")
        reply = self.script[purpose].pop(0)
        reply, finish = reply if isinstance(reply, tuple) else (reply, "stop")     # (text, "length") = cut off
        content = reply if isinstance(reply, str) else __import__("json").dumps(reply)
        return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}}


def fence(code):
    return f"```python\n{code}```"


def upload(p, kind, filename, data):
    sha = sha256_bytes(data)
    dest = p.root / "materials" / sha
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    p.upload_material("bhc", kind, filename, sha, len(data), storage_ref=f"materials/{sha}")
    return sha


@pytest.fixture
def setup(proj, monkeypatch, tmp_path):
    monkeypatch.setenv("META_API_KEY", "test-key")
    model = ScriptedModel({
        "curate": [CURATOR_REPLY],
        "plan": [PLAN_REPLY],
        "code": [fence(GOOD), fence(AUC_BROKEN), fence(AUC_FIXED), fence(GOOD)],
    })
    agents = Agents(proj, ModelClient(proj.blobs, transport=model, mode="live"), LocalExecutor())
    data = demo_data.make().to_csv(index=False).encode()
    ds_sha = upload(proj, "dataset", "gdm_synthetic.csv", data)
    upload(proj, "protocol", "protocol.txt", demo_data.PROTOCOL.encode())
    return proj, agents, model, ds_sha


def test_curate_plan_run_end_to_end(setup):
    p, agents, model, ds_sha = setup

    s = agents.curate("bhc")
    rep = p.blobs.get_json(s.curator_reports[-1]["report_sha256"])
    assert rep["flags"][0]["column"] == "patient_name" and rep["prompt"]["name"] == "curator"
    assert s.curator_reports[-1]["material_sha256"] == ds_sha

    s = agents.propose_plan("coauthor")
    plan = s.plan["analyses"]
    assert plan["primary"]["protocol_quote_verified"] is True
    assert plan["auc"]["protocol_quote_verified"] is False        # invented quote is caught
    assert plan["auc"]["columns_not_in_data"] == ["waist_cm"]     # invented column is caught

    for a in ("bhc", "coauthor"):
        s = p.confirm_plan(a, s.plan["sha"])
    assert s.stage == "ANALYSIS"

    s = agents.run_planned("bhc")
    prim, auc = s.analyses["primary"], s.analyses["auc"]
    assert prim["status"] == "ok" and prim["label"] == "confirmatory" and prim["basis"] == "pre-specified"
    # Planned for 400 women and 40 events; 388 and 18 were analysed. Flagged; the label stays.
    assert prim["sample_check"]["shortfall"] is True
    assert prim["sample_check"]["targets"] == {"n": {"target": 400, "achieved": 388, "short": True},
                                               "events": {"target": 40, "achieved": 18, "short": True}}
    assert auc["label"] == "exploratory" and auc["basis"] == "pre-specified"     # secondary: not powered for it
    assert any("listed as a secondary analysis" in r
               for r in s.plan_locked["analyses"]["auc"]["design"]["reasons_not_confirmatory"])

    # Figures: Saksi draws the forest plot and flow diagram from the recorded results; the ROC curve,
    # required by rule for an AUC analysis, is drawn by the analysis code in the sandbox.
    figs = {(f["analysis_id"], f["name"]): f for f in s.figures}
    assert figs[("primary", "forest_plot")]["drawn_by"] == "saksi" and figs[("primary", "forest_plot")]["number"] == 1
    assert figs[("primary", "participant_flow")]["drawn_by"] == "saksi"
    assert auc["spec"]["required_figure"]["name"] == "roc_curve" and auc["execution"]["required_figure_drawn"]
    roc = figs[("auc", "roc_curve")]
    assert roc["drawn_by"] == "analysis code" and roc["label"] == "exploratory"
    # The code's caption did not say exploratory; Saksi states the label, keeping the original.
    assert roc["caption"].startswith("Exploratory (pre-specified) analysis. ROC curve")
    assert roc["caption_from_code"] == "ROC curve of the primary model in 388 women."
    assert roc["caption_check"]["ok"] is True
    assert p.blobs.get_text(roc["svg_sha256"]).lstrip().startswith("<?xml")
    assert all(f["caption_check"]["untraced"] == 0 for f in s.figures)
    # AUC_FIXED claims 0 excluded while 12 of 400 records were dropped: Saksi flags the flow diagram.
    assert "do not add up" in figs[("auc", "participant_flow")]["warning"]
    assert figs[("primary", "participant_flow")]["warning"] is None
    assert len(prim["execution"]["attempts"]) == 1
    assert prim["execution"]["provenance"]["data_sha256"] == ds_sha
    assert auc["status"] == "ok" and [a["outcome"] for a in auc["execution"]["attempts"]] == ["error", "ok"]
    assert len(auc["llm_call_seqs"]) == 2                           # failed attempt and fix both logged
    out = p.blobs.get_json(auc["outputs_sha256"])
    assert 0.5 < out["estimates"][0]["estimate"] < 1.0

    s = agents.run_request("coauthor", "q1", "Same model stratified by ethnicity")
    x = s.analyses["x_q1"]
    assert x["label"] == "exploratory" and x["basis"] == "post hoc" and x["requested_by"] == "coauthor"

    for a in s.authors:
        s = p.accept_analyses(a, s.analysis_set_sha())
    assert s.stage == "DRAFTING"
    assert p.log.verify().ok

    led = ledger(p.log.events())
    assert led["agent:analyst"]["on_behalf_of"] == {"coauthor": 2, "bhc": 2}
    assert led["agent:analyst"]["counts"]["model_calls"] == 5     # plan + primary + auc x2 + request

    # The models never received a data row.
    sent = json.dumps(model.bodies)
    for leak in ("Person 1", "S100001", "S100399"):
        assert leak not in sent


def test_rejected_code_is_recorded_as_failure(setup):
    p, agents, model, _ = setup
    model.script["code"] = [fence("import os\ndef analyse(data_path, seed):\n    return {}")] * 2
    agents.curate("bhc")
    model.script["plan"] = [{"analyses": [PLAN_REPLY["analyses"][0]]}]
    s = agents.propose_plan("bhc")
    for a in ("bhc", "coauthor"):
        s = p.confirm_plan(a, s.plan["sha"])
    s = agents.run_planned("bhc")
    a = s.analyses["primary"]
    assert a["status"] == "error" and "import of 'os'" in a["error"]
    assert [x["outcome"] for x in a["execution"]["attempts"]] == ["rejected", "rejected"]
    # A rerun gets a new id; the failure stays in the record.
    model.script["code"] = [fence(GOOD)]
    s = agents.run_planned("bhc")
    assert s.analyses["primary.r1"]["status"] == "ok" and s.analyses["primary"]["status"] == "error"


def test_bad_json_from_curator_is_retried_once_then_refused(setup):
    p, agents, model, _ = setup
    model.script["curate"] = ["sorry, I cannot", "still not JSON"]
    with pytest.raises(AgentError, match="after one retry"):
        agents.curate("bhc")
    assert [e.type for e in p.log.events()[-2:]] == ["llm.called", "llm.called"]
    assert "YOUR PREVIOUS REPLY WAS NOT VALID JSON" in model.bodies[-1]["messages"][1]["content"]


def test_curator_retry_can_succeed(setup):
    p, agents, model, _ = setup
    model.script["curate"] = ['{"summary": "x", "dictionary": [', CURATOR_REPLY]
    s = agents.curate("bhc")
    assert s.curator_reports[-1]["summary"].startswith("400 records")
    assert len(p.blobs.get_json(s.curator_reports[-1]["report_sha256"])["earlier_call_seqs"]) == 1


def test_wide_dataset_is_curated_in_batches_and_a_cut_off_batch_is_split(proj, monkeypatch):
    """29 Oct, live: a real dataset with many columns broke the Data Curator. Its single reply was cut
    off at the length limit mid-dictionary ("Expecting ',' delimiter")."""
    import numpy as np
    import pandas as pd
    monkeypatch.setenv("META_API_KEY", "k")
    rng = np.random.default_rng(1)
    wide = pd.DataFrame({f"v{i:03d}": rng.normal(size=60).round(2) for i in range(90)})
    upload(proj, "dataset", "wide.csv", wide.to_csv(index=False).encode())
    upload(proj, "protocol", "protocol.txt", demo_data.PROTOCOL.encode())
    entry = lambda a, b: {"dictionary": [{"column": f"v{i:03d}", "label": "x", "type": "numeric",
                                          "units_or_coding": "unknown", "role": "unclear"} for i in range(a, b)],
                          "flags": [{"column": f"v{a:03d}", "severity": "low", "issue": "check", "suggested_check": "c"}]}
    model = ScriptedModel({"curate": [
        ('{"dictionary": [{"column": "v000"}', "length"),          # first 40 columns: cut off
        entry(0, 20), entry(20, 40), entry(40, 80), entry(80, 90),
        {"summary": "90 numeric columns.", "protocol_variables_missing_from_data": ["gdm"], "flags": []}]})
    agents = Agents(proj, ModelClient(proj.blobs, transport=model, mode="live"), LocalExecutor())
    s = agents.curate("bhc")
    rep = proj.blobs.get_json(s.curator_reports[-1]["report_sha256"])
    assert rep["columns"] == 90 and len(rep["dictionary"]) == 90 and len(rep["flags"]) == 4
    assert rep["summary"] == "90 numeric columns." and rep["protocol_variables_missing_from_data"] == ["gdm"]
    assert [x["name"] for x in rep["prompts"]] == ["curator_columns", "curator_overview"]
    calls = [e for e in proj.log.events() if e.type == "llm.called"]
    assert len(calls) == 6                                      # the cut-off reply is in the record too
    assert "cut off" not in json.dumps(model.bodies[-1]) and "v089" in model.bodies[-1]["messages"][1]["content"]


def test_tampered_material_is_refused(setup):
    p, agents, model, ds_sha = setup
    path = p.root / "materials" / ds_sha
    path.write_bytes(path.read_bytes() + b"\n1,2,3")
    with pytest.raises(AgentError, match="does not match"):
        agents.curate("bhc")


def test_returned_error_is_labelled_honestly(setup):
    p, agents, model, _ = setup
    cannot = "def analyse(data_path, seed):\n    return {'error': 'column waist_cm not found'}\n"
    model.script["code"] = [fence(cannot), fence(GOOD)]
    agents.curate("bhc")
    model.script["plan"] = [{"analyses": [PLAN_REPLY["analyses"][0]]}]
    s = agents.propose_plan("bhc")
    for a in ("bhc", "coauthor"):
        s = p.confirm_plan(a, s.plan["sha"])
    s = agents.run_planned("bhc")
    a = s.analyses["primary"]
    assert a["status"] == "ok"
    assert [x["outcome"] for x in a["execution"]["attempts"]] == ["returned_error", "ok"]


def _to_drafting(p, agents, model):
    agents.curate("bhc")
    model.script["plan"] = [{"analyses": [PLAN_REPLY["analyses"][0]]}]
    s = agents.propose_plan("bhc")
    for a in ("bhc", "coauthor"):
        s = p.confirm_plan(a, s.plan["sha"])
    model.script["code"] = [fence(GOOD), fence(GOOD)]
    agents.run_planned("bhc")
    s = agents.run_request("coauthor", "q1", "Same model, any extra")
    for a in s.authors:
        s = p.accept_analyses(a, s.analysis_set_sha())
    assert s.stage == "DRAFTING"
    est = {e["name"]: e for e in p.blobs.get_json(s.analyses["primary"]["outputs_sha256"])["estimates"]}
    return est["booking_bmi"]


def _draft(bmi, extra="The adjusted OR for previous GDM was 9.99 [A:primary]. ",
           exploratory="A further model gave a similar estimate [A:x_q1]."):
    return ("[Authors: introduction]\n\n## Methods\nA multivariable logistic regression was pre-specified.\n\n"
            "## Results\n"
            f"Of 400 records, 388 were analysed, with 18 GDM events against 40 planned [A:primary]. "
            f"Booking BMI was associated with GDM "
            f"(adjusted OR {bmi['estimate']:.2f}, 95% CI {bmi['ci_lower']:.2f} to {bmi['ci_upper']:.2f}) [A:primary]. "
            f"{extra}{exploratory}\n\n## Discussion\nThe estimate was imprecise.\n")


def test_writer_fixes_its_own_draft_after_check(setup):
    p, agents, model, _ = setup
    bmi = _to_drafting(p, agents, model)
    model.script["write"] = [_draft(bmi),
                             _draft(bmi, extra="", exploratory="An exploratory model gave a similar estimate [A:x_q1].")]
    s = agents.write_draft("bhc")
    d = s.drafts[1]
    assert d["by"] == "agent:writer" and d["requested_by"] == "bhc"
    assert d["sections"] == ["Methods", "Results", "Discussion"]
    assert d["checks"]["ok"] is True
    assert [a["passed_check"] for a in d["writer_attempts"]] == [False, True]
    first = p.blobs.get_json(d["writer_attempts"][0]["findings_sha256"])
    assert [u["number"] for u in first["untraced"]] == ["9.99"] and first["exploratory_not_labelled"]
    assert d["writer_attempts"][1]["prompt"]["name"] == "rewrite"
    # The rewrite request told the agent exactly what was wrong.
    assert '"interval_includes_null": true' in model.bodies[-2]["messages"][1]["content"]   # facts, not judgement
    assert "Figure 1 [A:primary]: Adjusted odds ratio with 95% confidence intervals" in model.bodies[-2]["messages"][1]["content"]
    rewrite_msg = model.bodies[-1]["messages"][1]["content"]
    assert "UNTRACED 9.99" in rewrite_msg and "EXPLORATORY NOT LABELLED" in rewrite_msg
    # Both model calls are in the log: the failed attempt and the final one.
    calls = [e for e in p.log.events() if e.type == "llm.called" and e.actor_id == "agent:writer"]
    assert len(calls) == 2


def test_writer_cannot_pass_by_dropping_an_analysis(setup):
    p, agents, model, _ = setup
    bmi = _to_drafting(p, agents, model)
    # Second attempt "fixes" things by deleting the awkward analysis.
    model.script["write"] = [_draft(bmi), _draft(bmi, extra="", exploratory="")]
    s = agents.write_draft("bhc")
    d = s.drafts[1]
    assert d["checks"]["ok"] is False and d["checks"]["analyses_not_reported"] == ["x_q1"]
    assert [a["passed_check"] for a in d["writer_attempts"]] == [False, False]
    assert s.stage == "AUTHOR_REVISION"          # escalated to the authors, with the findings


def test_unresolved_draft_goes_to_authors_who_can_fix_it(setup):
    from tests.conftest import ANSWERS, RESTATE
    p, agents, model, _ = setup
    bmi = _to_drafting(p, agents, model)
    model.script["write"] = [_draft(bmi), _draft(bmi)]       # agent fails twice
    s = agents.write_draft("bhc")
    d = s.drafts[1]
    assert d["checks"]["untraced"] == 1 and d["checks"]["exploratory_not_labelled"] == 1
    full = p.blobs.get_json(d["checks_sha256"])
    assert full["untraced"][0]["number"] == "9.99"
    # An author edit that fixes both problems is re-checked automatically.
    text = p.blobs.get_text(d["text_sha256"])
    text = text.replace("The adjusted OR for previous GDM was 9.99 [A:primary]. ", "")
    text = text.replace("A further model", "A further exploratory model")
    s = p.edit_draft("coauthor", 1, text, "removed unsupported figure; labelled exploratory analysis")
    assert s.drafts[2]["checks"]["ok"] is True
    # The sign-off records the check of the exact version signed.
    p.open_draft("bhc", 2)
    s = p.sign_off("bhc", 2, RESTATE, ANSWERS)
    assert s.signoffs[2]["bhc"]["checks_at_signoff"]["ok"] is True


def test_wide_dataset_analysis_sees_only_its_columns():
    from saksi.agents import _focus
    prof = {"rows": 10, "columns": [{"name": f"v{i}"} for i in range(100)]}
    d = {"profile": prof, "dictionary": [{"column": f"v{i}"} for i in range(100)]}
    dic, p, names = _focus(d, [{"outcome": {"column": "v1"}, "exposures": [{"column": "v2"}], "covariates": []}])
    assert [c["name"] for c in p["columns"]] == ["v1", "v2"] and len(dic) == 2 and len(names) == 100
    assert _focus(d, [{"title": "author request"}])[2] is None          # nothing named: everything
