# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Reviewer Assistant and the review loop, with a scripted model."""

import json

import pytest

from saksi.agents import AgentError
from saksi.workflow import WorkflowError, ledger
from tests.conftest import ANSWERS, RESTATE, sign_all
from tests.test_agents import _draft, _to_drafting, setup  # noqa: F401  (setup is a fixture)

LIMITATION = "With few outcome events, the adjusted estimates are imprecise and the model may be overfitted."
SECRET = "PRIVATE-AUTHOR-NOTE-7731"


def _passing_draft(bmi):
    return _draft(bmi, extra="", exploratory="An exploratory model gave a similar estimate [A:x_q1].")


@pytest.fixture
def in_review(setup, monkeypatch):
    p, agents, model, _ = setup
    monkeypatch.setenv("BASETEN_API_KEY", "test-baseten-key")
    bmi = _to_drafting(p, agents, model)
    model.script["write"] = [_passing_draft(bmi)]
    s = agents.write_draft("bhc")
    assert s.drafts[1]["checks"]["ok"]
    # Author-side material the Reviewer Assistant must never receive.
    text = p.blobs.get_text(s.drafts[1]["text_sha256"])
    p.edit_draft("coauthor", 1, text.replace("The estimate was imprecise.", "The estimate was imprecise. "),
                 summary=f"tidied spacing {SECRET}")
    for a in ("bhc", "coauthor"):
        p.open_draft(a, 2)
        s = p.sign_off(a, 2, f"{RESTATE} {SECRET}", ANSWERS)
    assert s.stage == "REVIEW" and s.round == 1
    return p, agents, model


ROUND1 = {"primary_claim": "Booking BMI was associated with GDM.",
          "summary": "One confirmatory model and one exploratory model. Few events.",
          "points_of_inquiry": [
              {"id": "x", "category": "statistics", "severity": "major",
               "quote": "Booking BMI was associated with GDM",
               "question": "How many outcome events were there per model parameter?",
               "why_it_matters": "Few events per parameter makes the model unstable."},
              "not a point",
              {"category": "made_up", "severity": "huge", "quote": "BMI caused diabetes in every woman",
               "question": "Is this causal?", "why_it_matters": "Observational data."}]}


def test_round_one_briefing_is_independent_and_checked(in_review):
    p, agents, model = in_review
    with pytest.raises(WorkflowError, match="only a reviewer"):
        agents.brief_reviewer("bhc")                         # authors cannot commission it
    model.script["review"] = [ROUND1]
    s = agents.brief_reviewer("rev1")
    b = s.briefings[-1]
    assert b["requested_by"] == "rev1" and b["round"] == 1 and b["draft_version"] == 2
    pts = b["points_of_inquiry"]
    assert [x["id"] for x in pts] == ["P1", "P2"]
    assert pts[0]["quote_verified"] is True and pts[1]["quote_verified"] is False     # invented quote caught
    assert pts[1]["category"] == "other" and pts[1]["severity"] == "minor"

    body = model.bodies[-1]
    assert body["model"] == "meta-models/muse-spark-open-1026"                       # a different model
    assert len(body["messages"]) == 2                                               # fresh context
    sent = json.dumps(body)
    assert SECRET not in sent                                                       # no author-side history
    for leak in ("Person 1", "S100001"):                                            # no data rows
        assert leak not in sent
    assert "ANALYSIS CODE AS RUN" in sent and "AUTOMATIC NUMBER CHECK" in sent and "INTERVAL FACTS" in sent

    full = p.blobs.get_json(b["briefing_sha256"])
    assert full["inputs"]["draft_sha256"] == s.drafts[2]["text_sha256"]
    assert full["automatic_number_check"]["ok"] is True and full["withheld"]
    bmi = next(f for f in full["interval_facts"] if f["name"] == "booking_bmi")
    assert bmi["null_value"] == 1.0 and bmi["interval_includes_null"] is True
    led = ledger(p.log.events())["agent:reviewer_assist"]
    assert led["counts"] == {"review_briefings": 1, "model_calls": 1} and led["on_behalf_of"] == {"rev1": 1}


def test_unusable_briefing_is_logged_then_refused(in_review):
    p, agents, model = in_review
    model.script["review"] = ["I think the paper is fine."]
    with pytest.raises(AgentError):
        agents.brief_reviewer("rev1")
    assert p.log.events()[-1].type == "llm.called" and not p.state().briefings


def test_two_review_rounds_end_to_end(in_review):
    p, agents, model = in_review
    model.script["review"] = [ROUND1]
    s = agents.brief_reviewer("rev1")
    seq = s.briefings[-1]["seq"]

    # The reviewer writes the items; one adopts the assistant's first point.
    with pytest.raises(WorkflowError, match="no point P9"):
        p.add_item("rev1", "R1", "Too few events.", assisted_by_seq=seq, from_point="P9")
    with pytest.raises(WorkflowError, match="not a review briefing"):
        p.add_item("rev1", "R1", "Too few events.", assisted_by_seq=1, from_point="P1")
    p.add_item("rev1", "R1", "Please report events per parameter and discuss overfitting.", location="Results",
               assisted_by_seq=seq, from_point="P1")
    p.add_item("rev1", "R2", "Please add a calibration plot.", location="Results")
    s = p.submit_review("rev1", "revise", "Two items.")
    assert s.stage == "RESPONSE" and s.reviewed_versions[1] == 2

    text = p.blobs.get_text(s.drafts[2]["text_sha256"])
    s = p.edit_draft("bhc", 2, text.replace("The estimate was imprecise.", LIMITATION), "added limitation for R1")
    p.respond("bhc", "R1", "revised", "Limitation added to the Discussion.")
    p.respond("coauthor", "R2", "rebutted", "Calibration was not in the confirmed plan; adding it would be exploratory.")
    sign_all(p)
    s = p.submit_response("bhc")
    assert s.stage == "REVIEW" and s.round == 2

    model.script["review"] = [{"primary_claim": "Unchanged.", "summary": "R1 dealt with; R2 rebutted.",
                               "points_of_inquiry": [],
                               "item_assessments": [
                                   {"item_id": "R1", "assessment": "addressed", "quote": LIMITATION[:60], "note": "ok"},
                                   {"item_id": "R2", "assessment": "partly_addressed", "quote": "",
                                    "note": "The rebuttal is reasonable for a confirmatory paper."},
                                   {"item_id": "R9", "assessment": "addressed", "quote": "", "note": "?"}]}]
    s = agents.brief_reviewer("rev1")
    b = s.briefings[-1]
    assert b["round"] == 2 and b["draft_version"] == 3
    sent = model.bodies[-1]
    assert "later review round" in sent["messages"][0]["content"]
    assert f"+{LIMITATION}" in sent["messages"][1]["content"]                       # it sees what changed
    assert "Calibration was not in the confirmed plan" in sent["messages"][1]["content"]
    full = p.blobs.get_json(b["briefing_sha256"])
    assert [(a["item_id"], a["assessment"], a["quote_verified"]) for a in full["item_assessments"]] == \
        [("R1", "addressed", True), ("R2", "partly_addressed", False)]
    assert full["unknown_item_ids"] == ["R9"] and full["items_not_assessed"] == []
    assert full["inputs"]["previous_version"] == 2

    # The reviewer decides.
    p.give_verdict("rev1", "R1", "satisfied")
    p.give_verdict("rev1", "R2", "agree_to_disagree", "I would still like calibration reported.")
    s = p.submit_review("rev1", "accept", "Accepted.")
    assert s.stage == "ATTESTATION"
    for a in ("bhc", "coauthor"):
        s = p.attest(a, 3, "I have read and stand behind this version.")
    assert s.stage == "FINAL" and p.log.verify().ok
    led = ledger(p.log.events())
    assert led["rev1"]["counts"]["review_comments_from_assistant"] == 1
    assert led["agent:reviewer_assist"]["counts"]["review_briefings"] == 2
