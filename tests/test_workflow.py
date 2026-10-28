# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

import pytest

from saksi.workflow import Project, WorkflowError, ledger
from tests.conftest import ANSWERS, AUTHOR_A, RESTATE, SHA, run_planned, sign_all, to_analysis, to_revision


def test_full_lifecycle_two_rounds(proj):
    p = proj
    to_revision(p)
    s = sign_all(p)
    assert s.stage == "REVIEW" and s.round == 1

    p.add_item("rev1", "R1", "The odds ratio in Table 2 is not adjusted for parity.", location="Table 2")
    p.add_item("rev1", "R2", "Say that the GBC model is exploratory.", location="Results")
    s = p.submit_review("rev1", "revise")
    assert s.stage == "RESPONSE"

    s = p.edit_draft("coauthor", s.draft_version, "Draft two text, adjusted for parity.", "Adjusted model")
    p.respond("coauthor", "R1", "revised")
    p.respond("bhc", "R2", "rebutted", note="The GBC is a protocol-specified sensitivity analysis.")
    sign_all(p)
    s = p.submit_response("bhc")
    assert s.stage == "REVIEW" and s.round == 2

    p.give_verdict("rev1", "R1", "satisfied")
    p.give_verdict("rev1", "R2", "agree_to_disagree", note="I still read it as exploratory.")
    s = p.submit_review("rev1", "accept")
    assert s.stage == "ATTESTATION"

    for a in s.authors:
        s = p.attest(a, s.draft_version, "I take responsibility for this work as recorded.")
    assert s.stage == "FINAL"
    assert p.log.verify().ok

    stages = [e.payload["to"] for e in p.log.events() if e.type == "stage.advanced"]
    assert stages == ["ANALYSIS", "DRAFTING", "AUTHOR_REVISION", "REVIEW", "RESPONSE", "REVIEW", "ATTESTATION", "FINAL"]


def test_every_author_must_confirm_plan(proj):
    s = to_analysis(proj)
    assert s.stage == "ANALYSIS"
    # Restart: only one author confirms.
    p2 = Project(proj.root.parent, "demo2")
    p2.create("x", "primary", AUTHOR_A)
    p2.register_human("bhc", {"actor_id": "b", "name": "B"}, "author")
    p2.upload_material("bhc", "dataset", "d.csv", SHA, 1)
    p2.upload_material("bhc", "protocol", "p.pdf", SHA, 1)
    s = p2.propose_plan("bhc", [{"id": "a1"}], None)
    s = p2.confirm_plan("bhc", s.plan["sha"])
    assert s.stage == "INGEST"


def test_changed_plan_voids_confirmations(proj):
    p = proj
    p.upload_material("bhc", "dataset", "d.csv", SHA, 1)
    p.upload_material("bhc", "protocol", "p.pdf", SHA, 1)
    s = p.propose_plan("agent:analyst", [{"id": "a1"}], None)
    old = s.plan["sha"]
    p.confirm_plan("bhc", old)
    s = p.propose_plan("agent:analyst", [{"id": "a1"}, {"id": "a2"}], None)
    assert s.plan_confirmed_by == set()
    with pytest.raises(WorkflowError, match="changed"):
        p.confirm_plan("coauthor", old)


def test_plan_needs_dataset_and_protocol(proj):
    s = proj.propose_plan("agent:analyst", [{"id": "a1"}], None)
    with pytest.raises(WorkflowError, match="dataset and a protocol"):
        proj.confirm_plan("bhc", s.plan["sha"])


def test_exploratory_label_is_assigned_by_engine(proj):
    to_analysis(proj)
    s = proj.complete_analysis("agent:analyst", "extra1", plan_analysis_id=None, spec={}, code_sha256=SHA,
                               outputs_sha256=SHA, execution={}, status="ok")
    assert s.analyses["extra1"]["label"] == "exploratory"
    with pytest.raises(WorkflowError, match="not in the confirmed plan"):
        proj.complete_analysis("agent:analyst", "fake", plan_analysis_id="invented", spec={}, code_sha256=SHA,
                               outputs_sha256=SHA, execution={}, status="ok")


def test_acceptance_voided_by_new_analysis(proj):
    to_analysis(proj)
    s = run_planned(proj)
    old = s.analysis_set_sha()
    proj.accept_analyses("bhc", old)
    proj.complete_analysis("agent:analyst", "extra1", plan_analysis_id=None, spec={}, code_sha256=SHA,
                           outputs_sha256=SHA, execution={}, status="ok")
    with pytest.raises(WorkflowError, match="changed"):
        proj.accept_analyses("coauthor", old)
    s = proj.state()
    for a in s.authors:
        s = proj.accept_analyses(a, s.analysis_set_sha())
    assert s.stage == "DRAFTING"


def test_cannot_accept_until_planned_analyses_succeed(proj):
    to_analysis(proj)
    s = proj.complete_analysis("agent:analyst", "primary", plan_analysis_id="primary", spec={}, code_sha256=SHA,
                               outputs_sha256=None, execution={}, status="error", error="did not converge")
    with pytest.raises(WorkflowError, match="not yet completed"):
        proj.accept_analyses("bhc", s.analysis_set_sha())


def test_signoff_is_structured_and_version_bound(proj):
    s = to_revision(proj)
    with pytest.raises(WorkflowError, match="open this version"):
        proj.sign_off("bhc", s.draft_version, RESTATE, ANSWERS)
    proj.open_draft("bhc", s.draft_version)
    with pytest.raises(WorkflowError, match="own words"):
        proj.sign_off("bhc", s.draft_version, "Looks good.", ANSWERS)
    with pytest.raises(WorkflowError, match="unanswered"):
        proj.sign_off("bhc", s.draft_version, RESTATE, {})
    proj.sign_off("bhc", s.draft_version, RESTATE, ANSWERS)
    # An edit by the co-author voids the sign-off: the stage stays put.
    s = proj.edit_draft("coauthor", s.draft_version, "Changed text.", "tweak")
    proj.open_draft("coauthor", s.draft_version)
    s = proj.sign_off("coauthor", s.draft_version, RESTATE, ANSWERS)
    assert s.stage == "AUTHOR_REVISION"
    s = sign_all(proj)
    assert s.stage == "REVIEW"


def test_agent_edits_must_name_the_author(proj):
    s = to_revision(proj)
    with pytest.raises(WorkflowError, match="name the author"):
        proj.edit_draft("agent:writer", s.draft_version, "agent text", "rewrite")
    s = proj.edit_draft("agent:writer", s.draft_version, "agent text", "rewrite", requested_by="coauthor")
    row = ledger(proj.log.events())["agent:writer"]
    assert row["on_behalf_of"] == {"coauthor": 1}


def test_roles_are_enforced(proj):
    s = to_revision(proj)
    with pytest.raises(WorkflowError, match="may not"):
        proj.edit_draft("rev1", s.draft_version, "reviewer rewrites", "x")
    with pytest.raises(WorkflowError, match="may not"):
        proj.register_human("coauthor", {"actor_id": "z", "name": "Z"}, "author")
    with pytest.raises(WorkflowError, match="may not"):
        proj.complete_analysis("bhc", "x", plan_analysis_id=None, spec={}, code_sha256=SHA,
                               outputs_sha256=SHA, execution={}, status="ok")


def test_review_cannot_accept_with_open_items(proj):
    to_revision(proj)
    sign_all(proj)
    proj.add_item("rev1", "R1", "Unclear denominator.")
    with pytest.raises(WorkflowError, match="open"):
        proj.submit_review("rev1", "accept")


def test_verdict_needs_response_and_revised_needs_edit(proj):
    to_revision(proj)
    sign_all(proj)
    proj.add_item("rev1", "R1", "Unclear denominator.")
    with pytest.raises(WorkflowError, match="response first"):
        proj.give_verdict("rev1", "R1", "satisfied")
    proj.submit_review("rev1", "revise")
    with pytest.raises(WorkflowError, match="new draft version"):
        proj.respond("bhc", "R1", "revised")
    with pytest.raises(WorkflowError, match="explain"):
        proj.respond("bhc", "R1", "rebutted")


def test_still_concerned_reopens_item(proj):
    to_revision(proj)
    sign_all(proj)
    proj.add_item("rev1", "R1", "Unclear denominator.")
    proj.submit_review("rev1", "revise")
    proj.respond("bhc", "R1", "clarified", note="Denominator is all eligible records.")
    sign_all(proj)
    proj.submit_response("bhc")
    s = proj.give_verdict("rev1", "R1", "still_concerned", note="Show the flow diagram.")
    assert s.items["R1"]["status"] == "open"
    s = proj.submit_review("rev1", "revise")
    assert s.stage == "RESPONSE" and s.round == 2


def test_round_limit_forces_resolution(tmp_path):
    from tests.conftest import AUTHOR_B, REVIEWER, TickClock
    p = Project(tmp_path, "lim", clock=TickClock())
    p.create("x", "primary", AUTHOR_A, max_rounds=1)
    p.register_human("bhc", AUTHOR_B, "author")
    p.register_human("bhc", REVIEWER, "reviewer")
    to_revision(p)
    sign_all(p)
    p.add_item("rev1", "R1", "Concern.")
    with pytest.raises(WorkflowError, match="round limit"):
        p.submit_review("rev1", "revise")


def test_submit_response_requires_fresh_signoffs(proj):
    to_revision(proj)
    sign_all(proj)
    proj.add_item("rev1", "R1", "Concern.")
    s = proj.submit_review("rev1", "revise")
    proj.edit_draft("bhc", s.draft_version, "Revised text.", "fix")
    proj.respond("bhc", "R1", "revised")
    with pytest.raises(WorkflowError, match="sign off"):
        proj.submit_response("bhc")


def test_review_waits_for_a_reviewer(tmp_path):
    from tests.conftest import AUTHOR_B, TickClock
    p = Project(tmp_path, "nr", clock=TickClock())
    p.create("x", "primary", AUTHOR_A)
    p.register_human("bhc", AUTHOR_B, "author")
    to_revision(p)
    s = sign_all(p)
    assert s.stage == "AUTHOR_REVISION"
    p.register_human("bhc", {"actor_id": "r", "name": "R"}, "reviewer")
    s = p.start_review("bhc")
    assert s.stage == "REVIEW"


def test_identity_rules(proj):
    with pytest.raises(WorkflowError, match="ORCID"):
        proj.register_human("bhc", {"actor_id": "x", "name": "X", "orcid": "1234"}, "author")
    with pytest.raises(WorkflowError, match="already belongs"):
        proj.register_human("bhc", {"actor_id": "x", "name": "X", "orcid": "0000-0002-8627-6248"}, "reviewer")
    with pytest.raises(WorkflowError, match="already registered"):
        proj.register_human("bhc", {"actor_id": "rev1", "name": "X"}, "author")
    with pytest.raises(WorkflowError, match="agent"):
        proj.register_human("bhc", {"actor_id": "agent:evil", "name": "X"}, "author")


def test_refused_command_writes_nothing(proj):
    before = len(proj.log.events())
    with pytest.raises(WorkflowError):
        proj.sign_off("bhc", 1, RESTATE, ANSWERS)
    assert len(proj.log.events()) == before


def test_ledger_attributes_each_human(proj):
    to_revision(proj)
    sign_all(proj)
    led = ledger(proj.log.events())
    assert led["bhc"]["counts"]["verification_decisions"] == 3   # plan, analyses, sign-off
    assert led["coauthor"]["counts"]["verification_decisions"] == 3
    assert led["bhc"]["verification_seconds"] > 0
    assert "materials" in led["bhc"]["counts"] and "materials" not in led["coauthor"]["counts"]


def test_state_is_rebuilt_identically(proj):
    to_revision(proj)
    sign_all(proj)
    a = proj.state().to_dict()
    b = Project(proj.root.parent, "demo").state().to_dict()
    assert a == b
