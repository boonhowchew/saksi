# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Confirmatory only if pre-specified, named primary by the protocol and covered by its sample-size justification."""

from saksi import demo_data
from saksi.canonical import sha256_bytes
from saksi.design import assess, sample_check
from tests.conftest import SHA

PRIMARY = {"id": "primary", "protocol_quote": "Multivariable logistic regression with GDM as the outcome",
           "role": "primary", "primary_quote": "Primary objective: To estimate the association between booking body",
           "sample_size_quote": "400 women (about 40 GDM events) give 80% power",
           "sample_size_target": {"n": 400, "events": 40}}
P = demo_data.PROTOCOL


def test_primary_with_verified_quotes_is_confirmatory():
    d = assess(PRIMARY, P)
    assert d["label"] == "confirmatory" and d["reasons_not_confirmatory"] == []


def test_secondary_is_exploratory_even_when_pre_specified():
    d = assess({**PRIMARY, "role": "secondary", "primary_quote": ""}, P)
    assert d["label"] == "exploratory" and "listed as a secondary analysis" in d["reasons_not_confirmatory"][0]
    assert [c["met"] for c in d["conditions"]] == [True, False, True] and d["deviation"] is None


def test_authors_may_name_a_primary_with_a_reason_and_it_is_a_disclosed_departure():
    """Author's decision (30 Oct): authors may name the primary analysis when the protocol does not, or change the
    protocol's, before any result, with a reason the draft must disclose. Described and powered still apply."""
    unnamed = {**PRIMARY, "primary_quote": ""}
    assert assess(unnamed, P)["label"] == "exploratory"                    # no protocol sentence and no reason
    d = assess({**unnamed, "primary_reason": "The funder's report named BMI as the main question."}, P)
    assert d["label"] == "confirmatory" and d["primary_basis"] == "authors"
    assert d["deviation"]["kind"] == "primary_named_by_authors" and "funder's report" in d["deviation"]["statement"]
    # Naming it primary does not make up for a protocol that never described or powered it.
    assert assess({**unnamed, "primary_reason": "x", "sample_size_quote": ""}, P)["label"] == "exploratory"
    # The protocol's own primary, given another role: exploratory, and a departure to disclose.
    d = assess({**PRIMARY, "role": "secondary", "primary_reason": "Objective 2 answers the funder's question."}, P)
    assert d["label"] == "exploratory" and d["deviation"]["kind"] == "protocol_primary_changed"
    # The protocol's primary as the protocol says: no departure.
    assert assess(PRIMARY, P)["deviation"] is None and assess(PRIMARY, P)["primary_basis"] == "protocol"


def test_invented_quotes_fail_safe_to_exploratory():
    assert assess({**PRIMARY, "sample_size_quote": "800 women give 90% power"}, P)["label"] == "exploratory"
    assert assess({**PRIMARY, "primary_quote": "The primary aim is to validate the model"}, P)["label"] == "exploratory"
    assert assess({**PRIMARY, "sample_size_quote": ""}, P)["label"] == "exploratory"
    # A target that is not in the quoted justification
    assert assess({**PRIMARY, "sample_size_target": {"n": 500}}, P)["label"] == "exploratory"


def test_no_protocol_text_means_exploratory():
    d = assess(PRIMARY, None)
    assert d["label"] == "exploratory" and "could not read the protocol" in d["reasons_not_confirmatory"][0]


def test_sample_check_flags_shortfall_without_changing_label():
    d = assess(PRIMARY, P)
    c = sample_check(d, {"n_analysed": 388, "n_events": 18})
    assert c["shortfall"] and c["targets"]["events"] == {"target": 40, "achieved": 18, "short": True}
    assert sample_check(d, {"n_analysed": 410, "n_events": 45})["shortfall"] is False
    assert sample_check(d, {"n_analysed": 410})["not_reported_by_analysis"] == ["events"]
    # Any analysis the sample-size calculation covers gets the check, whatever its label;
    secondary = assess({**PRIMARY, "role": "secondary", "primary_quote": ""}, P)
    assert secondary["label"] == "exploratory" and sample_check(secondary, {"n_analysed": 388})["shortfall"]
    # one it does not cover gets none.
    assert sample_check(assess({**PRIMARY, "sample_size_quote": "", "sample_size_target": {}}, P), {"n_analysed": 1}) is None


def test_engine_decides_whoever_proposes(proj):
    """An author proposing a plan by hand cannot declare an analysis confirmatory."""
    data = P.encode()
    sha = sha256_bytes(data)
    (proj.root / "materials").mkdir(parents=True, exist_ok=True)
    (proj.root / "materials" / sha).write_bytes(data)
    proj.upload_material("bhc", "dataset", "d.csv", SHA, 10)
    proj.upload_material("bhc", "protocol", "protocol.txt", sha, len(data), storage_ref=f"materials/{sha}")
    claimed = {**PRIMARY, "id": "sens", "role": "secondary", "label": "confirmatory", "primary_quote": "",
               "design": {"label": "confirmatory"}, "protocol_quote_verified": True,
               "protocol_quote": "an invented sentence that is not in the protocol"}
    s = proj.propose_plan("bhc", [PRIMARY, claimed], protocol_ref=sha)
    assert s.plan["analyses"]["primary"]["label"] == "confirmatory"
    assert s.plan["analyses"]["sens"]["label"] == "exploratory"
    assert s.plan["analyses"]["sens"]["protocol_quote_verified"] is False
    # Tampering with the stored protocol after upload: nothing can be verified, so nothing is confirmatory.
    (proj.root / "materials" / sha).write_bytes(data + b"\nedited")
    s = proj.propose_plan("bhc", [PRIMARY], protocol_ref=sha, note="retry after the file changed")
    assert s.plan["analyses"]["primary"]["label"] == "exploratory"


def test_pdf_damage_is_tolerated_but_numbers_are_not():
    damaged = P.replace("400 women\n(about 40 GDM events)", "400 wo-\nmen (about 40 GDM\nevents)").replace("fi", "\ufb01")
    assert assess(PRIMARY, damaged)["label"] == "confirmatory"
    d = assess({**PRIMARY, "sample_size_quote":
                "4000 women (about 40 GDM events) give 80% power to detect an odds ratio of 1.10 per kg/m2"}, P)
    assert d["label"] == "exploratory" and "numbers differ" in d["conditions"][2]["detail"]


def _protocol(proj):
    data = P.encode()
    sha = sha256_bytes(data)
    (proj.root / "materials").mkdir(parents=True, exist_ok=True)
    (proj.root / "materials" / sha).write_bytes(data)
    proj.upload_material("bhc", "dataset", "d.csv", SHA, 10)
    proj.upload_material("bhc", "protocol", "protocol.txt", sha, len(data), storage_ref=f"materials/{sha}")
    return sha


def test_only_authors_name_a_primary_and_the_draft_must_disclose_it(proj):
    sha = _protocol(proj)
    unnamed = {**PRIMARY, "primary_quote": "", "primary_reason": "The steering committee chose BMI."}
    # An agent cannot name a primary the protocol does not: the reason is dropped, so it stays exploratory.
    s = proj.propose_plan("agent:analyst", [dict(unnamed)], protocol_ref=sha)
    assert s.plan["analyses"]["primary"]["label"] == "exploratory" and "primary_reason" not in s.plan["analyses"]["primary"]
    # An author can, and is recorded as the one who did.
    s = proj.propose_plan("bhc", [dict(unnamed)], protocol_ref=sha, note="Named BMI primary: the steering committee chose it.")
    a = s.plan["analyses"]["primary"]
    assert a["label"] == "confirmatory" and a["primary_reason_by"] == "bhc" and a["design"]["primary_basis"] == "authors"
    # The analysis carries the departure, and the draft check requires it to be disclosed.
    for x in ("bhc", "coauthor"):
        s = proj.confirm_plan(x, s.plan["sha"])
    out = proj.blobs.put_json({"n_analysed": 400, "n_events": 40, "estimates": [{"name": "bmi", "estimate": 1.07}]})
    s = proj.complete_analysis("agent:analyst", "primary", plan_analysis_id="primary", spec={"id": "primary"},
                               code_sha256=SHA, outputs_sha256=out, execution={}, status="ok")
    assert s.analyses["primary"]["protocol_deviation"]["kind"] == "primary_named_by_authors"
    assert proj.check_text("The odds ratio was 1.07 [A:primary].")["departure_not_disclosed"] == ["primary"]
    r = proj.check_text("The authors named this the primary analysis, which the protocol did not, because the "
                        "steering committee chose it [A:primary]. The odds ratio was 1.07 [A:primary] (Table 1, Figure 1).")
    assert r["departure_not_disclosed"] == [] and r["ok"], {k: v for k, v in r.items() if v and k not in ("traced", "numbers_checked", "untraced")}
    return


def test_changing_the_protocols_primary_needs_a_reason(proj):
    sha = _protocol(proj)
    import pytest
    from saksi.workflow import WorkflowError
    with pytest.raises(WorkflowError, match="say why it is now a secondary"):
        proj.propose_plan("bhc", [{**PRIMARY, "role": "secondary"}], protocol_ref=sha, note="demoted")


def test_tracer_requires_the_departure_to_be_disclosed():
    from saksi.tracing import trace
    an, out = {"p": {"label": "confirmatory"}}, {"p": {"estimate": 1.07}}
    r = trace("The adjusted odds ratio was 1.07 [A:p].", an, out, must_disclose=["p"])
    assert not r["ok"] and r["departure_not_disclosed"] == ["p"]
    r = trace("The protocol named no primary analysis; the authors named this one primary before any result, "
              "because the steering committee chose it [A:p]. The adjusted odds ratio was 1.07 [A:p].", an, out,
              must_disclose=["p"])
    assert r["ok"] and r["departure_not_disclosed"] == []
