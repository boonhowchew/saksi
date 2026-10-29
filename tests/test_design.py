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
    d = assess({**PRIMARY, "role": "secondary"}, P)
    assert d["label"] == "exploratory" and "not the primary hypothesis" in d["reasons_not_confirmatory"][0]


def test_invented_quotes_fail_safe_to_exploratory():
    assert assess({**PRIMARY, "sample_size_quote": "800 women give 90% power"}, P)["label"] == "exploratory"
    assert assess({**PRIMARY, "primary_quote": "The primary aim is to validate the model"}, P)["label"] == "exploratory"
    assert assess({**PRIMARY, "sample_size_quote": ""}, P)["label"] == "exploratory"
    # A target that is not in the quoted justification
    assert assess({**PRIMARY, "sample_size_target": {"n": 500}}, P)["label"] == "exploratory"


def test_no_protocol_text_means_exploratory():
    d = assess(PRIMARY, None)
    assert d["label"] == "exploratory" and "not available" in d["reasons_not_confirmatory"][0]


def test_sample_check_flags_shortfall_without_changing_label():
    d = assess(PRIMARY, P)
    c = sample_check(d, {"n_analysed": 388, "n_events": 18})
    assert c["shortfall"] and c["targets"]["events"] == {"target": 40, "achieved": 18, "short": True}
    assert sample_check(d, {"n_analysed": 410, "n_events": 45})["shortfall"] is False
    assert sample_check(d, {"n_analysed": 410})["not_reported_by_analysis"] == ["events"]
    assert sample_check(assess({**PRIMARY, "role": "secondary"}, P), {"n_analysed": 1}) is None


def test_engine_decides_whoever_proposes(proj):
    """An author proposing a plan by hand cannot declare an analysis confirmatory."""
    data = P.encode()
    sha = sha256_bytes(data)
    (proj.root / "materials").mkdir(parents=True, exist_ok=True)
    (proj.root / "materials" / sha).write_bytes(data)
    proj.upload_material("bhc", "dataset", "d.csv", SHA, 10)
    proj.upload_material("bhc", "protocol", "protocol.txt", sha, len(data), storage_ref=f"materials/{sha}")
    claimed = {**PRIMARY, "id": "sens", "role": "secondary", "label": "confirmatory",
               "design": {"label": "confirmatory"}, "protocol_quote_verified": True,
               "protocol_quote": "an invented sentence that is not in the protocol"}
    s = proj.propose_plan("bhc", [PRIMARY, claimed], protocol_ref=sha)
    assert s.plan["analyses"]["primary"]["label"] == "confirmatory"
    assert s.plan["analyses"]["sens"]["label"] == "exploratory"
    assert s.plan["analyses"]["sens"]["protocol_quote_verified"] is False
    # Tampering with the stored protocol after upload: nothing can be verified, so nothing is confirmatory.
    (proj.root / "materials" / sha).write_bytes(data + b"\nedited")
    s = proj.propose_plan("bhc", [PRIMARY], protocol_ref=sha)
    assert s.plan["analyses"]["primary"]["label"] == "exploratory"
