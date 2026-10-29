# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

from saksi.tracing import trace

ANALYSES = {"primary": {"label": "confirmatory"}, "x_q1": {"label": "exploratory"}}
OUTPUTS = {
    "primary": {"n_analysed": 388, "n_excluded": 12,
                "estimates": [{"name": "booking_bmi", "estimate": 1.0673717, "ci_lower": 0.9607546,
                               "ci_upper": 1.1858204, "p_value": 0.2263},
                              {"name": "previous_gdm", "estimate": 3.647152, "ci_lower": 0.937525,
                               "ci_upper": 14.18812, "p_value": 0.00042}]},
    "x_q1": {"estimates": [{"name": "Malay", "estimate": 1.21, "ci_lower": 0.8, "ci_upper": 1.9}]},
}


def test_correct_numbers_are_traced():
    text = ("Of 400 records, 388 were analysed and 12 excluded for missing BMI [A:primary]. "
            "Each unit of booking BMI was associated with an adjusted odds ratio of 1.07 (95% CI 0.96 to 1.19; "
            "p = 0.226) [A:primary]. Previous GDM: OR 3.65 (0.94–14.19; p < 0.001) [A:primary].")
    r = trace(text, ANALYSES, OUTPUTS, dataset_counts=[400])
    assert r["ok"], r
    assert r["numbers_checked"] == 11          # 400 388 12 1.07 0.96 1.19 0.226 3.65 0.94 14.19 <0.001 (95 skipped)


def test_invented_number_is_flagged():
    text = "Booking BMI was associated with GDM (OR 2.15, 95% CI 1.40 to 3.30) [A:primary]."
    r = trace(text, ANALYSES, OUTPUTS)
    assert not r["ok"]
    assert {u["number"] for u in r["untraced"]} == {"2.15", "1.40", "3.30"}


def test_number_without_citation_is_flagged():
    r = trace("Booking BMI was associated with GDM (OR 1.07).", ANALYSES, OUTPUTS)
    assert [u["number"] for u in r["untraced"]] == ["1.07"]


def test_exploratory_must_be_labelled():
    bad = trace("Among Malay women the OR was 1.21 [A:x_q1].", ANALYSES, OUTPUTS)
    good = trace("In an exploratory analysis, among Malay women the OR was 1.21 [A:x_q1].", ANALYSES, OUTPUTS)
    assert bad["exploratory_not_labelled"] and not bad["ok"]
    assert good["ok"]


def test_unknown_reference_is_flagged():
    r = trace("The AUC was 0.64 [A:auc_model].", ANALYSES, OUTPUTS)
    assert r["unknown_analysis_refs"] == ["auc_model"] and not r["ok"]


def test_headings_and_conventions_ignored():
    text = "## 2. Methods\n\nWe used a significance level of p < 0.05 and report 95% confidence intervals."
    r = trace(text, ANALYSES, OUTPUTS)
    assert r["numbers_checked"] == 0 and r["ok"]


def test_hyphenated_ranges_are_not_negative_numbers():
    r = trace("OR 3.65 (0.94-14.19) [A:primary].", ANALYSES, OUTPUTS)
    assert r["ok"] and r["numbers_checked"] == 3
