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


def test_heading_directly_followed_by_text_is_still_checked():
    """Regression: the live Writing Agent put text on the line right after '## Results'.
    The tracer skipped those paragraphs whole and reported 0 numbers checked."""
    text = "## Results\nBooking BMI: adjusted OR 2.15 (95% CI 1.40 to 3.30) [A:primary].\n\n## Discussion\nOR 1.07 [A:primary]."
    r = trace(text, ANALYSES, OUTPUTS)
    assert r["numbers_checked"] == 4
    assert {u["number"] for u in r["untraced"]} == {"2.15", "1.40", "3.30"}


def test_numbers_in_result_notes_are_traceable():
    outputs = {"primary": {"estimates": [], "notes": ["Number of GDM events in analysed set: 18."]}}
    r = trace("There were 18 events [A:primary]. There were 19 events [A:primary].", ANALYSES, outputs)
    assert [u["number"] for u in r["untraced"]] == ["19"]


def test_checker_never_passes_when_it_checked_nothing_but_numbers_exist(monkeypatch):
    import saksi.tracing as t
    monkeypatch.setattr(t, "NUM", __import__("re").compile(r"(?!x)x(<)?(x)(%)?"))  # simulate a parser that sees nothing
    r = t.trace("The OR was 1.07 [A:primary].", ANALYSES, OUTPUTS)
    assert r["numbers_checked"] == 0 and r["nothing_checked_but_numbers_present"] and not r["ok"]


def test_live_writing_agent_draft_2026_10_29():
    """The first draft written by the live Writing Agent (Muse Spark 1.3) on synthetic data.
    Results are rebuilt from that run's printed output; the p-value is recomputed from the
    same deterministic synthetic data. Before the 29 Oct fix this reported 0 numbers checked."""
    from pathlib import Path
    import statsmodels.formula.api as smf
    from saksi import demo_data
    df = demo_data.make().dropna(subset=["gdm", "booking_bmi", "age", "parity", "previous_gdm"])
    p_bmi = float(smf.logit("gdm ~ booking_bmi + age + parity + previous_gdm", data=df).fit(disp=0)
                  .pvalues["booking_bmi"])
    draft = (Path(__file__).parent / "fixtures" / "live_draft_2026-10-29.md").read_text()
    outputs = {
        "primary_bmi_gdm_adjusted": {"n_analysed": 388, "n_excluded": 12, "estimates": [
            {"estimate": 1.0673717768873334, "ci_lower": 0.9607527632500208, "ci_upper": 1.1858227773830956,
             "p_value": p_bmi}], "notes": ["Number of GDM events in analysed set: 18."]},
        "secondary_discrimination_auc": {"n_analysed": 388, "n_excluded": 12, "estimates": [
            {"estimate": 0.6485735735735736, "ci_lower": 0.506280008662886, "ci_upper": 0.7908671384842612}],
            "notes": ["Analysed n=388 with 18 GDM events; excluded n=12."]}}
    labels = {k: {"label": "confirmatory"} for k in outputs}
    r = trace(draft, labels, outputs, dataset_counts=[400])
    assert r["numbers_checked"] == 22 and r["ok"]
    # And the same draft with one number changed is caught.
    bad = trace(draft.replace("1.07 (95% confidence interval 0.96", "1.17 (95% confidence interval 0.96", 1),
                labels, outputs, dataset_counts=[400])
    assert [u["number"] for u in bad["untraced"]] == ["1.17"]
