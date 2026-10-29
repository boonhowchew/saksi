# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

from saksi.figures import flow_svg, forest_svg, required_figure, saksi_figures
from saksi.tracing import trace

ORS = [{"name": "booking_bmi", "measure": "adjusted odds ratio", "estimate": 1.067, "ci_lower": 0.961, "ci_upper": 1.186},
       {"name": "previous_gdm", "measure": "adjusted odds ratio", "estimate": 3.65, "ci_lower": 0.94, "ci_upper": 14.19}]


def test_forest_plot_is_drawn_from_recorded_estimates_only():
    svg, measure = forest_svg(ORS)
    assert measure == "adjusted odds ratio" and "(log scale)" in svg
    assert "1.07 (0.96 to 1.19)" in svg and "3.65 (0.94 to 14.19)" in svg and "booking_bmi" in svg
    assert forest_svg(ORS) == (svg, measure)                       # deterministic
    assert forest_svg([{"name": "AUC", "measure": "AUC", "estimate": 0.65, "ci_lower": 0.51, "ci_upper": 0.79}]) is None
    assert forest_svg([{"name": "x<script>", "measure": "odds ratio", "estimate": 2, "ci_lower": 1, "ci_upper": 3}])[0] \
        .count("<script>") == 0                                   # text is escaped


def test_flow_diagram_from_counts():
    svg = flow_svg(400, {"n_analysed": 388, "n_events": 18, "exclusion_reason": "missing booking BMI"}, "GDM")
    for s in ("n = 400", "n = 12", "n = 388", "18 with GDM", "missing booking BMI"):
        assert s in svg
    assert flow_svg(400, {"estimates": []}) is None
    # Counts that do not add up are drawn as recorded and flagged on the figure.
    bad = flow_svg(400, {"n_analysed": 388, "n_excluded": 0})
    assert "Warning: counts do not add up: 400 records, but 0 excluded + 388 analysed = 388" in bad
    figs = saksi_figures({"title": "t"}, {"n_analysed": 388, "n_excluded": 0}, 400, "confirmatory", None)
    assert figs[0]["warning"].startswith("counts do not add up")


def test_same_flow_is_drawn_once_however_exclusions_are_worded(proj):
    from tests.conftest import to_analysis
    to_analysis(proj)
    for aid, reason in (("primary", "missing BMI"), ("sens1", "complete-case analysis: missing values")):
        out = proj.blobs.put_json({"n_analysed": 388, "n_excluded": 12, "n_events": 18, "exclusion_reason": reason})
        proj.complete_analysis("agent:analyst", aid, plan_analysis_id=aid, spec={"id": aid}, code_sha256="a" * 64,
                               outputs_sha256=out, execution={}, status="ok")
    flows = [f for f in proj.state().figures if f["name"] == "participant_flow"]
    assert [f["analysis_id"] for f in flows] == ["primary"]


def test_required_figure_is_chosen_by_rule():
    assert required_figure({"method": "AUC with 95% CI", "effect_measure": "AUC"})["name"] == "roc_curve"
    assert required_figure({"method": "Cox proportional hazards", "effect_measure": "hazard ratio"})["name"] == "kaplan_meier"
    assert required_figure({"method": "multivariable logistic regression", "effect_measure": "odds ratio"}) is None


def test_saksi_captions_say_exploratory_and_pass_the_number_check():
    figs = saksi_figures({"title": "BMI and GDM"}, {"estimates": ORS, "n_analysed": 388}, 400, "exploratory", "post hoc")
    assert [f["name"] for f in figs] == ["forest_plot", "participant_flow"]
    for f in figs:
        assert f["caption"].startswith("Exploratory (post hoc) analysis.")
        r = trace(f["caption"] + " [A:a]", {"a": {"label": "exploratory"}}, {"a": {}})
        assert r["ok"], r
