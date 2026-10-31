# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Reporting guidelines by study design (EQUATOR Network), as short reminders in Saksi's own words.

The design comes from the plan (proposed from the protocol, confirmed by the authors); the guideline follows by rule.
The Writing Agent is given the items for Methods, Results and Discussion; after each draft, a check lists each item
as reported, partly reported or missing, with a quotation from the draft that code verifies. The official checklists
are the authority; these lines are prompts, not their text.
"""

from __future__ import annotations

from typing import Any

DESIGNS = {
    "cross_sectional": "cross-sectional study", "cohort": "cohort study", "case_control": "case-control study",
    "rct": "randomised controlled trial", "diagnostic_accuracy": "diagnostic accuracy study",
    "prediction_model": "prediction model development or validation", "systematic_review": "systematic review",
    "qualitative": "qualitative study", "other": "other design",
}

_STROBE_COMMON = {
    "methods": [
        "Setting, locations and the periods of recruitment, exposure, follow-up and data collection",
        "Eligibility criteria and how participants were selected",
        "Clear definitions of outcomes, exposures, predictors, potential confounders and effect modifiers",
        "Data sources and how each variable was measured",
        "Efforts to address potential sources of bias",
        "How the study size was arrived at",
        "How quantitative variables were handled (grouping, transformations) and why",
        "All statistical methods, including those used to control for confounding",
        "Methods for subgroups, interactions and sensitivity analyses",
        "How missing data were addressed",
    ],
    "results": [
        "Numbers at each stage (eligible, included, analysed) and reasons for non-participation; a flow diagram",
        "Characteristics of participants and information on exposures and potential confounders",
        "Number of participants with missing data for each variable of interest",
        "Unadjusted estimates and confounder-adjusted estimates with their precision (95% CI), and which confounders were adjusted for",
        "Other analyses done (subgroups, interactions, sensitivity analyses)",
    ],
    "discussion": [
        "Key results with reference to the study objectives",
        "Limitations, including sources of potential bias or imprecision, with their direction and magnitude",
        "A cautious overall interpretation, considering objectives, limitations, multiplicity of analyses and other evidence",
        "Generalisability (external validity) of the results",
    ],
}

GUIDELINES: dict[str, dict[str, Any]] = {
    "cross_sectional": {"name": "STROBE (cross-sectional studies)", "items": {
        **_STROBE_COMMON, "methods": _STROBE_COMMON["methods"] + ["How the sampling strategy was taken into account, if relevant"]}},
    "cohort": {"name": "STROBE (cohort studies)", "items": {
        **_STROBE_COMMON,
        "methods": _STROBE_COMMON["methods"] + ["How loss to follow-up was addressed"],
        "results": _STROBE_COMMON["results"] + ["Follow-up time and the number of outcome events or summary measures over time"]}},
    "case_control": {"name": "STROBE (case-control studies)", "items": {
        **_STROBE_COMMON,
        "methods": _STROBE_COMMON["methods"] + ["Case ascertainment, control selection and the rationale; matching criteria and how matching was handled"],
        "results": _STROBE_COMMON["results"] + ["Numbers in each exposure category among cases and controls"]}},
    "rct": {"name": "CONSORT", "items": {
        "methods": ["Trial design and allocation ratio; important changes after it started, with reasons",
                    "Eligibility criteria, settings and locations",
                    "The interventions for each group, in enough detail to replicate",
                    "Pre-specified primary and secondary outcomes, how and when assessed",
                    "How the sample size was determined",
                    "Random sequence generation, allocation concealment and who implemented them",
                    "Who was blinded and how",
                    "Statistical methods for primary and secondary outcomes; additional analyses",
                    "Analysis population (intention-to-treat or other) and handling of missing outcome data"],
        "results": ["Participant flow for each group: randomised, received intervention, analysed; losses and exclusions with reasons",
                    "Dates of recruitment and follow-up",
                    "Baseline characteristics of each group (a table, without significance tests of baseline differences)",
                    "Numbers analysed in each group",
                    "For each outcome, results for each group and the estimated effect size with its 95% CI",
                    "For binary outcomes, both absolute and relative effect sizes",
                    "Other analyses, distinguishing pre-specified from exploratory",
                    "Important harms or unintended effects in each group"],
        "discussion": ["Limitations: sources of bias, imprecision and multiplicity of analyses",
                       "Generalisability", "Interpretation consistent with the results, balancing benefits and harms"]}},
    "diagnostic_accuracy": {"name": "STARD", "items": {
        "methods": ["Participants: eligibility, recruitment, setting and sampling", "The index test and the reference standard, in detail, and their rationale",
                    "Definition of and rationale for test positivity cut-offs, pre-specified or exploratory",
                    "Whether assessors were blinded to the other test's results",
                    "Methods for estimating diagnostic accuracy and their precision", "How indeterminate results and missing data were handled",
                    "Intended sample size and how it was determined"],
        "results": ["Flow of participants, with a diagram", "Baseline demographic and clinical characteristics",
                    "Distribution of severity of disease and alternative diagnoses", "Cross-tabulation of index test results by reference standard results",
                    "Estimates of diagnostic accuracy (sensitivity, specificity, predictive values, AUC) with 95% CIs",
                    "Any adverse events from the tests"],
        "discussion": ["Limitations, including sources of bias, statistical uncertainty and generalisability", "Implications for practice and the intended use of the test"]}},
    "prediction_model": {"name": "TRIPOD+AI", "items": {
        "methods": ["Source of data, setting and dates", "Eligibility criteria and treatments received",
                    "Outcome definition and how and when it was assessed", "Candidate predictors: definition, measurement and timing",
                    "How the sample size was arrived at, including events per candidate parameter",
                    "How missing data were handled", "Model type, predictor selection, and handling of continuous predictors",
                    "How model performance was measured (discrimination, calibration) and how internal validation was done",
                    "Any model updating or recalibration"],
        "results": ["Flow of participants and the number with the outcome", "Characteristics of the participants, including missing data",
                    "The full model (all coefficients, intercept or baseline) so it can be used", "Discrimination with 95% CI",
                    "Calibration (plot, slope and intercept)", "Apparent and optimism-corrected performance from internal validation"],
        "discussion": ["Limitations: non-representative sample, few events per predictor, missing data", "Interpretation of performance and of any overfitting",
                       "Intended use, applicability and next steps (external validation, impact)"]}},
    "systematic_review": {"name": "PRISMA 2020", "items": {
        "methods": ["Eligibility criteria", "Information sources and search strategy", "Selection and data collection processes",
                    "Risk of bias assessment", "Effect measures and synthesis methods", "Assessment of reporting bias and certainty of evidence"],
        "results": ["Study selection with a flow diagram", "Characteristics of included studies", "Risk of bias in each study",
                    "Results of individual studies and of syntheses, with 95% CIs and heterogeneity", "Certainty of the evidence"],
        "discussion": ["Interpretation in the context of other evidence", "Limitations of the evidence and of the review processes",
                       "Implications for practice, policy and research"]}},
    "qualitative": {"name": "SRQR / COREQ", "items": {
        "methods": ["Qualitative approach and research paradigm", "Researcher characteristics and reflexivity",
                    "Context, sampling strategy and its rationale", "Data collection methods and instruments",
                    "Data processing and analysis; techniques to enhance trustworthiness"],
        "results": ["Synthesis and interpretation: themes, with supporting quotations", "Links to empirical data"],
        "discussion": ["Integration with prior work, implications and transferability", "Limitations"]}},
}
GUIDELINES["other"] = {"name": "STROBE (general observational)", "items": _STROBE_COMMON}


def for_design(design: str | None) -> dict[str, Any]:
    return GUIDELINES.get(design or "other", GUIDELINES["other"])


def item_list(design: str | None) -> list[dict[str, str]]:
    g = for_design(design)
    return [{"id": f"{sec[0].upper()}{i + 1}", "section": sec, "item": it}
            for sec in ("methods", "results", "discussion") for i, it in enumerate(g["items"].get(sec, []))]
