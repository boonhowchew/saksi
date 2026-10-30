# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Saksi's proactive suggestions on the Results: fixed rules from epidemiology and biostatistics, applied by code
to the recorded results, so the same results always get the same advice.

Each suggestion says what was found, why it matters, and, where one exists, the action that would address it
(a sensitivity analysis as an exploratory variant, a redo to complete what the standard sequence requires, a
rewritten text, Table 1). The authors decide; nothing runs until they ask.
"""

from __future__ import annotations

import re
from typing import Any

from .blobs import BlobStore

EPV_MIN, VIF_MAX, MISSING_PCT, C_LOW = 10, 5, 5.0, 0.7
PREDICTION = re.compile(r"predict|discriminat|\bauc\b|risk score|validat|c-statistic", re.I)


def _variant_exists(s: Any, aid: str, prompt: str) -> bool:
    return any((a["spec"].get("based_on") or {}).get("analysis_id") == aid and a["spec"].get("author_request") == prompt
               for a in s.analyses.values() if not a.get("withdrawn"))


def suggestions(s: Any, blobs: BlobStore) -> list[dict[str, Any]]:
    from .tables import build
    from .workflow import current
    cur = current(s)
    out: list[dict[str, Any]] = []

    def add(key, aid, title, why, action=None, kind="action"):
        if action and action.get("type") == "variant" and _variant_exists(s, aid, action["instruction"]):
            return
        out.append({"id": key, "analysis_id": aid, "kind": kind if action else "note", "title": title, "why": why,
                    "action": action})

    if s.plan_locked is None:
        return out
    # Table 1 is the first result of every quantitative paper.
    if not any(a.get("standard") == "table1" for a in cur.values()):
        add("table1", None, "Build Table 1",
            "Every Results section opens with the characteristics of the participants, usually by the outcome.",
            {"type": "table1"})
    quality = next((blobs.get_json(a["outputs_sha256"]) for a in cur.values() if a.get("standard") == "data_quality"), None)
    missing = {c["column"]: c.get("missing_pct") or 0 for c in (quality or {}).get("columns", [])}
    for t in build(s, blobs, with_duplicates=True):
        if t.get("duplicate_of"):
            a = s.analyses.get(t["analysis_id"]) or {}
            add(f"dup:{t['analysis_id']}", t["analysis_id"], f"Same results as Table {t['duplicate_of']['number']}",
                f"{t['analysis_id']} gives exactly the same table as {t['duplicate_of']['analysis_id']}, so it adds nothing "
                "to the paper.",
                {"type": "leave_out", "analysis_id": t["analysis_id"],
                 "instruction": f"Same results as {t['duplicate_of']['analysis_id']}"} if a.get("basis") == "post hoc" else None)
    for aid, a in sorted(cur.items(), key=lambda kv: kv[1]["seq"]):
        if a.get("standard"):
            continue
        out_ = blobs.get_json(a["outputs_sha256"])
        title = a["spec"].get("title") or aid
        est = out_.get("estimates") or []
        if not a.get("description"):
            add(f"text:{aid}", aid, f"Write the text for “{title}”",
                "Each result needs a passage that reports it.", {"type": "rewrite_text", "analysis_id": aid, "instruction": ""})
        elif not (a["description"].get("checks") or {}).get("ok"):
            add(f"text:{aid}", aid, f"The text for “{title}” has numbers not in the results",
                "Every number in the text must come from the computed results.",
                {"type": "rewrite_text", "analysis_id": aid, "instruction": "Use only numbers that are in the results."})
        if a.get("kind") != "multivariable":
            continue
        d = out_.get("diagnostics") if isinstance(out_.get("diagnostics"), dict) else None
        if d is None:
            add(f"diag:{aid}", aid, f"Model diagnostics missing for “{title}”",
                "A multivariable model is not trusted until it is checked: events per parameter, collinearity, "
                "linearity, influential observations, fit, discrimination and calibration.",
                {"type": "redo", "analysis_id": aid, "instruction": "Add Saksi's standard model diagnostics (saksi_checks)."})
            continue
        if est and not any(re.search(r"crude|unadjusted", str(e.get("measure", "")), re.I) for e in est):
            add(f"crude:{aid}", aid, f"Crude estimates missing for “{title}”",
                "Reporting each variable's crude (unadjusted) estimate beside the adjusted one shows how much "
                "adjustment changed it, as the standard sequence requires.",
                {"type": "redo", "analysis_id": aid,
                 "instruction": "Report the crude (unadjusted) estimate of each independent variable beside its adjusted estimate."})
        asm, fit, infl = d.get("assumptions") or {}, d.get("fit") or {}, d.get("influence") or {}
        epv = asm.get("events_per_parameter")
        if isinstance(epv, (int, float)) and epv < EPV_MIN:
            add(f"epv:{aid}", aid, f"Few outcome events per predictor parameter ({epv})",
                f"Below about {EPV_MIN}, logistic regression estimates are biased away from the null and unstable. "
                "Firth's penalised likelihood reduces this small-sample bias.",
                {"type": "variant", "analysis_id": aid,
                 "instruction": "As a sensitivity analysis, refit this model with Firth's bias-reduced penalised logistic "
                                "regression (implement the Jeffreys-prior penalty with numpy), and report its estimates with "
                                "profile or Wald 95% CIs beside the original ones."})
        vif = (asm.get("collinearity") or {}).get("vif") or {}
        high = [k for k, v in vif.items() if isinstance(v, (int, float)) and v > VIF_MAX]
        if high:
            add(f"vif:{aid}", aid, f"Collinearity: {', '.join(high)}",
                f"A variance inflation factor above {VIF_MAX} inflates standard errors and makes estimates unstable.",
                {"type": "variant", "analysis_id": aid,
                 "instruction": f"As a sensitivity analysis, refit without the collinear predictor(s) {', '.join(high)} "
                                "(keep the exposure), and compare the estimates."})
        bt = asm.get("linearity_box_tidwell") or {}
        nonlin = [k for k, v in bt.items() if isinstance(v, dict) and v.get("nonlinear")]
        if nonlin:
            add(f"lin:{aid}", aid, f"Possibly non-linear: {', '.join(nonlin)}",
                "The Box-Tidwell test suggests the log-odds may not be linear in these variables, which can bias estimates.",
                {"type": "variant", "analysis_id": aid,
                 "instruction": f"As a sensitivity analysis, model {', '.join(nonlin)} with restricted cubic splines "
                                "(3 knots, patsy cr()), and compare the estimates and fit."})
        if (infl.get("cooks_above_1") or 0) > 0 or (infl.get("cooks_above_4_over_n") or 0) > 0.05 * (d.get("n") or 1e9):
            add(f"infl:{aid}", aid, "Influential observations",
                f"{infl.get('cooks_above_4_over_n')} observations have Cook's distance above 4/n. Results should not depend "
                "on a few records.",
                {"type": "variant", "analysis_id": aid,
                 "instruction": "As a sensitivity analysis, refit excluding observations with Cook's distance above 4/n, "
                                "report how many were excluded, and compare the estimates."})
        c = fit.get("c_statistic")
        text = " ".join(str(a["spec"].get(k) or "") for k in ("title", "objective", "method"))
        if isinstance(c, (int, float)) and PREDICTION.search(text):
            add(f"boot:{aid}", aid, "Apparent performance only",
                "Discrimination and calibration measured on the data the model was fitted to are optimistic. "
                "Bootstrap internal validation estimates the optimism.",
                {"type": "variant", "analysis_id": aid,
                 "instruction": "Estimate optimism-corrected C-statistic and calibration slope by bootstrap internal "
                                "validation (200 resamples, using seed), and report apparent, optimism and corrected values."})
        elif isinstance(c, (int, float)) and c < C_LOW:
            add(f"c:{aid}", aid, f"Limited discrimination (C-statistic {c:.2f})",
                "The model separates those with and without the outcome only modestly; say so when reporting it.")
        used = [x.get("column") for x in [a["spec"].get("outcome") or {}] + list(a["spec"].get("exposures") or [])
                + list(a["spec"].get("covariates") or []) if isinstance(x, dict) and x.get("column")]
        worst = sorted(((missing.get(col) or 0, col) for col in used), reverse=True)
        if worst and worst[0][0] > MISSING_PCT and "imput" not in str(a["spec"].get("missing_data", "")).lower():
            add(f"mi:{aid}", aid, f"Missing data: {worst[0][1]} {worst[0][0]}% missing",
                "A complete-case analysis is unbiased only if data are missing completely at random (or missingness does "
                "not depend on the outcome). Multiple imputation tests how much this matters.",
                {"type": "variant", "analysis_id": aid,
                 "instruction": "As a sensitivity analysis, handle missing values by multiple imputation by chained "
                                "equations (statsmodels MICE, 20 imputations, using seed), pool with Rubin's rules, and "
                                "compare with the complete-case estimates."})
    post_hoc = [a for a in cur.values() if a.get("basis") == "post hoc"]
    if len(post_hoc) >= 3:
        add("multiplicity", None, f"{len(post_hoc)} exploratory analyses added after the plan",
            "With several exploratory analyses some findings will be due to chance. Report them as hypotheses and state "
            "that no adjustment for multiple comparisons was made (or make one).")
    return out
