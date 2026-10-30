# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Standard model diagnostics, supplied by Saksi inside the analysis sandbox as `saksi_checks`.

Analysis code must call one of these on its last (and every) multivariable model and return
the result under "diagnostics". The same checks are then computed the same way for every study.

The checks and thresholds follow the author's teaching materials (scientific-writing workshop
modules, Module 1, "Logistic Regression Model Diagnostics"; and his book on clinical
epidemiology), restated here in our own words:
  fit         likelihood-ratio test against the null model; C-statistic; calibration (slope,
              intercept, observed against expected by tenths of risk; Hosmer-Lemeshow with its
              limits stated); pseudo-R2 only as a secondary indicator
  influence   Pearson residuals beyond +/-2; deviance residuals outside (-2, 2); Cook's distance
              above 4/n (and above 1); leverage; DfBeta above 2/sqrt(n)
  assumptions multicollinearity (correlations above 0.8; VIF above 10, or above 2.5 by the
              stricter standard; tolerance below 0.1); linearity of the logit for continuous
              predictors (Box-Tidwell); independence (a design question: clustered data need
              GEE or mixed models); at least 10 outcome events per predictor parameter
Only counts and summaries are returned, never a row of data.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

VIF_LIMIT, VIF_STRICT, TOL_LIMIT, CORR_LIMIT, EPV_MIN = 10.0, 2.5, 0.1, 0.8, 10


def _f(x, d=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, d)


def _design(result):
    """Predictor matrix (without the constant), its names, and the outcome."""
    model = result.model
    X = np.asarray(model.exog, dtype=float)
    names = list(getattr(model, "exog_names", None) or [f"x{i}" for i in range(X.shape[1])])
    y = np.asarray(model.endog, dtype=float)
    keep = [i for i, n in enumerate(names) if not (n.lower() in ("const", "intercept") or np.allclose(X[:, i], 1.0))]
    return X, names, keep, y


def _collinearity(X, names, keep):
    out = {"vif": {}, "tolerance": {}, "high_correlations": []}
    if len(keep) < 2:
        return out
    Z = X[:, keep]
    with np.errstate(all="ignore"):
        corr = np.corrcoef(Z, rowvar=False)
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            r = corr[a, b]
            if np.isfinite(r) and abs(r) > CORR_LIMIT:
                out["high_correlations"].append({"a": names[keep[a]], "b": names[keep[b]], "r": _f(r, 3)})
    ones = np.ones((Z.shape[0], 1))
    for j in range(len(keep)):
        others = np.hstack([ones, np.delete(Z, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        resid = Z[:, j] - others @ beta
        ss_tot = float(((Z[:, j] - Z[:, j].mean()) ** 2).sum())
        r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
        tol = max(1 - r2, 1e-12)
        out["vif"][names[keep[j]]] = _f(1 / tol, 3)
        out["tolerance"][names[keep[j]]] = _f(tol, 3)
    return out


def _box_tidwell(y, X, names, keep):
    """Linearity of the logit for continuous, strictly positive predictors: add x*ln(x)."""
    import statsmodels.api as sm
    out = {}
    for j in keep:
        x = X[:, j]
        if len(np.unique(x)) <= 10 or np.any(x <= 0):
            continue
        Xb = np.column_stack([X, x * np.log(x)])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = sm.GLM(y, Xb, family=sm.families.Binomial()).fit()
            out[names[j]] = {"p_value": _f(r.pvalues[-1], 4), "nonlinear": bool(r.pvalues[-1] < 0.05)}
        except Exception as e:  # report, don't hide
            out[names[j]] = {"error": str(e)[:120]}
    return out


def _calibration(y, p):
    order = np.argsort(p)
    groups = np.array_split(order, 10)
    deciles, hl = [], 0.0
    for g in groups:
        if len(g) == 0:
            continue
        obs, exp, n = float(y[g].sum()), float(p[g].sum()), len(g)
        deciles.append({"n": n, "mean_predicted": _f(exp / n), "observed_proportion": _f(obs / n)})
        pbar = exp / n
        if 0 < pbar < 1:
            hl += (obs - exp) ** 2 / (n * pbar * (1 - pbar))
    from scipy import stats
    import statsmodels.api as sm
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    slope = intercept = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        intercept, slope = _f(c.params[0], 3), _f(c.params[1], 3)
    except Exception:
        pass
    return {"slope": slope, "intercept": intercept, "deciles": deciles,
            "apparent": "computed on the data the model was fitted to, so a slope near 1 and an intercept near 0 "
                        "are expected; they say little about performance in new patients, which needs internal "
                        "(bootstrap) or external validation",
            "hosmer_lemeshow": {"chi2": _f(hl, 3), "df": max(len(deciles) - 2, 1),
                                "p_value": _f(stats.chi2.sf(hl, max(len(deciles) - 2, 1)), 4),
                                "note": "underpowered in small samples and oversensitive in large ones; "
                                        "read with the calibration plot"}}


def logistic_checks(result, clustered_by: str | None = None) -> dict:
    """Diagnostics for a fitted binary logistic model (statsmodels Logit or GLM Binomial results)."""
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    X, names, keep, y = _design(result)
    n, events = int(len(y)), int(y.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        infl = glm.get_influence()
    p = np.asarray(glm.fittedvalues, dtype=float)
    params = len(keep)
    epv = min(events, n - events) / params if params else None
    null_ll = float(sm.GLM(y, np.ones((n, 1)), family=sm.families.Binomial()).fit().llf)
    lr = 2 * (float(glm.llf) - null_ll)
    from scipy import stats
    cooks = np.asarray(infl.cooks_distance[0], dtype=float)
    dfb = np.abs(np.asarray(infl.dfbetas, dtype=float))
    pearson, deviance = np.asarray(glm.resid_pearson), np.asarray(glm.resid_deviance)
    lev = np.asarray(infl.hat_matrix_diag, dtype=float)
    dfb_cut = 2 / math.sqrt(n)
    out = {
        "model": "logistic", "n": n, "events": events, "predictor_parameters": params,
        "fit": {"likelihood_ratio": {"chi2": _f(lr, 3), "df": params, "p_value": _f(stats.chi2.sf(lr, params), 6)},
                "c_statistic": _f(roc_auc_score(y, p), 4),
                "pseudo_r2_mcfadden": _f(1 - float(glm.llf) / null_ll if null_ll else None, 4),
                "pseudo_r2_note": "secondary indicator only; it is not variance explained",
                "calibration": _calibration(y, p)},
        "influence": {"pearson_residuals_beyond_2": int((np.abs(pearson) > 2).sum()),
                      "deviance_residuals_outside_2": int((np.abs(deviance) > 2).sum()),
                      "cooks_distance_max": _f(cooks.max(), 4), "cooks_above_4_over_n": int((cooks > 4 / n).sum()),
                      "cooks_above_1": int((cooks > 1).sum()), "leverage_max": _f(lev.max(), 4),
                      "leverage_mean": _f(lev.mean(), 4), "dfbeta_threshold": _f(dfb_cut, 4),
                      "dfbeta_above_threshold": {names[j]: int((dfb[:, j] > dfb_cut).sum()) for j in keep}},
        "assumptions": {"events_per_parameter": _f(epv, 2), "collinearity": _collinearity(X, names, keep),
                        "linearity_box_tidwell": _box_tidwell(y, X, names, keep),
                        "independence": ("clustered by " + clustered_by + ": use GEE or a mixed model"
                                         if clustered_by else "assumed independent; state whether the design clusters "
                                         "observations (for example by clinic or repeated measures)")},
    }
    out["flags"] = _flags(out)
    return out


def linear_checks(result, clustered_by: str | None = None) -> dict:
    """Diagnostics for a fitted linear model (statsmodels OLS results)."""
    from scipy import stats
    from statsmodels.stats.diagnostic import het_breuschpagan
    X, names, keep, y = _design(result)
    n = int(len(y))
    resid = np.asarray(result.resid, dtype=float)
    infl = result.get_influence()
    cooks = np.asarray(infl.cooks_distance[0], dtype=float)
    try:
        bp = het_breuschpagan(resid, X)
        bp_out = {"lm": _f(bp[0], 3), "p_value": _f(bp[1], 4)}
    except Exception as e:
        bp_out = {"error": str(e)[:120]}
    sw = stats.shapiro(resid[:5000]) if n >= 3 else None
    out = {
        "model": "linear", "n": n, "predictor_parameters": len(keep),
        "fit": {"r2": _f(result.rsquared, 4), "adjusted_r2": _f(result.rsquared_adj, 4),
                "f_test": {"f": _f(result.fvalue, 3), "p_value": _f(result.f_pvalue, 6)}},
        "influence": {"cooks_distance_max": _f(cooks.max(), 4), "cooks_above_4_over_n": int((cooks > 4 / n).sum()),
                      "standardised_residuals_beyond_2": int((np.abs(infl.resid_studentized_internal) > 2).sum())},
        "assumptions": {"residual_normality_shapiro": {"w": _f(sw[0], 4), "p_value": _f(sw[1], 4)} if sw else None,
                        "homoscedasticity_breusch_pagan": bp_out, "collinearity": _collinearity(X, names, keep),
                        "observations_per_parameter": _f(n / max(len(keep), 1), 2),
                        "independence": ("clustered by " + clustered_by + ": use GEE or a mixed model"
                                         if clustered_by else "assumed independent; state whether the design clusters "
                                         "observations")},
    }
    out["flags"] = _flags(out)
    return out


def _flags(d: dict) -> list[str]:
    """Plain-language problems, by the thresholds above."""
    f, a, i, fit = [], d["assumptions"], d["influence"], d["fit"]
    epv = a.get("events_per_parameter")
    if epv is not None and epv < EPV_MIN:
        f.append(f"{epv} outcome events per predictor parameter, below the {EPV_MIN} recommended: risk of overfitting")
    col = a.get("collinearity") or {}
    for k, v in (col.get("vif") or {}).items():
        if v is not None and v > VIF_LIMIT:
            f.append(f"VIF {v} for {k} (above {VIF_LIMIT}): its estimate is unreliable")
        elif v is not None and v > VIF_STRICT:
            f.append(f"VIF {v} for {k} (above {VIF_STRICT}, the stricter standard)")
    for c in col.get("high_correlations") or []:
        f.append(f"{c['a']} and {c['b']} correlate at {c['r']}")
    for k, v in (a.get("linearity_box_tidwell") or {}).items():
        if v.get("nonlinear"):
            f.append(f"{k} may not be linear in the logit (Box-Tidwell p = {v['p_value']}): consider splines or categories")
    if i.get("cooks_above_1"):
        f.append(f"{i['cooks_above_1']} observations with Cook's distance above 1: check them, and report a sensitivity analysis")
    if d["model"] == "logistic":
        c = fit.get("c_statistic")
        if c is not None and c < 0.7:
            f.append(f"C-statistic {c}: limited discrimination")
        cal = fit.get("calibration") or {}
        if cal.get("slope") is not None and not 0.8 <= cal["slope"] <= 1.2:
            f.append(f"calibration slope {cal['slope']} (1 is ideal)")
    else:
        n = (a.get("residual_normality_shapiro") or {}).get("p_value")
        if n is not None and n < 0.05:
            f.append("residuals depart from normality (Shapiro-Wilk p < 0.05)")
        h = (a.get("homoscedasticity_breusch_pagan") or {}).get("p_value")
        if h is not None and h < 0.05:
            f.append("residual variance is not constant (Breusch-Pagan p < 0.05)")
    return f
