# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""The standard sequence of analysis, applied to every quantitative study.

  1. Data quality          Saksi's own code (below): missing values, duplicates, outliers.
  2. Table 1               Saksi's own code (below): characteristics stratified by the outcome
                           (or the group the report compares), with the summary and test chosen
                           by rule from the data.
  3. Other descriptive analyses     \\
  4. Univariable analyses            > the protocol's analyses, written by the Analysis Agent,
  5. Multivariable analyses         /  run in this order
  6. Model diagnostics     after every multivariable model, computed by `saksi_checks`

The sequence follows the author's teaching materials (scientific-writing workshop modules,
Module 1; book on clinical epidemiology): audit the data, describe the sample (Table 1),
then infer, univariable before multivariable, and check the model before trusting it.
Steps 1 and 2 are fixed code so that every study is described the same way; the agent only
proposes which variables Table 1 holds and what it is stratified by, and the authors confirm
that with the plan. Outputs hold counts and summaries only, never a row.
"""

from __future__ import annotations

import json
from typing import Any

KIND_ORDER = {"descriptive": 0, "univariable": 1, "multivariable": 2, "other": 3}
STEPS = {"data_quality": "Data quality", "table1": "Table 1"}

QUALITY = r'''
import numpy as np
import pandas as pd

def analyse(data_path, seed):
    df = pd.read_csv(data_path)
    n = int(len(df))
    cols = []
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        e = {"column": str(c), "missing": miss, "missing_pct": round(100.0 * miss / n, 1) if n else None,
             "unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            x = s.dropna().astype(float)
            q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
            iqr = q3 - q1
            e.update({"min": float(x.min()), "max": float(x.max()),
                      "outliers_beyond_1_5_iqr": int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()),
                      "extreme_beyond_3_iqr": int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum())})
        cols.append(e)
    row_missing = df.isna().mean(axis=1)
    return {"n_rows": n, "n_columns": int(len(df.columns)), "duplicate_rows": int(df.duplicated().sum()),
            "rows_over_20pct_missing": int((row_missing > 0.2).sum()),
            "columns_with_missing": int(sum(1 for e in cols if e["missing"])),
            "constant_columns": [e["column"] for e in cols if e["unique"] <= 1],
            "columns": cols, "estimates": [],
            "notes": ["Outliers by the interquartile-range rule (1.5 and 3 times the IQR beyond the quartiles); "
                      "they are counted, not removed."]}
'''

TABLE1 = r'''
import numpy as np
import pandas as pd
from scipy import stats

SPEC = __SPEC__
MAX_LEVELS, MAX_CATEGORIES = 6, 15

def _num(x, d=4):
    x = float(x)
    return None if np.isnan(x) else round(x, d)

def _summary(x, normal):
    x = x.dropna().astype(float)
    if not len(x):
        return {"n": 0}
    if normal:
        return {"n": int(len(x)), "mean": _num(x.mean()), "sd": _num(x.std())}
    return {"n": int(len(x)), "median": _num(x.median()), "q1": _num(x.quantile(0.25)), "q3": _num(x.quantile(0.75))}

def analyse(data_path, seed):
    df = pd.read_csv(data_path)
    by = SPEC.get("stratify_by")
    notes = []
    if by and by not in df.columns:
        return {"error": f"the stratifying column {by!r} is not in the data"}
    if by and df[by].nunique() > MAX_LEVELS:
        notes.append(f"{by} has more than {MAX_LEVELS} values, so Table 1 is not stratified")
        by = None
    d = df[df[by].notna()] if by else df
    levels = sorted(d[by].unique().tolist(), key=lambda v: (str(type(v)), v)) if by else []
    groups = [{"level": str(l), "n": int((d[by] == l).sum())} for l in levels]
    rows = []
    for v in SPEC.get("variables") or []:
        if v not in d.columns:
            rows.append({"variable": v, "error": "not in the data"})
            continue
        if v == by:
            continue
        s = d[v]
        miss = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            x = s.dropna().astype(float)
            sw = stats.shapiro(x.sample(min(len(x), 5000), random_state=seed)).pvalue if len(x) >= 3 else None
            skew = float(stats.skew(x)) if len(x) > 2 else None
            # Shapiro-Wilk rejects trivial departures in large samples, so above 300 use |skewness| < 1.
            normal = bool((sw is not None and sw >= 0.05) or (len(x) > 300 and skew is not None and abs(skew) < 1))
            row = {"variable": v, "type": "continuous", "summary": "mean (SD)" if normal else "median (IQR)",
                   "normality": {"shapiro_wilk_p": _num(sw) if sw is not None else None, "skewness": _num(skew, 3) if skew is not None else None},
                   "overall": _summary(x, normal), "missing": miss}
            if by:
                samples = [d.loc[d[by] == l, v].dropna().astype(float) for l in levels]
                row["by_group"] = {str(l): _summary(d.loc[d[by] == l, v], normal) for l in levels}
                if all(len(z) > 1 for z in samples) and len(samples) >= 2:
                    if normal and len(samples) == 2:
                        lev = stats.levene(*samples).pvalue
                        row["levene_p"] = _num(lev)
                        row["test"] = "Student t-test" if lev >= 0.05 else "Welch t-test"
                        row["p_value"] = _num(stats.ttest_ind(*samples, equal_var=bool(lev >= 0.05)).pvalue, 6)
                    elif normal:
                        row["test"], row["p_value"] = "one-way ANOVA", _num(stats.f_oneway(*samples).pvalue, 6)
                    elif len(samples) == 2:
                        row["test"], row["p_value"] = "Mann-Whitney U test", _num(stats.mannwhitneyu(*samples).pvalue, 6)
                    else:
                        row["test"], row["p_value"] = "Kruskal-Wallis test", _num(stats.kruskal(*samples).pvalue, 6)
        else:
            cats = s.dropna()
            if cats.nunique() > MAX_CATEGORIES:
                rows.append({"variable": v, "type": "categorical", "error": f"more than {MAX_CATEGORIES} categories"})
                continue
            order = sorted(cats.unique().tolist(), key=lambda c: (str(type(c)), c))
            row = {"variable": v, "type": "categorical", "summary": "n (%)", "missing": miss, "categories": []}
            for c in order:
                k = int((cats == c).sum())
                cat = {"level": str(c), "overall": {"n": k, "pct": round(100.0 * k / len(cats), 1)}}
                if by:
                    cat["by_group"] = {}
                    for l in levels:
                        g = d.loc[(d[by] == l) & d[v].notna(), v]
                        kk = int((g == c).sum())
                        cat["by_group"][str(l)] = {"n": kk, "pct": round(100.0 * kk / len(g), 1) if len(g) else None}
                row["categories"].append(cat)
            if by and len(order) > 1:
                tab = pd.crosstab(d[v], d[by])
                expected = stats.contingency.expected_freq(tab.values)
                if tab.shape == (2, 2) and (expected < 5).any():
                    row["test"], row["p_value"] = "Fisher's exact test", _num(stats.fisher_exact(tab.values).pvalue, 6)
                else:
                    chi = stats.chi2_contingency(tab.values)
                    row["test"], row["p_value"] = "chi-square test", _num(chi.pvalue, 6)
                    if (expected < 5).mean() > 0.2:
                        row["test_note"] = "more than 20% of expected counts are below 5; read with care"
        rows.append(row)
    return {"stratify_by": by, "groups": groups, "n_analysed": int(len(d)), "n_excluded": int(len(df) - len(d)),
            "exclusion_reason": f"missing {by}" if by else "", "rows": rows, "estimates": [], "notes": notes +
            ["Continuous variables: mean (SD) when approximately normal (Shapiro-Wilk p >= 0.05, or |skewness| < 1 "
             "above 300 values), otherwise median (IQR); compared by t-test (Welch when Levene p < 0.05) or "
             "Mann-Whitney / ANOVA or Kruskal-Wallis. Categorical: n (%), chi-square, or Fisher's exact test when a "
             "2 x 2 table has an expected count below 5."]}
'''


def position(a: dict[str, Any]) -> tuple[int, int]:
    """Reporting order: data quality, Table 1, descriptive, univariable, multivariable, other, post hoc.
    A redone analysis keeps the place of the one it replaced (order_seq)."""
    at = a.get("order_seq", a.get("seq", 0))
    if a.get("standard"):
        return (0, 0 if a["standard"] == "data_quality" else 1)
    if a.get("basis") == "post hoc":
        return (6, at)
    return (2 + KIND_ORDER.get(a.get("kind"), 3), at)


def code(step: str, table1: dict[str, Any] | None = None) -> str:
    """Saksi's own code for a standard step."""
    if step == "data_quality":
        return QUALITY
    if step == "table1":
        spec = {"stratify_by": (table1 or {}).get("stratify_by"), "variables": list((table1 or {}).get("variables") or [])}
        return TABLE1.replace("__SPEC__", json.dumps(spec))
    raise ValueError(f"no standard step {step!r}")


def clean_table1(raw: Any) -> dict[str, Any] | None:
    """The plan's Table 1 specification, tidied: a stratifying column (or none) and up to 40 variables."""
    if not isinstance(raw, dict):
        return None
    by = raw.get("stratify_by")
    vs = [str(v) for v in (raw.get("variables") or []) if isinstance(v, str) and v.strip()][:40]
    if not vs:
        return None
    return {"stratify_by": str(by) if isinstance(by, str) and by.strip() else None, "variables": vs,
            "why": str(raw.get("why") or "")[:300]}
