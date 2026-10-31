# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Deterministic dataset profile: the only view of the data a model ever sees.

Rules, enforced here in code rather than by prompt:
  * No row is ever included.
  * Category labels with fewer than SMALL_CELL records are pooled, not shown.
  * Columns that look like identifiers or free text show no values at all.
  * Dates show only their year range.
"""

from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd

SMALL_CELL = 5
MAX_CATEGORIES = 20
IDENTIFIER_NAME = re.compile(
    r"(^|_|\b)(name|nama|ic|nric|mykad|passport|phone|tel|mobile|email|e_mail|address|alamat|postcode|"
    r"dob|birth|mrn|rn|patient_id|hospital_no|record_no)($|_|\b)", re.I)


NA_CODES = ["#NULL!", "."]      # SPSS and SAS write these for missing values; pandas does not know them


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    if name.endswith(".tsv"):
        return pd.read_csv(io.BytesIO(data), sep="\t", na_values=NA_CODES)
    return pd.read_csv(io.BytesIO(data), na_values=NA_CODES)


def _r(x: Any) -> Any:
    try:
        return None if pd.isna(x) else round(float(x), 4)
    except (TypeError, ValueError):
        return None


def profile(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        name = str(c)
        nonnull = s.dropna()
        uniq = int(nonnull.nunique())
        col: dict[str, Any] = {"name": name, "missing": int(s.isna().sum()), "unique": uniq,
                               "pandas_dtype": str(s.dtype)}
        id_like_name = bool(IDENTIFIER_NAME.search(name))
        if pd.api.types.is_bool_dtype(s) or (pd.api.types.is_numeric_dtype(s) and uniq <= 2):
            col["kind"] = "binary"
            col["levels"] = _levels(nonnull)
        elif pd.api.types.is_numeric_dtype(s):
            if id_like_name or (uniq == n and n > 20 and _is_integer_like(nonnull)):
                col["kind"] = "identifier_like"
            else:
                col["kind"] = "numeric"
                q = nonnull.quantile([0.25, 0.5, 0.75]) if len(nonnull) else pd.Series([None] * 3)
                col["summary"] = {"min": _r(nonnull.min()), "p25": _r(q.iloc[0]), "median": _r(q.iloc[1]),
                                  "p75": _r(q.iloc[2]), "max": _r(nonnull.max()), "mean": _r(nonnull.mean()),
                                  "sd": _r(nonnull.std())}
                col["integer_valued"] = _is_integer_like(nonnull)
                if uniq <= MAX_CATEGORIES:
                    col["levels"] = _levels(nonnull)
        else:
            dates = _as_dates(nonnull)
            if dates is not None:
                col["kind"] = "date"
                col["year_range"] = [int(dates.min().year), int(dates.max().year)] if len(dates) else None
            elif id_like_name or (uniq > MAX_CATEGORIES and uniq > 0.5 * max(len(nonnull), 1)):
                col["kind"] = "identifier_or_free_text"
            else:
                col["kind"] = "categorical"
                col["levels"] = _levels(nonnull)
        if id_like_name:
            col["warning"] = "name suggests a direct identifier; values withheld"
        cols.append(col)
    return {"rows": n, "columns": cols, "small_cell_threshold": SMALL_CELL,
            "note": "Aggregate profile only. No individual records are included."}


def _is_integer_like(s: pd.Series) -> bool:
    try:
        return bool(len(s)) and bool((s.astype(float) % 1 == 0).all())
    except (TypeError, ValueError):
        return False


def _levels(s: pd.Series) -> dict[str, Any]:
    counts = s.astype(str).value_counts()
    shown = {k: int(v) for k, v in counts.items() if v >= SMALL_CELL}
    hidden = counts[counts < SMALL_CELL]
    out: dict[str, Any] = {"counts": dict(list(shown.items())[:MAX_CATEGORIES])}
    if len(hidden):
        out["pooled_small_levels"] = {"levels": int(len(hidden)), "records": int(hidden.sum())}
    return out


def _as_dates(s: pd.Series):
    if not len(s) or not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return None
    sample = s.astype(str).head(200)
    if not sample.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() > 0.9:
        return None
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return parsed.dropna() if parsed.notna().mean() > 0.9 else None


def leaks_rows(profile_obj: dict[str, Any], df: pd.DataFrame, max_checks: int = 50) -> list[str]:
    """Self-check used in tests: identifier-like values must never appear in the profile text."""
    import json
    text = json.dumps(profile_obj)
    leaks = []
    for col in profile_obj["columns"]:
        if col["kind"] in {"identifier_like", "identifier_or_free_text"}:
            for v in df[col["name"]].dropna().astype(str).head(max_checks):
                if len(v) >= 4 and v in text:
                    leaks.append(f"{col['name']}={v}")
    return leaks
