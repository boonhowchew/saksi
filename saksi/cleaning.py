# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Data preparation: a fixed set of cleaning steps, proposed by the Analysis Agent from the Data Curator's findings,
chosen by the authors, and applied by this code (never by model-written code) to the uploaded file.

Each step is a small, typed instruction that a reviewer can read and repeat. Applying them yields a new dataset,
stored beside the original (which is never changed), with a log of what each step did, counted by code.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

TYPES = {
    "drop_columns": "Remove columns (for example direct identifiers, or duplicate encodings of the same variable)",
    "missing_codes": "Read codes as missing values (for example #NULL!, 999, -9)",
    "valid_range": "Set values outside a plausible range to missing",
    "recode": "Recode values (for example 'Y'/'N' to 1/0, or spelling variants to one label)",
    "derive_threshold": "Derive a yes/no variable from a threshold (for example BMI above 27)",
    "derive_any": "Derive a yes/no variable that is yes if any of several yes/no variables is yes",
    "drop_duplicates": "Remove duplicated records",
    "exclude_missing": "Exclude records with no value for a variable the study requires (for example the outcome)",
}
OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,60}$")


def _numeric_if_possible(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s)
    except (ValueError, TypeError):
        return s


def validate(steps: list[dict[str, Any]], columns: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep the steps that are well formed and name real columns (or columns derived by an earlier step).
    Returns (steps, problems)."""
    have, ok, problems = set(columns), [], []
    for i, st in enumerate(steps or []):
        if not isinstance(st, dict) or st.get("type") not in TYPES:
            problems.append(f"step {i + 1}: unknown type {st.get('type') if isinstance(st, dict) else st!r}")
            continue
        t, sid = st["type"], str(st.get("id") or f"s{i + 1}")
        cols = [c for c in (st.get("columns") or ([st["column"]] if st.get("column") else [])) if c != "*"]
        unknown = [c for c in cols if c not in have]
        if unknown:
            problems.append(f"{sid}: not in the data: {', '.join(map(str, unknown))}")
            continue
        clean = {"id": sid, "type": t, "title": str(st.get("title") or TYPES[t])[:200],
                 "reason": str(st.get("reason") or "")[:400]}
        if t in ("drop_columns", "derive_any"):
            if not cols:
                problems.append(f"{sid}: no columns"); continue
            clean["columns"] = cols
        if t == "missing_codes":
            codes = [str(c) for c in st.get("codes") or []][:20]
            if not codes:
                problems.append(f"{sid}: no codes"); continue
            clean.update({"columns": cols or ["*"], "codes": codes})
        if t in ("valid_range", "recode", "derive_threshold", "exclude_missing"):
            if len(cols) != 1:
                problems.append(f"{sid}: needs exactly one column"); continue
            clean["column"] = cols[0]
        if t == "valid_range":
            lo, hi = st.get("min"), st.get("max")
            if not all(isinstance(x, (int, float)) or x is None for x in (lo, hi)) or (lo is None and hi is None):
                problems.append(f"{sid}: give a numeric min and/or max"); continue
            clean.update({"min": lo, "max": hi})
        if t == "recode":
            mapping = {str(k): v for k, v in (st.get("mapping") or {}).items()}
            if not mapping:
                problems.append(f"{sid}: no mapping"); continue
            clean["mapping"] = mapping
        if t == "derive_threshold":
            if st.get("op") not in OPS or not isinstance(st.get("value"), (int, float)):
                problems.append(f"{sid}: needs op (>, >=, <, <=) and a numeric value"); continue
            clean.update({"op": st["op"], "value": st["value"]})
        if t in ("derive_threshold", "derive_any"):
            new = str(st.get("new") or "")
            if not NAME.match(new) or new in have:
                problems.append(f"{sid}: the new column needs a new, simple name"); continue
            clean["new"] = new
            clean["label"] = str(st.get("label") or new)[:120]
            have.add(new)
        if t == "drop_duplicates":
            clean["columns"] = cols
        if t == "drop_columns":
            have -= set(cols)
        ok.append(clean)
    return ok, problems


def apply(df: pd.DataFrame, steps: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply validated steps in order. Returns the prepared data and, for each step, what it did (counted)."""
    df = df.copy()
    log = []
    for st in steps:
        t, n0 = st["type"], len(df)
        if t == "drop_columns":
            df = df.drop(columns=st["columns"])
            effect = f"removed {len(st['columns'])} column(s): {', '.join(st['columns'])}"
        elif t == "missing_codes":
            cols = list(df.columns) if st["columns"] == ["*"] else st["columns"]
            before = int(df[cols].isna().sum().sum())
            codes = set(st["codes"])
            for c in cols:
                df[c] = df[c].mask(df[c].astype(str).str.strip().isin(codes))
                df[c] = _numeric_if_possible(df[c])
            effect = f"{int(df[cols].isna().sum().sum()) - before} value(s) set to missing"
        elif t == "valid_range":
            c = st["column"]
            x = pd.to_numeric(df[c], errors="coerce")
            bad = ((st["min"] is not None) & (x < (st["min"] if st["min"] is not None else -np.inf))) | \
                  ((st["max"] is not None) & (x > (st["max"] if st["max"] is not None else np.inf)))
            df[c] = x.mask(bad)
            effect = f"{int(bad.sum())} value(s) outside {st['min']} to {st['max']} set to missing"
        elif t == "recode":
            c = st["column"]
            s = df[c].astype(str).str.strip()
            hit = s.isin(st["mapping"].keys())
            df.loc[hit, c] = s[hit].map(st["mapping"])
            df[c] = _numeric_if_possible(df[c])
            effect = f"{int(hit.sum())} value(s) recoded"
        elif t == "derive_threshold":
            x = pd.to_numeric(df[st["column"]], errors="coerce")
            df[st["new"]] = np.where(x.isna(), np.nan, OPS[st["op"]](x, st["value"]).astype(float))
            v = df[st["new"]]
            effect = f"new column {st['new']}: {int((v == 1).sum())} yes, {int((v == 0).sum())} no, {int(v.isna().sum())} missing"
        elif t == "derive_any":
            x = df[st["columns"]].apply(pd.to_numeric, errors="coerce")
            v = np.where(x.isna().all(axis=1), np.nan, (x == 1).any(axis=1).astype(float))
            df[st["new"]] = v
            v = df[st["new"]]
            effect = f"new column {st['new']}: {int((v == 1).sum())} yes, {int((v == 0).sum())} no, {int(v.isna().sum())} missing"
        elif t == "drop_duplicates":
            df = df.drop_duplicates(subset=st["columns"] or None)
            effect = f"{n0 - len(df)} duplicated record(s) removed"
        elif t == "exclude_missing":
            df = df[df[st["column"]].notna() & ~df[st["column"]].astype(str).str.strip().isin(["", "#NULL!", "."])]
            effect = f"{n0 - len(df)} record(s) without {st['column']} excluded"
        log.append({"id": st["id"], "type": t, "title": st["title"], "reason": st.get("reason", ""), "effect": effect,
                    "rows_after": int(len(df)), "columns_after": int(df.shape[1])})
    return df, log
