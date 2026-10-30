# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Publication-ready tables, built by code from the recorded results. No model touches them.

Table 1 (characteristics, stratified) comes from the standard step; every other analysis with
estimates gets a results table, crude and adjusted side by side when both are reported. Tables are
numbered in reporting order (the standard sequence), the same numbers the Writing Agent is given.
Formatting follows common journal practice: three horizontal rules, no vertical lines, estimates
with 95% CI as "x (a to b)", P values to three decimals or "<0.001", footnotes for tests,
abbreviations, the sample and the label (confirmatory or exploratory).

The page shows the same table objects; `to_docx` gives a Word file the authors can paste from.
"""

from __future__ import annotations

import io
import re
from typing import Any

from .blobs import BlobStore
from .standard import position

ABBREVIATIONS = [("odds ratio", "OR"), ("hazard ratio", "HR"), ("risk ratio", "RR"), ("relative risk", "RR"),
                 ("incidence rate ratio", "IRR"), ("rate ratio", "IRR"), ("mean difference", "MD"),
                 ("area under the curve", "AUC"), ("area under the roc curve", "AUC")]
LETTERS = "abcdefghij"


def _p(p: Any) -> str:
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _x(v: Any, d: int | None = None) -> str:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "" if v is None else str(v)
    if d is None:
        d = 3 if v != 0 and abs(v) < 0.1 else 2
    return f"{v:.{d}f}"


def _measure(m: str) -> tuple[str, list[tuple[str, str]]]:
    """'adjusted odds ratio' -> ('Adjusted OR', [('OR', 'odds ratio')])."""
    low, used = (m or "").strip().lower(), []
    out = m.strip() if m else "Estimate"
    for full, short in ABBREVIATIONS:
        if full in low:
            out = re.sub(full, short, out, flags=re.I)
            used.append((short, full))
            break
    if low in ("auc", "c-statistic"):
        used.append(("AUC", "area under the receiver operating characteristic curve"))
    return out[:1].upper() + out[1:], used


def _labels(dictionary: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {d["column"]: d for d in dictionary or [] if isinstance(d, dict) and d.get("column")}


def _name(col: str, labels: dict[str, dict[str, Any]], continuous: bool = False) -> str:
    d = labels.get(col) or {}
    name = str(d.get("label") or col)
    units = str(d.get("units_or_coding") or "")
    if continuous and units and not re.search(r"\d\s*/\s*\d|=", units):
        name += f", {units}"
    return name


def _label_note(a: dict[str, Any]) -> str:
    if a.get("label") == "confirmatory":
        return ("Primary (confirmatory) analysis; named primary by the authors, as stated in Methods."
                if a.get("primary_basis") == "authors" else "Primary (confirmatory) analysis, as pre-specified.")
    return f"Exploratory analysis ({a.get('basis') or 'pre-specified'})."


def _table1(aid: str, a: dict[str, Any], t: dict[str, Any], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by, groups = t.get("stratify_by"), t.get("groups") or []
    by_name = _name(by, labels) if by else ""
    cols = ["Characteristic", f"Overall\n(n = {t.get('n_analysed')})"] + \
        [f"{by_name}: {g['level']}\n(n = {g['n']})" for g in groups] + (["P value"] if groups else [])
    tests: dict[str, str] = {}

    def mark(r: dict[str, Any]) -> str:
        key = r.get("test") or ""
        if not key:
            return ""
        if r.get("test_note"):
            key += " (more than 20% of expected counts below 5; read with care)"
        return tests.setdefault(key, LETTERS[len(tests) % len(LETTERS)])

    def cont(x: dict[str, Any] | None, summary: str) -> str:
        if not x or not x.get("n"):
            return ""
        if summary == "mean (SD)":
            return f"{_x(x.get('mean'), 1)} ({_x(x.get('sd'), 1)})"
        return f"{_x(x.get('median'), 1)} ({_x(x.get('q1'), 1)} to {_x(x.get('q3'), 1)})"

    def npct(x: dict[str, Any] | None) -> str:
        return f"{x['n']} ({_x(x.get('pct'), 1)})" if x and x.get("pct") is not None else (str(x["n"]) if x else "")

    rows, missing = [], []
    for r in t.get("rows") or []:
        if r.get("error"):
            continue
        v = r["variable"]
        if r.get("missing"):
            missing.append(f"{_name(v, labels)} {r['missing']}")
        p = [_p(r.get("p_value"))] if groups else []
        m = {len(cols) - 1: mark(r)} if groups and r.get("p_value") is not None else {}
        if r.get("type") == "continuous":
            s = r.get("summary", "")
            rows.append({"cells": [f"{_name(v, labels, True)}, {s}", cont(r.get("overall"), s)]
                         + [cont((r.get("by_group") or {}).get(g["level"]), s) for g in groups] + p, "marks": m})
        else:
            rows.append({"cells": [f"{_name(v, labels)}, n (%)", ""] + [""] * len(groups) + p, "marks": m})
            for c in r.get("categories") or []:
                rows.append({"cells": [str(c["level"]), npct(c.get("overall"))]
                             + [npct((c.get("by_group") or {}).get(g["level"])) for g in groups]
                             + ([""] if groups else []), "indent": True})
    notes = ["Values are mean (SD), median (IQR, as lower to upper quartile) or n (%), chosen for each variable "
             "by a fixed rule from its distribution."]
    notes += [f"{letter} {test}." for test, letter in tests.items()]
    if missing:
        notes.append("Missing values: " + "; ".join(missing) + ".")
    if t.get("n_excluded"):
        notes.append(f"{t['n_excluded']} records without {by_name} are not shown.")
    return {"analysis_id": aid, "kind": "table1",
            "title": "Characteristics of the participants" + (f", by {by_name}" if by else ""),
            "columns": cols, "rows": rows, "footnotes": notes}


def _results(aid: str, a: dict[str, Any], out: dict[str, Any], labels: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    est = [e for e in out.get("estimates") or [] if isinstance(e, dict) and e.get("name") is not None]
    if not est:
        return None
    measures = list(dict.fromkeys(str(e.get("measure") or "Estimate") for e in est))
    terms = list(dict.fromkeys(str(e["name"]) for e in est))
    by = {(str(e["name"]), str(e.get("measure") or "Estimate")): e for e in est}
    cols, abbr = ["Variable"], {"CI": "confidence interval"}
    for m in measures:
        head, used = _measure(m)
        abbr.update(dict(used))
        cols += [f"{head} (95% CI)", "P value"]

    def cell(e: dict[str, Any] | None) -> list[str]:
        if not e:
            return ["", ""]
        lo, hi = e.get("ci_lower"), e.get("ci_upper")
        ci = f" ({_x(lo)} to {_x(hi)})" if lo is not None and hi is not None else ""
        return [f"{_x(e.get('estimate'))}{ci}", _p(e.get("p_value"))]

    rows = [{"cells": [labels.get(t, {}).get("label") or t] + [x for m in measures for x in cell(by.get((t, m)))]}
            for t in terms]
    # Drop P-value columns that are empty throughout (for example an AUC).
    keep = [i for i in range(len(cols)) if cols[i] != "P value" or any(r["cells"][i] for r in rows)]
    cols = [cols[i] for i in keep]
    for r in rows:
        r["cells"] = [r["cells"][i] for i in keep]
    notes = []
    n, ev, ex = out.get("n_analysed"), out.get("n_events"), out.get("n_excluded")
    if n is not None:
        notes.append(f"n = {n} analysed" + (f", {ev} with the outcome" if ev is not None else "")
                     + (f"; {ex} excluded" + (f" ({out['exclusion_reason']})" if out.get("exclusion_reason") else "")
                        if ex else "") + ".")
    model = (out.get("model") or {}).get("formula_or_description")
    if model:
        notes.append(f"Model: {model}.")
    sc = a.get("sample_check") or {}
    if sc.get("shortfall"):
        parts = [f"{r['achieved']} of {r['target']} planned {'participants' if k == 'n' else 'outcome events'}"
                 for k, r in sc["targets"].items() if r.get("short")]
        notes.append("Below the protocol's planned sample: " + "; ".join(parts) + ".")
    d = out.get("diagnostics") if isinstance(out.get("diagnostics"), dict) else None
    if d:
        f, asm = d.get("fit") or {}, d.get("assumptions") or {}
        cal = f.get("calibration") or {}
        vif = (asm.get("collinearity") or {}).get("vif") or {}
        bits = [f"{asm['events_per_parameter']} events per predictor parameter" if asm.get("events_per_parameter") is not None else "",
                f"C-statistic {_x(f.get('c_statistic'))}" if f.get("c_statistic") is not None else "",
                f"calibration slope {_x(cal.get('slope'))} (apparent)" if cal.get("slope") is not None else "",
                f"Hosmer-Lemeshow P = {_p((cal.get('hosmer_lemeshow') or {}).get('p_value'))}"
                if (cal.get("hosmer_lemeshow") or {}).get("p_value") is not None else "",
                f"adjusted R² {_x(f.get('adjusted_r2'))}" if f.get("adjusted_r2") is not None else "",
                f"largest variance inflation factor {_x(max(vif.values()))}" if vif else ""]
        notes.append("Model diagnostics: " + "; ".join(b for b in bits if b) + ".")
    notes.append(_label_note(a))
    notes.append("; ".join(f"{k}, {v}" for k, v in sorted(abbr.items())) + ".")
    return {"analysis_id": aid, "kind": "results", "title": str(a["spec"].get("title") or aid),
            "columns": cols, "rows": rows, "footnotes": notes}


def build(s: Any, blobs: BlobStore) -> list[dict[str, Any]]:
    """Every table, numbered in reporting order, from the current (not replaced) analyses."""
    from .workflow import current
    labels = {}
    if s.curator_reports:
        labels = _labels(blobs.get_json(s.curator_reports[-1]["report_sha256"]).get("dictionary"))
    tables = []
    for aid, a in sorted(current(s).items(), key=lambda kv: position(kv[1])):
        if a.get("standard") == "data_quality" or not blobs.has(a["outputs_sha256"]):
            continue
        out = blobs.get_json(a["outputs_sha256"])
        t = _table1(aid, a, out, labels) if a.get("standard") == "table1" else _results(aid, a, out, labels)
        if t:
            tables.append(t | {"label": a.get("label")})
    for i, t in enumerate(tables, start=1):
        t["number"] = i
    return tables


def listing(s: Any, blobs: BlobStore) -> str:
    rows = [f"Table {t['number']} [A:{t['analysis_id']}]: {t['title']}" for t in build(s, blobs)]
    return "\n".join(rows) or "(none)"


def to_docx(tables: list[dict[str, Any]], heading: str | None = None) -> bytes:
    """A Word file with the tables in journal style: three rules, no vertical lines, footnotes below."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    def borders(cell, **edges):
        tcpr = cell._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        for edge, on in edges.items():
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single" if on else "nil")
            if on:
                e.set(qn("w:sz"), "8"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
            b.append(e)
        tcpr.append(b)

    def tight(para):
        f = para.paragraph_format
        f.space_before, f.space_after, f.line_spacing = Pt(1), Pt(1), 1.0

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10)
    if heading:
        doc.add_paragraph().add_run(heading).bold = True
    for t in tables:
        p = doc.add_paragraph()
        p.add_run(f"Table {t['number']}. ").bold = True
        p.add_run(t["title"])
        tb = doc.add_table(rows=1 + len(t["rows"]), cols=len(t["columns"]))
        last = len(t["rows"])
        for j, h in enumerate(t["columns"]):
            c = tb.rows[0].cells[j]
            c.text = ""
            for k, part in enumerate(h.split("\n")):
                run = c.paragraphs[0].add_run(part)
                run.bold = True
                if k < len(h.split("\n")) - 1:
                    run.add_break()
            tight(c.paragraphs[0])
            if j:
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            borders(c, top=True, bottom=True, left=False, right=False)
        for i, r in enumerate(t["rows"], start=1):
            for j, v in enumerate(r["cells"]):
                c = tb.rows[i].cells[j]
                c.text = ""
                para = c.paragraphs[0]
                para.add_run(("    " if r.get("indent") and j == 0 else "") + str(v))
                mk = (r.get("marks") or {}).get(j) or (r.get("marks") or {}).get(str(j))
                if mk:
                    para.add_run(mk).font.superscript = True
                if j:
                    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                tight(para)
                borders(c, top=False, bottom=(i == last), left=False, right=False)
        for n in t["footnotes"]:
            fp = doc.add_paragraph()
            m = re.match(r"^([a-j]) (.*)$", n)
            if m:
                sup = fp.add_run(m.group(1))
                sup.font.superscript, sup.font.size = True, Pt(8.5)
                n = " " + m.group(2)
            fp.add_run(n).font.size = Pt(8.5)
            fp.paragraph_format.space_after = Pt(0)
        doc.add_paragraph()
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
