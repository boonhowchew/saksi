# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Figures. Never drawn by a model.

Two kinds:
  * Drawn by Saksi from the recorded results, deterministically, with no model and no data
    access: a forest plot for ratio or difference estimates, and a participant flow diagram
    from the recorded counts. Same results, same SVG.
  * Drawn by the analysis code in the sandbox, from the data, when the analysis needs a
    figure that only the data can give (ROC curve, Kaplan-Meier, calibration). Which figure
    is required is decided here by a fixed rule from the confirmed plan, not by a model.
"""

from __future__ import annotations

import math
from typing import Any
from xml.sax.saxutils import escape

from .tracing import NULL_VALUES

INK, MUTED, RULE = "#1B2A4A", "#5B6680", "#C9D1DB"
FONT = "font-family:Helvetica,Arial,sans-serif"

# Data-dependent figures, chosen by rule from the plan's wording.
REQUIRED = (
    (("auc", "auroc", "roc", "discrimination", "c-statistic", "c statistic"), "roc_curve",
     "ROC curve: sensitivity against 1 - specificity, with the diagonal for no discrimination"),
    (("kaplan", "survival", "time-to-event", "time to event", "hazard"), "kaplan_meier",
     "Kaplan-Meier curves by group, with numbers at risk"),
    (("calibration",), "calibration_plot", "Calibration plot: observed against predicted risk, by tenth of risk"),
)


def required_figure(spec: dict[str, Any]) -> dict[str, str] | None:
    text = " ".join(str(spec.get(k) or "") for k in ("title", "objective", "method", "effect_measure")).lower()
    for words, name, what in REQUIRED:
        if any(w in text for w in words):
            return {"name": name, "what": what}
    return None


def _t(x: float, y: float, s: str, size: int = 12, anchor: str = "start", colour: str = INK, weight: str = "normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{colour}" '
            f'font-weight="{weight}" style="{FONT}">{escape(s)}</text>')


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def forest_svg(estimates: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Forest plot of estimates sharing one measure. Returns (svg, measure) or None."""
    rows = []
    for e in estimates or []:
        lo, hi, est = e.get("ci_lower"), e.get("ci_upper"), e.get("estimate")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (lo, hi, est)):
            continue
        label = f"{e.get('measure') or ''} {e.get('name') or ''}"
        null = next((v for pat, v in NULL_VALUES if pat.search(label)), None)
        if null not in (0.0, 1.0):          # forest plots for ratios and differences only
            continue
        rows.append((str(e.get("name") or ""), float(est), float(lo), float(hi), null, str(e.get("measure") or "")))
    if not rows:
        return None
    null, measure = rows[0][4], rows[0][5]
    rows = [r for r in rows if r[4] == null and r[5] == measure]
    log = null == 1.0
    if log and any(min(r[1], r[2]) <= 0 for r in rows):
        return None
    f = math.log if log else (lambda v: v)
    lo_v = min([f(r[2]) for r in rows] + [f(null)])
    hi_v = max([f(r[3]) for r in rows] + [f(null)])
    pad = (hi_v - lo_v) * 0.08 or 0.1
    lo_v, hi_v = lo_v - pad, hi_v + pad
    W, left, plot_w, top, row_h = 700, 170, 330, 34, 30
    H = top + row_h * len(rows) + 60
    X = lambda v: left + (f(v) - lo_v) / (hi_v - lo_v) * plot_w
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">',
           f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
           _t(10, 20, "Term", 12, colour=MUTED, weight="bold"),
           _t(left + plot_w + 20, 20, "Estimate (95% CI)", 12, colour=MUTED, weight="bold")]
    x0 = X(null)
    out.append(f'<line x1="{x0:.1f}" y1="{top - 8}" x2="{x0:.1f}" y2="{top + row_h * len(rows)}" stroke="{MUTED}" '
               f'stroke-dasharray="4 3"/>')
    for i, (name, est, lo, hi, _, _) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        out += [_t(10, y + 4, name[:26]),
                f'<line x1="{X(lo):.1f}" y1="{y:.1f}" x2="{X(hi):.1f}" y2="{y:.1f}" stroke="{INK}" stroke-width="1.6"/>',
                f'<rect x="{X(est) - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" fill="{INK}"/>',
                _t(left + plot_w + 20, y + 4, f"{_fmt(est)} ({_fmt(lo)} to {_fmt(hi)})")]
    ya = top + row_h * len(rows) + 8
    out.append(f'<line x1="{left}" y1="{ya}" x2="{left + plot_w}" y2="{ya}" stroke="{RULE}"/>')
    cands = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100] if log else None
    if log:
        ticks = [t for t in cands if lo_v <= math.log(t) <= hi_v]
    else:
        step = 10 ** math.floor(math.log10((hi_v - lo_v) / 4 or 1))
        ticks = [round(k * step, 10) for k in range(math.ceil(lo_v / step), math.floor(hi_v / step) + 1)][:9]
    for t in ticks:
        out += [f'<line x1="{X(t):.1f}" y1="{ya}" x2="{X(t):.1f}" y2="{ya + 5}" stroke="{MUTED}"/>',
                _t(X(t), ya + 18, f"{t:g}", 11, "middle", MUTED)]
    out.append(_t(left + plot_w / 2, ya + 38, f"{measure or 'Estimate'}{' (log scale)' if log else ''}", 12, "middle", MUTED))
    out.append("</svg>")
    return "".join(out), measure


def _box(x: float, y: float, w: float, lines: list[str]) -> tuple[str, float]:
    h = 16 * len(lines) + 16
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#FFFFFF" stroke="{INK}"/>']
    for i, ln in enumerate(lines):
        parts.append(_t(x + w / 2, y + 22 + 16 * i, ln, 13, "middle", INK, "bold" if i == 0 else "normal"))
    return "".join(parts), h


def _wrap(s: str, n: int = 34) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


def flow_counts_problem(rows: int | None, outputs: dict[str, Any]) -> str | None:
    """The flow must add up: records in the dataset = excluded + analysed."""
    n, excl = outputs.get("n_analysed"), outputs.get("n_excluded")
    if all(isinstance(v, int) and not isinstance(v, bool) for v in (rows, n, excl)) and rows != n + excl:
        return f"counts do not add up: {rows} records, but {excl} excluded + {n} analysed = {excl + n}"
    return None


def flow_svg(rows: int | None, outputs: dict[str, Any], outcome: str = "outcome") -> str | None:
    """Participant flow diagram from recorded counts only."""
    n = outputs.get("n_analysed")
    if not isinstance(n, int) or isinstance(n, bool):
        return None
    excl = outputs.get("n_excluded")
    if not isinstance(excl, int) and isinstance(rows, int):
        excl = rows - n
    events = outputs.get("n_events")
    W, cx, w = 640, 60, 260
    parts, y = [], 16
    if isinstance(rows, int):
        b, h = _box(cx, y, w, ["Records in the dataset", f"n = {rows}"])
        parts.append(b)
        top_bottom = y + h
        y = top_bottom + 60
        if isinstance(excl, int):
            reason = _wrap(str(outputs.get("exclusion_reason") or "reason not reported"), 30)[:3]
            eb, eh = _box(cx + w + 50, top_bottom + 8, 250, ["Excluded", f"n = {excl}"] + reason)
            parts.append(eb)
            midy = top_bottom + 8 + eh / 2
            parts.append(f'<line x1="{cx + w / 2}" y1="{midy}" x2="{cx + w + 50}" y2="{midy}" stroke="{INK}"/>')
            y = max(y, top_bottom + 8 + eh + 24)
        parts.append(f'<line x1="{cx + w / 2}" y1="{top_bottom}" x2="{cx + w / 2}" y2="{y}" stroke="{INK}"/>')
    lines = ["Analysed", f"n = {n}"] + ([f"{events} with {outcome}"] if isinstance(events, int) else [])
    b, h = _box(cx, y, w, lines)
    parts.append(b)
    H = y + h + 16
    problem = flow_counts_problem(rows, outputs)
    if problem:                                    # drawn as recorded, and said plainly on the figure itself
        parts.append(_t(cx, H + 4, "Warning: " + problem, 12, colour="#9A6416", weight="bold"))
        H += 22
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">'
            f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(parts) + "</svg>")


def calibration_svg(deciles: list[dict[str, Any]]) -> str | None:
    """Observed proportion against mean predicted risk, by tenths of predicted risk, with the diagonal."""
    pts = [(d.get("mean_predicted"), d.get("observed_proportion"), d.get("n")) for d in deciles or []]
    pts = [(x, y, n) for x, y, n in pts if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    if len(pts) < 3:
        return None
    top = max(max(x for x, _, _ in pts), max(y for _, y, _ in pts), 0.05)
    top = min(1.0, math.ceil(top * 10) / 10)
    W, H, L, T, S = 420, 400, 60, 20, 320
    X = lambda v: L + v / top * S
    Y = lambda v: T + S - v / top * S
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">',
           f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
           f'<rect x="{L}" y="{T}" width="{S}" height="{S}" fill="none" stroke="{RULE}"/>',
           f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(top):.1f}" y2="{Y(top):.1f}" stroke="{MUTED}" stroke-dasharray="4 3"/>']
    out.append('<polyline fill="none" stroke="' + INK + '" stroke-width="1.5" points="'
               + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y, _ in pts) + '"/>')
    for x, y, _ in pts:
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{INK}"/>')
    for k in range(0, 6):
        v = top * k / 5
        out += [_t(X(v), T + S + 16, f"{v:.2g}", 11, "middle", MUTED), _t(L - 8, Y(v) + 4, f"{v:.2g}", 11, "end", MUTED)]
    out += [_t(L + S / 2, H - 6, "Mean predicted risk, by tenth of risk", 12, "middle", MUTED),
            f'<text x="16" y="{T + S / 2:.0f}" font-size="12" fill="{MUTED}" text-anchor="middle" style="{FONT}" '
            f'transform="rotate(-90 16 {T + S / 2:.0f})">Observed proportion</text>', "</svg>"]
    return "".join(out)


def saksi_figures(spec: dict[str, Any], outputs: dict[str, Any], rows: int | None,
                  label: str, basis: str | None) -> list[dict[str, str]]:
    """The figures Saksi draws itself for one analysis, with captions built from recorded numbers only."""
    figs = []
    prefix = ""      # the label belongs in the text, not in captions or headings (author's decision, 31 Oct)
    title = spec.get("title") or spec.get("id") or "the analysis"
    fp = forest_svg(outputs.get("estimates") or [])
    if fp:
        svg, measure = fp
        what = (measure or "Estimates")
        figs.append({"name": "forest_plot", "svg": svg,
                     "caption": f"{prefix}{what[:1].upper() + what[1:]} with 95% confidence intervals: {title}."})
    cal = (((outputs.get("diagnostics") or {}).get("fit") or {}).get("calibration") or {}) \
        if isinstance(outputs.get("diagnostics"), dict) else {}
    cs = calibration_svg(cal.get("deciles") or [])
    if cs:
        figs.append({"name": "calibration_plot", "svg": cs,
                     "caption": f"{prefix}Calibration of the model for {title}: observed proportion against mean "
                                "predicted risk by tenth of predicted risk; the dashed line is perfect calibration."})
    outcome = ((spec.get("outcome") or {}).get("name") if isinstance(spec.get("outcome"), dict) else None) or "outcome"
    fl = flow_svg(rows, outputs, outcome)
    if fl:
        n, excl = outputs.get("n_analysed"), outputs.get("n_excluded")
        if not isinstance(excl, int) and isinstance(rows, int):
            excl = rows - n
        figs.append({"name": "participant_flow", "svg": fl, "warning": flow_counts_problem(rows, outputs),
                     "flow_counts": [rows, n, excl, outputs.get("n_events")],
                     "caption": f"{prefix}Participant flow for {title}: records in the dataset, exclusions and records analysed."})
    return figs
