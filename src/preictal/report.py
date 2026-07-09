"""Self-contained HTML report for a pre-ictal run.

Assembles one static .html file — deliverable plot, step-by-step validation
figures (raw→feature→z→CWT and bins→AUC→collapse), the survivor list, and the
full scale-summary table — with every PNG base64-inlined so the file travels
(email attachment, committee slide) and opens in any browser. No server, no
kernel. Reuses figures.py (same plots as the email). Best-effort throughout:
missing plotly/kaleido or an ungatherable trace degrade gracefully to a
table-only report rather than failing.
"""

from __future__ import annotations

import base64
import html
import os
from datetime import datetime

_ALPHA = 0.05


def _b64_img(path: str | None, alt: str = "") -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return ""
    return (f'<img alt="{html.escape(alt)}" '
            f'style="max-width:100%;height:auto;border:1px solid #e0e0e0;'
            f'border-radius:4px" src="data:image/png;base64,{data}">')


def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return html.escape(str(x))


def _period_label(pf) -> str:
    try:
        f = float(pf)
    except (TypeError, ValueError):
        return "?"
    if f <= 0:
        return "?"
    p = 1.0 / f
    if p < 90:
        return f"{p:.0f}s"
    if p < 5400:
        return f"{p / 60:.1f}min"
    return f"{p / 3600:.1f}h"


def _survivors(rows, alpha: float = _ALPHA) -> list:
    out = [dict(r) for r in rows
           if (r.get("collapse_stat") or 0) > 0
           and r.get("null_p") is not None and r["null_p"] < alpha]
    out.sort(key=lambda r: r.get("collapse_stat") or 0, reverse=True)
    return out


def _scale_table(rows) -> str:
    cols = [("feature", "feature"), ("pseudo_freq_hz", "period"),
            ("collapse_stat", "collapse"), ("null_mean", "null μ"),
            ("null_std", "null σ"), ("null_p", "null p"),
            ("collapse_loso_mean", "LOSO"), ("forecast_roc_auc", "ROC"),
            ("forecast_pr_auc", "PR"), ("n_seizures", "n")]
    head = "".join(f"<th>{html.escape(t)}</th>" for _, t in cols)
    body = []
    for r in rows:
        d = dict(r)
        sig = (d.get("null_p") is not None and d["null_p"] < _ALPHA
               and (d.get("collapse_stat") or 0) > 0)
        tds = []
        for k, _ in cols:
            v = (_period_label(d.get(k)) if k == "pseudo_freq_hz"
                 else _fmt(d.get(k)))
            tds.append(f"<td>{v}</td>")
        row_style = ' style="background:#fff1f0"' if sig else ""
        body.append(f"<tr{row_style}>{''.join(tds)}</tr>")
    return (f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def _headline(run: dict, surv: list) -> str:
    if surv:
        items = "".join(
            f"<li><b>{html.escape(s['feature'])}</b> cleanest at "
            f"~{_period_label(s.get('pseudo_freq_hz'))} — collapse "
            f"{_fmt(s.get('collapse_stat'))}, p={_fmt(s.get('null_p'))}, "
            f"forecast ROC {_fmt(s.get('forecast_roc_auc'))}</li>"
            for s in surv)
        return (f"<p class='ok'><b>{len(surv)} scale(s) survive the surrogate "
                f"null</b> (uncorrected p&lt;{_ALPHA}):</p><ul>{items}</ul>"
                f"<p class='muted'>Note: uncorrected across "
                f"{run.get('n_scales','?')} scales — apply FDR before "
                f"reporting.</p>")
    return ("<p class='null'>No pre-ictal time-scale survived the surrogate "
            "null — consistent with no detectable pre-ictal change at these "
            "scales (or too few seizures in scope).</p>")


_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1040px;margin:24px auto;
padding:0 18px;color:#1a1a1a;line-height:1.5}
h1{font-size:22px;margin-bottom:2px} h2{font-size:16px;margin-top:28px;
border-bottom:1px solid #eee;padding-bottom:4px} .muted{color:#777;font-size:13px}
.ok{color:#0a7d2c} .null{color:#8a5a00;background:#fffaf0;padding:8px 12px;border-radius:4px}
table{border-collapse:collapse;font-size:12px;width:100%} th,td{border:1px solid #e2e2e2;
padding:3px 7px;text-align:right} th{background:#fafafa} td:first-child,th:first-child{text-align:left}
figure{margin:14px 0} figcaption{font-size:12.5px;color:#555;margin-top:6px}
.step{font-size:13px;color:#444;background:#f7f9fc;border-left:3px solid #1f77b4;
padding:8px 12px;margin:10px 0;border-radius:3px}
"""


def render_run_report(store, run: dict, rows: list, config: dict,
                      out_dir: str) -> str:
    """Render the self-contained HTML report into *out_dir*; return its path."""
    from src.preictal import figures
    os.makedirs(out_dir, exist_ok=True)
    run = dict(run)
    rid = run.get("id")

    deliver = figures.render_run_figures(run, rows, out_dir)
    pipeline_png = scoring_png = None
    trace_meta = ""
    try:                                               # validation trace (heavy)
        from src.preictal import engine
        trace = engine.trace_one_seizure(
            store, config, scope=run.get("scope", "daily"),
            period_start=run.get("period_start"),
            period_end=run.get("period_end"))
        if trace:
            pipeline_png = figures.render_pipeline_figure(trace, out_dir)
            scoring_png = figures.render_scoring_figure(trace, out_dir)
            sz = trace.get("seizure", {})
            trace_meta = (f"trace seizure: {html.escape(str(sz.get('animal')))} "
                          f"file {sz.get('file_id')} · ceiling "
                          f"{sz.get('ceiling_sec', 0):.0f}s · "
                          f"{len(trace.get('bin_edges', []) or []) - 1} lead-time bins")
    except Exception:                                  # noqa: BLE001
        pass

    surv = _survivors(rows)
    deliver_html = "".join(
        f"<figure>{_b64_img(p, 'collapse vs scale')}</figure>" for p in deliver)
    parts = [
        f"<h1>Pre-ictal sweep report — run {rid}</h1>",
        f"<p class='muted'>{html.escape(str(run.get('scope','')))} · "
        f"period {html.escape(str(run.get('period_start')))}"
        f"{'…' + html.escape(str(run.get('period_end'))) if run.get('period_end') and run.get('period_end') != run.get('period_start') else ''} · "
        f"{run.get('n_seizures','?')} seizures in scope · "
        f"{run.get('n_scales','?')} CWT scales · status "
        f"{html.escape(str(run.get('status','')))}</p>",
        _headline(run, surv),
        "<h2>Deliverable — pre-ictal gradient vs time-scale</h2>",
        "<p class='step'>Collapse statistic (signed AUC−0.5, averaged over "
        "near-vs-far lead-time bin pairs) at each CWT time-scale, against the "
        "circular-shift surrogate null (±2σ band). A real pre-ictal signal "
        "pokes <b>above</b> the null band with forecast ROC &gt; 0.5.</p>",
        deliver_html or "<p class='muted'>(no deliverable figure)</p>",
    ]
    if pipeline_png or scoring_png:
        parts.append("<h2>Validation — the analysis step by step</h2>")
        parts.append(f"<p class='muted'>{trace_meta}</p>")
    if pipeline_png:
        parts.append(
            "<p class='step'>One seizure through every transformation: "
            "<b>(1)</b> raw EEG lead-up → <b>(2)</b> feature trajectory "
            "(per-window) → <b>(3)</b> robust-z (median/MAD) → <b>(4)</b> "
            "Morlet CWT magnitude scalogram. Red dotted lines are the "
            "lead-time bin edges; onset is at the right (0 s).</p>")
        parts.append(f"<figure>{_b64_img(pipeline_png, 'pipeline trace')}"
                     f"<figcaption>Eyeball each stage: does the feature track "
                     f"the raw signal? does z center it? is the scalogram "
                     f"banded sensibly?</figcaption></figure>")
    if scoring_png:
        parts.append(
            "<p class='step'>The scoring step: per-lead-time-bin |W| "
            "distribution (left — the quantity compared) and the bin-vs-bin "
            "<b>AUC matrix</b> (right) whose near-vs-far upper triangle "
            "averages to the collapse gradient.</p>")
        parts.append(f"<figure>{_b64_img(scoring_png, 'scoring step')}</figure>")
    parts += [
        "<h2>All scales</h2>",
        "<p class='muted'>Rows highlighted red survive the uncorrected null "
        "(collapse&gt;0, p&lt;0.05).</p>",
        _scale_table(rows),
        f"<hr><p class='muted'>Generated "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        f"pocket {html.escape(str(run.get('derivatives_path') or '—'))}</p>",
    ]
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>Pre-ictal run {rid}</title><style>{_CSS}</style></head>"
           f"<body>{''.join(parts)}</body></html>")
    path = os.path.join(out_dir, f"preictal_run{rid}_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
