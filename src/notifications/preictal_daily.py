"""Daily pre-ictal digest.

Reads the latest completed DAILY pre-ictal run and reports the headline
deliverable: which CWT time-scales show a pre-ictal gradient that SURVIVES the
surrogate null (null_p < alpha), per feature -- with the forecasting ROC/PR and
the LOSO CI. It never runs the sweep (the worker does); it only summarizes the
last one. Mirrors eod_digest's shape: disabled/recipient guards, text+HTML,
emailer.send. Weekly/monthly land in stage 3.
"""

from __future__ import annotations

import logging
import os
from datetime import date

logger = logging.getLogger(__name__)

_ALPHA = 0.05


def _period_label(pseudo_freq_hz) -> str:
    """A CWT scale's pseudo-frequency (Hz) -> a human period label."""
    try:
        f = float(pseudo_freq_hz)
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


def _survivors(rows: list) -> list:
    """Scale rows with a POSITIVE collapse that beat the null, best first."""
    out = [r for r in rows
           if (r.get("collapse_stat") or 0) > 0
           and r.get("null_p") is not None and r["null_p"] < _ALPHA]
    out.sort(key=lambda r: r.get("collapse_stat") or 0, reverse=True)
    return out


def _best_per_feature(rows: list) -> dict:
    """feature -> its single best null-surviving scale row (or None)."""
    by_feat: dict = {}
    for r in rows:
        by_feat.setdefault(r["feature"], []).append(r)
    out = {}
    for feat, frows in by_feat.items():
        surv = _survivors(frows)
        out[feat] = surv[0] if surv else None
    return out


def _lines(run: dict, rows: list) -> list[str]:
    ls = [f"Pre-ictal daily sweep — run {run.get('id')} "
          f"({run.get('period_start')})",
          f"{run.get('n_seizures', 0)} seizures in scope, "
          f"{run.get('n_scales', 0)} CWT scales; ISI ceiling median "
          f"{run.get('ceiling_median_sec')}s.", ""]
    best = _best_per_feature(rows)
    any_surv = False
    for feat, r in sorted(best.items()):
        if r is None:
            ls.append(f"  {feat}: no scale survives the null.")
            continue
        any_surv = True
        loso = ""
        if r.get("collapse_loso_mean") is not None:
            loso = (f", LOSO {r['collapse_loso_mean']:.3f}"
                    f"[{r.get('collapse_loso_ci_lo')},"
                    f"{r.get('collapse_loso_ci_hi')}]")
        ls.append(
            f"  {feat}: cleanest at ~{_period_label(r['pseudo_freq_hz'])} "
            f"(collapse {r['collapse_stat']:.3f}, p={r['null_p']:.3g}{loso}; "
            f"forecast ROC {r.get('forecast_roc_auc')}, "
            f"PR {r.get('forecast_pr_auc')}).")
    if not any_surv:
        ls.append("")
        ls.append("No pre-ictal time-scale survived the surrogate null today "
                  "— consistent with no detectable pre-ictal change (or too "
                  "few seizures in scope).")
    if run.get("derivatives_path"):
        ls += ["", f"Pocket: {run['derivatives_path']}"]
    return ls


def send_preictal_daily(today: date, config: dict, store, emailer) -> dict:
    """Build + send the daily pre-ictal digest. Returns a status dict."""
    cfg = ((config.get("notifications", {}) or {})
           .get("preictal_daily", {}) or {})
    if not cfg.get("enabled", False):
        return {"sent": False, "reason": "disabled"}
    if store is None:
        return {"sent": False, "reason": "no store"}
    recipients = list(
        cfg.get("recipients")
        or ((config.get("alerting", {}) or {}).get("smtp", {})
            .get("recipients", []) or []))
    if not recipients:
        return {"sent": False, "reason": "no recipients"}
    run = store.latest_preictal_run("daily")
    if not run:
        return {"sent": False, "reason": "no completed daily run"}
    rows = store.preictal_scale_summary_for_run(run["id"])
    lines = _lines(run, rows)
    text = "\n".join(lines)

    # Figures + standalone HTML report (best-effort; only when a derivatives
    # root is configured, so unit tests with a bare config stay text-only).
    # Figures are INLINED (deliverable + validation) so the email itself is the
    # visual report. The .html report is NOT attached by default -- enterprise
    # mail security (e.g. umn.edu) quarantines HTML attachments as a phishing
    # risk, which silently drops the whole message; set attach_report: true to
    # opt back in. The report still lands in the run pocket regardless.
    figs, report_path = _figures_and_report(config, store, run, rows)
    img_html = "".join(
        f'<div style="margin-top:14px"><img src="cid:{os.path.basename(p)}" '
        f'style="max-width:920px;width:100%"></div>' for p in figs)
    html = ("<h3>Pre-ictal daily sweep</h3><pre>"
            + "\n".join(lines[1:]) + "</pre>" + img_html)
    if report_path:
        html += ("<p style='color:#777;font-size:12px'>Self-contained report "
                 f"saved in the run pocket: {report_path}</p>")

    n_surv = len(_survivors(rows))
    subject = (f"Pre-ictal daily — {n_surv} scale(s) survive the null "
               f"({today.isoformat()})")
    send_kwargs = dict(subject=subject, body=text, body_html=html,
                       recipients=recipients, subject_prefix=False)
    if figs:
        send_kwargs["attachments"] = figs
    if report_path and cfg.get("attach_report", False):
        send_kwargs["file_attachments"] = [report_path]
    try:
        ok = emailer.send(**send_kwargs)
    except Exception as e:  # noqa: BLE001
        logger.error("preictal daily digest send failed: %s", e)
        return {"sent": False, "reason": f"send error: {e}"}
    return {"sent": bool(ok), "recipients": len(recipients),
            "run_id": run["id"], "survivors": n_surv,
            "figures": len(figs), "report": bool(report_path)}


def _figures_and_report(config, store, run, rows):
    """(inline figure PNG paths, report HTML path) for the digest, or ([], None)
    when no derivatives root is configured or rendering isn't available. The
    report render also writes the deliverable + validation PNGs into out_dir;
    we collect them ALL (ordered deliverable -> pipeline -> scoring) for inline
    embedding so the email body is the full visual report."""
    root = ((config or {}).get("preictal", {}) or {}).get("derivatives_root")
    if not root:
        return [], None
    out_dir = os.path.join(root, "figures", f"run_{run['id']}")
    report_path = None
    try:
        from src.preictal import report as _report
        report_path = _report.render_run_report(store, dict(run), rows,
                                                 config, out_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("preictal digest report failed: %s", e)

    def _rank(name):
        if "pipeline" in name:
            return 1
        if "scoring" in name:
            return 2
        return 0                                       # deliverable first
    figs = []
    try:
        pngs = [f for f in os.listdir(out_dir) if f.lower().endswith(".png")]
        figs = [os.path.join(out_dir, f)
                for f in sorted(pngs, key=_rank)]
    except OSError:
        pass
    return figs, report_path
