"""Deliverable figures for a pre-ictal run.

Per feature, one PNG with two stacked panels:
  1. COLLAPSE vs time-scale, overlaid on the surrogate-null band (mean ±2σ) --
     the headline "is there a pre-ictal gradient at any scale, beyond chance?"
     Scales that beat the null (null_p < alpha, positive collapse) are starred.
  2. FORECAST ROC vs time-scale, against the 0.5 chance line.

Rendered to PNG via plotly + kaleido so the daily digest can attach/inline them.
Rendering is BEST-EFFORT: if plotly/kaleido are unavailable it returns [] and
the digest stays text-only rather than failing.
"""

from __future__ import annotations

import os

_ALPHA = 0.05


def _period_sec(pseudo_freq_hz) -> float | None:
    try:
        f = float(pseudo_freq_hz)
    except (TypeError, ValueError):
        return None
    return 1.0 / f if f > 0 else None


def _by_feature(rows) -> dict:
    out: dict = {}
    for r in rows:
        d = dict(r)
        out.setdefault(d.get("feature"), []).append(d)
    return out


def _points(frows) -> list[tuple[float, dict]]:
    pts = []
    for r in frows:
        p = _period_sec(r.get("pseudo_freq_hz"))
        if p is None or r.get("collapse_stat") is None:
            continue
        pts.append((p, r))
    pts.sort(key=lambda t: t[0])
    return pts


def render_run_figures(run: dict, rows: list, out_dir: str,
                       alpha: float = _ALPHA) -> list[str]:
    """Write one PNG per feature into *out_dir*; return the paths written
    (empty if plotly/kaleido are unavailable or there's nothing to plot)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:                                  # noqa: BLE001
        return []
    written: list[str] = []
    for feat, frows in _by_feature(rows).items():
        pts = _points(frows)
        if not pts:
            continue
        periods = [p for p, _ in pts]
        collapse = [r["collapse_stat"] for _, r in pts]
        nmean = [r.get("null_mean") for _, r in pts]
        nstd = [r.get("null_std") for _, r in pts]
        roc = [r.get("forecast_roc_auc") for _, r in pts]
        hi = [(m + 2 * s) if (m is not None and s is not None) else None
              for m, s in zip(nmean, nstd)]
        lo = [(m - 2 * s) if (m is not None and s is not None) else None
              for m, s in zip(nmean, nstd)]
        sig_x = [p for (p, r) in pts
                 if r.get("null_p") is not None and r["null_p"] < alpha
                 and (r.get("collapse_stat") or 0) > 0]
        sig_y = [r["collapse_stat"] for (_, r) in pts
                 if r.get("null_p") is not None and r["null_p"] < alpha
                 and (r.get("collapse_stat") or 0) > 0]
        n_sz = max((r.get("n_seizures") or 0) for _, r in pts)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
            subplot_titles=(
                f"{feat}: pre-ictal gradient vs time-scale (collapse vs null ±2σ)",
                "Forecast ROC vs time-scale"))
        # null band (mean ±2σ) as a filled envelope
        if all(v is not None for v in hi + lo):
            fig.add_trace(go.Scatter(
                x=periods + periods[::-1], y=hi + lo[::-1], fill="toself",
                fillcolor="rgba(140,140,140,0.25)", line=dict(width=0),
                name="null ±2σ", hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=periods, y=nmean, mode="lines",
                                 line=dict(color="gray", dash="dot"),
                                 name="null mean"), row=1, col=1)
        fig.add_trace(go.Scatter(x=periods, y=collapse, mode="lines+markers",
                                 line=dict(color="#1f77b4"),
                                 name="collapse"), row=1, col=1)
        if sig_x:
            fig.add_trace(go.Scatter(
                x=sig_x, y=sig_y, mode="markers",
                marker=dict(color="red", size=11, symbol="star"),
                name=f"p<{alpha}"), row=1, col=1)
        fig.add_hline(y=0.0, line=dict(color="black", width=1), row=1, col=1)
        fig.add_trace(go.Scatter(x=periods, y=roc, mode="lines+markers",
                                 line=dict(color="#2ca02c"),
                                 name="ROC"), row=2, col=1)
        fig.add_hline(y=0.5, line=dict(color="black", width=1, dash="dash"),
                      row=2, col=1)
        fig.update_xaxes(type="log",
                         title_text="pre-ictal time-scale / period (s)",
                         row=2, col=1)
        fig.update_yaxes(title_text="collapse (AUC−0.5, signed)", row=1, col=1)
        fig.update_yaxes(title_text="ROC AUC", range=[0, 1], row=2, col=1)
        fig.update_layout(
            width=900, height=650, template="plotly_white",
            title_text=(f"Pre-ictal run {run.get('id')} — "
                        f"{run.get('period_start')} ({n_sz} seizures)"),
            legend=dict(orientation="h", y=-0.13))
        os.makedirs(out_dir, exist_ok=True)
        safe_feat = "".join(c if c.isalnum() else "_" for c in str(feat))
        path = os.path.join(out_dir,
                            f"preictal_run{run.get('id')}_{safe_feat}.png")
        try:
            fig.write_image(path, width=900, height=650, scale=2)
            written.append(path)
        except Exception:                              # noqa: BLE001
            continue                                   # kaleido hiccup -> skip
    return written


def render_pipeline_figure(trace: dict, out_dir: str) -> str | None:
    """Step-by-step transformation of ONE seizure so the analysis can be
    eyeballed for correctness: raw EEG lead-up -> feature trajectory ->
    robust-z -> Morlet CWT |W| scalogram, sharing a seconds-before-onset axis
    with the lead-time bin edges overlaid. Returns the PNG path (or None)."""
    if not trace or not trace.get("seizure"):
        return None
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:                                  # noqa: BLE001
        return None
    sz = trace["seizure"]
    fs = float(sz["fs"]); ceil = float(sz["ceiling_sec"])
    sig = np.asarray(sz["signal"]); t = np.asarray(sz["trajectory"])
    z = np.asarray(sz["z"]); mag = np.asarray(sz["coeffs_abs"])
    lead_traj = np.asarray(sz["lead_sec"])
    freqs = np.asarray(trace["freqs"]); bin_edges = np.asarray(trace["bin_edges"])
    periods = [1.0 / f if f > 0 else None for f in freqs]
    # decimate raw signal for plotting; its own seconds-before-onset axis
    n = sig.size
    stride = max(1, n // 4000)
    sig_lead = ceil - np.arange(0, n, stride) / fs
    sig_ds = sig[::stride]
    # heatmap wants ascending x -> reverse (lead_traj is descending in index)
    x_asc = lead_traj[::-1]
    z_asc = mag[:, ::-1]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.22, 0.21, 0.21, 0.36],
        subplot_titles=(
            f"1) raw EEG lead-up (channel {sz['channel_index']}, fs {fs:.0f}Hz)",
            f"2) feature trajectory: {trace['feature']} (per {trace['step_sec']:.0f}s)",
            "3) robust-z trajectory (median/MAD)",
            "4) Morlet CWT |W| scalogram (period × lead-time)"))
    fig.add_trace(go.Scatter(x=sig_lead, y=sig_ds, line=dict(width=0.5,
                  color="#555"), name="raw"), row=1, col=1)
    fig.add_trace(go.Scatter(x=lead_traj, y=t, line=dict(color="#1f77b4"),
                  name="feature"), row=2, col=1)
    fig.add_trace(go.Scatter(x=lead_traj, y=z, line=dict(color="#ff7f0e"),
                  name="z"), row=3, col=1)
    fig.add_trace(go.Heatmap(x=x_asc, y=periods, z=z_asc, colorscale="Viridis",
                  colorbar=dict(title="|W|", len=0.3, y=0.14)), row=4, col=1)
    for e in bin_edges:                                # lead-time bin edges
        fig.add_vline(x=float(e), line=dict(color="rgba(200,0,0,0.35)",
                      width=1, dash="dot"))
    fig.update_xaxes(autorange="reversed")             # onset (0) at right
    fig.update_xaxes(title_text="seconds before onset  (→ 0 = onset)",
                     row=4, col=1)
    fig.update_yaxes(title_text="period (s)", type="log", row=4, col=1)
    fig.update_layout(
        width=980, height=900, template="plotly_white", showlegend=False,
        title_text=(f"Pipeline trace — {sz['animal']} file {sz['file_id']} "
                    f"(Racine {sz['racine']}), ceiling {ceil:.0f}s"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "preictal_pipeline_trace.png")
    try:
        fig.write_image(path, width=980, height=900, scale=2)
        return path
    except Exception:                                  # noqa: BLE001
        return None


def render_scoring_figure(trace: dict, out_dir: str) -> str | None:
    """The scoring step made visible: per-lead-time-bin |W| distribution (the
    quantity the AUC compares) and the bin-vs-bin AUC matrix whose upper
    triangle averages to the collapse gradient. Computed on the probed subset
    at a mid-band scale. Returns the PNG path (or None)."""
    subset = (trace or {}).get("subset")
    if not subset:
        return None
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:                                  # noqa: BLE001
        return None
    pooled = [np.asarray(p) for p in subset["pooled"]]
    auc = np.asarray(subset["auc_matrix"], dtype=float)
    nb = len(pooled)
    means = [float(p.mean()) if p.size else None for p in pooled]
    sds = [float(p.std()) if p.size else 0.0 for p in pooled]
    period = (1.0 / subset["pseudo_freq_hz"]) if subset["pseudo_freq_hz"] else 0
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.13,
        subplot_titles=(
            f"per-bin |W| at period ~{period:.0f}s (mean±sd)",
            f"bin-vs-bin AUC (collapse = {subset['collapse']:+.3f})"))
    fig.add_trace(go.Scatter(x=list(range(nb)), y=means,
                  error_y=dict(type="data", array=sds), mode="lines+markers",
                  line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Heatmap(z=auc, zmid=0.5, zmin=0.0, zmax=1.0,
                  colorscale="RdBu_r", colorbar=dict(title="AUC"),
                  x=list(range(nb)), y=list(range(nb))), row=1, col=2)
    fig.update_xaxes(title_text="lead-time bin (0 = nearest onset)", row=1, col=1)
    fig.update_xaxes(title_text="bin j", row=1, col=2)
    fig.update_yaxes(title_text="mean |W|", row=1, col=1)
    fig.update_yaxes(title_text="bin i", row=1, col=2)
    fig.update_layout(
        width=980, height=430, template="plotly_white", showlegend=False,
        title_text=(f"Scoring step — {subset['n']} probed seizures, "
                    f"mid-band scale index {subset['scale_index']}"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "preictal_scoring_step.png")
    try:
        fig.write_image(path, width=980, height=430, scale=2)
        return path
    except Exception:                                  # noqa: BLE001
        return None
