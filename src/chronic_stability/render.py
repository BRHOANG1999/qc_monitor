"""Figures for the chronic-stim stability map (matplotlib, headless-safe).

Kept separate from the analysis so the numbers can be unit-tested without a
display. Every function writes one PNG and returns its path.
"""

from __future__ import annotations

import datetime as _dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from src.periictal.forecast import _ecdf   # noqa: E402

_DPI = 130


def _dts(epochs):
    return [_dt.datetime.fromtimestamp(float(e)) for e in np.asarray(epochs)]


def ra_flatrun_fig(t_epoch, r, run: dict, eye_lo: float, eye_hi: float,
                   seizure_epochs, out_png: str, *, title: str = "") -> str:
    """Rₐ trend with the eyeballed span, the detected flat run, the per-point
    flat mask (rug), and a seizure rug."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = _dts(t_epoch)
    ax.plot(x, r, lw=0.8, color="#2b6cb0", label="Rₐ (kΩ)")
    ax.axvspan(_dt.datetime.fromtimestamp(eye_lo),
               _dt.datetime.fromtimestamp(eye_hi), color="#f6e05e", alpha=0.25,
               label="eyeballed window")
    if run.get("found"):
        ax.axvspan(_dt.datetime.fromtimestamp(run["epoch_start"]),
                   _dt.datetime.fromtimestamp(run["epoch_end"]),
                   color="#68d391", alpha=0.35, label="detected flat run")
        flat = run["flat_mask"]["flat"]
        fx = [x[i] for i in range(len(x)) if flat[i]]
        ax.plot(fx, [r.min()] * len(fx), "|", color="#2f855a", ms=6, alpha=0.6)
    sz = [s for s in np.asarray(seizure_epochs)
          if x[0].timestamp() <= s <= x[-1].timestamp()]
    ax.plot(_dts(sz), [r.max()] * len(sz), "v", color="#e53e3e", ms=5,
            label=f"seizures ({len(sz)})")
    ax.set_ylabel("Rₐ (kΩ)")
    ax.set_title(title or "SLM access resistance — flat-run detection")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def sensitivity_fig(sweep: list[dict], out_png: str) -> str:
    """Detected-window duration + start/end across the parameter grid."""
    found = [s for s in sweep if s.get("found")]
    fig, ax = plt.subplots(figsize=(9, 4))
    if found:
        durs = [s["duration_h"] for s in found]
        starts = _dts([s["epoch_start"] for s in found])
        ends = _dts([s["epoch_end"] for s in found])
        idx = np.arange(len(found))
        for i, (a, b) in enumerate(zip(starts, ends)):
            ax.plot([a, b], [i, i], color="#4a5568", lw=2, alpha=0.7)
        ax.scatter(starts, idx, s=12, color="#3182ce")
        ax.scatter(ends, idx, s=12, color="#dd6b20")
        ax.set_yticks(idx)
        ax.set_yticklabels([f"w{int(s.get('win_h',24))} "
                            f"d{s.get('disp_k',0):.1f} "
                            f"s{s.get('slope_k',0):.1f} ({d:.0f}h)"
                            for s, d in zip(found, durs)], fontsize=6)
    ax.set_title("Flat-run sensitivity sweep (start–end per parameter set)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def metric_timeseries_fig(epoch, mets: dict, ra_t, ra_r, intervals: list[dict],
                          run: dict, seizure_epochs, out_png: str) -> str:
    """Amplitude, corr, nrmse, Rₐ, and a polarity strip on one time axis, with
    configuration bands, the stable window, and a seizure rug."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    x = _dts(epoch)
    panels = [("amplitude", "amp (µV)", "#3182ce"),
              ("corr", "corr", "#38a169"), ("nrmse", "nRMSE", "#dd6b20")]
    for ax, (key, lab, col) in zip(axes[:3], panels):
        ax.plot(x, mets[key], ",", color=col, alpha=0.3)
        ax.set_ylabel(lab)
    axes[3].plot(_dts(ra_t), ra_r, lw=0.7, color="#805ad5")
    axes[3].set_ylabel("Rₐ (kΩ)")
    axes[4].plot(x, mets["polarity"], ",", color="#2d3748", alpha=0.3)
    axes[4].set_ylabel("polarity")
    axes[4].set_yticks([-1, 1])
    _overlay_bands(axes, epoch, intervals, run, seizure_epochs)
    axes[0].set_title("BCH111 per-trial artifact metrics + Rₐ over time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def _overlay_bands(axes, epoch, intervals, run, seizure_epochs):
    """Shade config intervals + stable window, add a seizure rug, on all axes."""
    e0, e1 = float(np.min(epoch)), float(np.max(epoch))
    cols = ["#ebf8ff", "#fefcbf", "#e6fffa", "#fff5f5"]
    for ax in axes:
        for k, iv in enumerate(intervals):
            lo = max(iv["start_epoch"], e0)
            hi = min(iv["end_epoch"], e1)
            if hi > lo:
                ax.axvspan(_dt.datetime.fromtimestamp(lo),
                           _dt.datetime.fromtimestamp(hi),
                           color=cols[k % len(cols)], alpha=0.5, zorder=0)
        if run.get("found"):
            ax.axvspan(_dt.datetime.fromtimestamp(run["epoch_start"]),
                       _dt.datetime.fromtimestamp(run["epoch_end"]),
                       color="#68d391", alpha=0.25, zorder=0)
    sz = [s for s in np.asarray(seizure_epochs) if e0 <= s <= e1]
    for s in sz:
        axes[0].axvline(_dt.datetime.fromtimestamp(s), color="#e53e3e",
                        lw=0.4, alpha=0.5)


def _plot_sample(x, cap: int = 20000) -> np.ndarray:
    """Finite values, deterministically down-sampled to <= *cap* points for
    plotting only (KS/MWU/CIs elsewhere use the full data)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size <= cap:
        return x
    step = int(np.ceil(x.size / cap))
    return x[::step]


def distribution_panel(groups: list, labels: list, metric_name: str,
                       out_png: str) -> str:
    """Overlaid ECDF + violin for 2+ metric samples (plot-sampled for speed)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#3182ce", "#dd6b20", "#38a169", "#805ad5"]
    for g, lab, c in zip(groups, labels, colors):
        g = _plot_sample(g)
        if g.size == 0:
            continue
        xs, ys = _ecdf(g)
        a1.step(xs, ys, where="post", color=c, lw=1.5, label=lab)
    a1.set_xlabel(metric_name)
    a1.set_ylabel("ECDF")
    a1.legend(fontsize=8)
    clean = [_plot_sample(g) for g in groups]
    clean = [g for g in clean if g.size]
    if clean:
        parts = a2.violinplot(clean, showmedians=True)
        a2.set_xticks(range(1, len(clean) + 1))
        a2.set_xticklabels(labels[:len(clean)], fontsize=7, rotation=15)
    a2.set_ylabel(metric_name)
    fig.suptitle(f"{metric_name}: distribution comparison")
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png


def age_stratified_fig(epoch, metric, config_start: float, metric_name: str,
                       out_png: str) -> str:
    """ECDFs of a metric within age quartiles of one configuration -- reveals a
    stable-vs-full gap that is really an age gradient."""
    epoch = np.asarray(epoch, float)
    metric = np.asarray(metric, float)
    age = (epoch - config_start) / 3600.0
    m = np.isfinite(age) & np.isfinite(metric)
    age, metric = age[m], metric[m]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if age.size >= 8:
        q = np.quantile(age, [0, .25, .5, .75, 1.0])
        cols = ["#bee3f8", "#63b3ed", "#3182ce", "#2c5282"]
        for i in range(4):
            sel = (age >= q[i]) & (age <= q[i + 1])
            if sel.sum() < 2:
                continue
            xs, ys = _ecdf(metric[sel])
            ax.step(xs, ys, where="post", color=cols[i], lw=1.4,
                    label=f"age Q{i+1} [{q[i]:.0f}-{q[i+1]:.0f} h]")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("ECDF")
    ax.set_title(f"{metric_name} by configuration age (collinearity check)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return out_png
