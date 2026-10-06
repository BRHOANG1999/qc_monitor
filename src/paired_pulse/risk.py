"""Risk-given-the-measurement figure (advisor's 'next version' of P(event within H | PPR)).

Three fixes over the old P(event|metric) plot:
 * **x = quantile RANK**, not the metric value -- so the ratio-junk tail (a tiny S1
   denominator blows PPR up to 3-10) no longer squashes the informative middle bins.
 * **y = RELATIVE RISK** (P in bin / base rate), so the modulation is read directly.
 * **circular-shift null bands** per bin, because the effective N is the handful of
   seizures and the ~300 pairs per pre-seizure window are heavily autocorrelated.

Plus a **time-of-day-stratified** version (bin by PPR rank WITHIN day / night): if the
low-PPR risk elevation survives within-stratum it is pre-ictal; if it collapses it is the
vigilance state seizures prefer (a different, still-useful claim). Vigilance scoring is
unavailable, so time-of-day is the proxy.

Output: data/BCH111_paired_pulse_lp500/periictal/risk/.  CLI: --risk.
"""

from __future__ import annotations

import datetime as _dt
import os

import numpy as np

from . import config as C, data as D, figures as G

# (horizon seconds, label, colour)
HORIZONS = [(600.0, "10 min", "#5b6ee1"), (1800.0, "30 min", "#2ee6a6"),
            (3600.0, "1 h", "#e6800f"), (7200.0, "2 h", "#d62f2f")]
_FEATS = [("ppr_peak_to_trough", "PPR p2p"), ("ppr_rms_amplitude", "PPR rms"),
          ("ppr_line_length", "PPR line-len"), ("ppr_max_slope", "PPR slope")]


def _rank_bins(v, n_bins):
    """Quantile-rank bin index per epoch (0..n_bins-1, equal counts); NaN -> -1."""
    out = np.full(v.size, -1, int)
    idx = np.flatnonzero(np.isfinite(v))
    if idx.size:
        r = np.argsort(np.argsort(v[idx]))
        out[idx] = np.minimum(r * n_bins // idx.size, n_bins - 1)
    return out


def _event_within(t, onsets, horizon):
    """Per epoch: is the NEXT onset within `horizon` seconds ahead?"""
    o = np.sort(np.asarray(onsets, float))
    if not o.size:
        return np.zeros(t.size, bool)
    i = np.searchsorted(o, t, side="right")
    nxt = np.where(i < o.size, o[np.clip(i, 0, o.size - 1)], np.inf)
    return (nxt - t) <= horizon


def _rr(ev, binidx, n_bins):
    base = ev.mean() if ev.size else np.nan
    p = np.array([ev[binidx == b].mean() if np.any(binidx == b) else np.nan
                  for b in range(n_bins)])
    return (p / base if base and base > 0 else np.full(n_bins, np.nan)), base


def risk_by_rank(mat, onsets, feature, *, n_bins=5, n_surr=2000, seed=0, keep=None):
    """Relative risk per PPR quantile-rank bin, per horizon, with a circular-shift null
    band. keep = optional boolean epoch mask (e.g. a time-of-day stratum)."""
    t = mat["t_epoch"].to_numpy(float); v = mat[feature].to_numpy(float)
    if keep is not None:
        t, v = t[keep], v[keep]
    binidx = _rank_bins(v, n_bins)
    ons = np.sort(np.asarray(onsets, float))
    lo, hi = t.min(), t.max(); span = hi - lo; ms = 3 * 3600.0
    rng = np.random.default_rng(seed)
    out = {"n_bins": n_bins, "n": int(np.sum(binidx >= 0))}
    for H, _lab, _col in HORIZONS:
        obs, base = _rr(_event_within(t, ons, H), binidx, n_bins)
        null = np.full((n_surr, n_bins), np.nan)
        for i in range(n_surr):
            sh = lo + ((ons - lo + rng.uniform(ms, span - ms)) % span)
            null[i], _ = _rr(_event_within(t, sh, H), binidx, n_bins)
        med = np.nanmedian(null, axis=0)
        p = np.array([(np.sum(np.abs(null[:, b] - med[b]) >= abs(obs[b] - med[b])) + 1)
                      / (np.isfinite(null[:, b]).sum() + 1) if np.isfinite(obs[b])
                      else np.nan for b in range(n_bins)])
        out[H] = {"rr": obs, "lo": np.nanpercentile(null, 2.5, axis=0),
                  "hi": np.nanpercentile(null, 97.5, axis=0), "base": base, "p": p}
    return out


def risk_by_rank_fig(results, out_png, *, title="") -> str:
    """One panel per feature: relative risk vs PPR quantile rank, a line per horizon +
    the circular-shift null 95% band; RR=1 reference. Bins outside the band are ringed."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(_FEATS), figsize=(3.8 * len(_FEATS), 4.5),
                             facecolor=C.BG, squeeze=False)
    for ax, (feat, flab) in zip(axes[0], _FEATS):
        if feat not in results:
            ax.axis("off"); continue
        r = results[feat]; nb = r["n_bins"]; x = np.arange(1, nb + 1)
        for H, lab, col in HORIZONS:
            d = r[H]
            ax.fill_between(x, d["lo"], d["hi"], color=col, alpha=0.12, lw=0)
            ax.plot(x, d["rr"], "-o", color=col, lw=1.6, ms=3.5)
            sig = d["p"] < 0.05
            if np.any(sig):
                ax.plot(x[sig], d["rr"][sig], "o", mfc="none", mec=col, ms=9, mew=1.6)
        ax.axhline(1.0, color=C.MUTED, lw=0.9, ls="--")
        G._dark(ax, flab)
        ax.set_xlabel("PPR quantile rank (1 = lowest)", color=C.TEXT)
        ax.set_xticks(x)
    axes[0][0].set_ylabel("relative risk  (P in bin / base rate)", color=C.TEXT)
    handles = [Line2D([], [], color=c, marker="o", label=l) for _h, l, c in HORIZONS]
    handles.append(Line2D([], [], marker="o", mfc="none", mec=C.MUTED, lw=0,
                          label="p<0.05 vs shift null"))
    leg = fig.legend(handles=handles, fontsize=7.5, framealpha=0.1, loc="upper right",
                     title="horizon")
    for tt in leg.get_texts():
        tt.set_color(C.TEXT)
    leg.get_title().set_color(C.TEXT)
    fig.suptitle(title, color=C.TEXT, fontsize=12)
    G._footnote(fig, G._base_note(
        "P(next event within H) as RELATIVE RISK by PPR quantile rank; shaded = "
        "circular-shift null 95%; rings = p<0.05"))
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    G._savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def risk_stratified_fig(mat, onsets, out_png, *, feature="ppr_peak_to_trough", n_bins=5,
                        n_surr=2000, horizons=(600.0, 1800.0), title="") -> str:
    """Relative risk vs WITHIN-STRATUM PPR rank, for day (06-18) vs night (18-06) epochs.
    If the low-PPR elevation survives within-stratum it is pre-ictal; if it collapses it
    is the time-of-day/vigilance state seizures prefer."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    t = mat["t_epoch"].to_numpy(float)
    hour = np.array([_dt.datetime.fromtimestamp(x).hour for x in t])
    strata = [("day 06–18", (hour >= 6) & (hour < 18)),
              ("night 18–06", (hour < 6) | (hour >= 18))]
    hcol = {h: c for h, l, c in HORIZONS}
    hlab = {h: l for h, l, c in HORIZONS}
    fig, axes = plt.subplots(1, len(strata), figsize=(5.4 * len(strata), 4.6),
                             facecolor=C.BG, squeeze=False)
    for ax, (snm, sm) in zip(axes[0], strata):
        res = risk_by_rank(mat, onsets, feature, n_bins=n_bins, n_surr=n_surr, keep=sm)
        x = np.arange(1, n_bins + 1)
        for H in horizons:
            d = res[H]
            ax.fill_between(x, d["lo"], d["hi"], color=hcol[H], alpha=0.13, lw=0)
            ax.plot(x, d["rr"], "-o", color=hcol[H], lw=1.7, ms=4)
            sig = d["p"] < 0.05
            if np.any(sig):
                ax.plot(x[sig], d["rr"][sig], "o", mfc="none", mec=hcol[H], ms=9, mew=1.6)
        ax.axhline(1.0, color=C.MUTED, lw=0.9, ls="--")
        b10 = res[horizons[0]]["base"]
        G._dark(ax, f"{snm}  (n={res['n']:,}, base {b10 * 100:.1f}%)")
        ax.set_xlabel("within-stratum PPR rank (1 = lowest)", color=C.TEXT)
        ax.set_xticks(x)
    axes[0][0].set_ylabel("relative risk  (P / stratum base)", color=C.TEXT)
    handles = [Line2D([], [], color=hcol[h], marker="o", label=hlab[h]) for h in horizons]
    leg = fig.legend(handles=handles, fontsize=8, framealpha=0.1, loc="upper right",
                     title="horizon")
    for tt in leg.get_texts():
        tt.set_color(C.TEXT)
    leg.get_title().set_color(C.TEXT)
    fig.suptitle(title, color=C.TEXT, fontsize=12)
    G._footnote(fig, G._base_note(f"{feature.replace('_', ' ')} · relative risk by "
                                  "WITHIN-stratum PPR rank; shaded = circular-shift null "
                                  "95% · time-of-day proxy for vigilance state"))
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    G._savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def run(*, since=None, force=False, n_surr=2000) -> dict:
    from .run import _ctx
    from src.preictal import isi as _isi
    store, ed = _ctx()
    mat = D.build_pp_matrix(store, ed, since=since, force=force)
    sz = _isi.scored_seizures(store, C.ANIMAL)
    t = mat["t_epoch"].to_numpy(float); lo, hi = t.min(), t.max()
    allon = np.sort(np.array([s.onset_epoch for s in sz], float))
    leadon = np.sort(np.array([s.onset_epoch for s in
                               _isi.leading_seizures(sz, 6 * 3600.0)], float))
    onsets = {"lead": leadon[(leadon >= lo) & (leadon <= hi)],
              "all": allon[(allon >= lo) & (allon <= hi)]}
    od = os.path.join(C.OUT_DIR, "periictal", "risk")
    os.makedirs(od, exist_ok=True)
    p = lambda n: os.path.join(od, n)                        # noqa: E731
    out = {}
    print(f"[paired_pulse.risk] {len(mat)} pairs; lead={onsets['lead'].size} "
          f"all={onsets['all'].size}", flush=True)
    for nm, ons in onsets.items():
        res = {f: risk_by_rank(mat, ons, f, n_surr=n_surr)
               for f, _ in _FEATS if f in mat.columns}
        out[f"rank_{nm}"] = risk_by_rank_fig(
            res, p(f"risk_by_rank_{nm}.png"),
            title=f"{C.ANIMAL} · {C.CHANNEL} · seizure risk by PPR quantile rank — "
            f"{nm} seizures (relative risk + shift null)")
        out[f"daynight_{nm}"] = risk_stratified_fig(
            mat, ons, p(f"risk_rank_daynight_{nm}.png"),
            title=f"{C.ANIMAL} · {C.CHANNEL} · risk by PPR rank, within day/night — "
            f"{nm} seizures")
        r10 = res.get("ppr_peak_to_trough", {}).get(600.0)
        if r10 is not None:
            print(f"[paired_pulse.risk] {nm} p2p 10-min RR by rank: "
                  f"{np.round(r10['rr'], 2)} p={np.round(r10['p'], 3)}", flush=True)
    for k, v in out.items():
        print(f"[paired_pulse.risk] {k} -> {v}", flush=True)
    return {"mat": mat, "out": out}
