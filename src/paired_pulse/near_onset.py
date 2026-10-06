"""Near-onset, PER-SEIZURE paired-pulse analyses (the 'what next' set).

 #1  S1 and S2 amplitude SEPARATELY over the last minutes per seizure. If S1 (the first
     pulse's response) rises while PPR falls, the network baseline excitability is already
     climbing -- the PPR drop would then be S2 saturating against a raised S1, not fresh
     disinhibition. One small-multiple panel per seizure.
 #3  10-s LINEAR bins over the last 5 min with the circular-shift null -- equal n per bin
     and matched resolution right at the edge (the 1-min/2-h sweep is coarse there).
 #4  PER-SEIZURE stats, not pooled pairs: each seizure's near-onset Spearman trend, and a
     direction-CONSISTENCY claim across the lead seizures (n=4 -> consistency, not power).

 (#2, raw LFP 60 s before each onset for detector latency, is NOT here: the evoked .mat
  holds only +/-500 ms windows around each stim, so 60 s of continuous LFP cannot be
  reconstructed from it -- that needs the source recording / video pipeline.)

Folder: data/BCH111_paired_pulse/periictal/near_onset/.
"""

from __future__ import annotations

import datetime as _dt
import os

import numpy as np

from . import config as C, data as D, figures as G, linear_bins as LB

_FEATS = [("s1_peak_to_trough", "S1 p2p", C.S1_COLOR),
          ("s2_peak_to_trough", "S2 p2p", C.S2_COLOR),
          ("ppr_peak_to_trough", "PPR p2p", C.ACCENT),
          ("ppr_line_length", "PPR line-len", C.FACIL_COLOR)]


# ---- #1: S1 & S2 amplitude separately, per seizure ------------------------------
def s1_s2_amplitude_fig(mat, onsets, out_png, *, win_min=10.0, bin_min=0.5,
                        title="") -> str:
    """Per-seizure small multiples: S1 and S2 p2p (each normalized to its own far-edge
    baseline) + PPR, over the last win_min before onset. Last 2 min shaded."""
    import matplotlib.pyplot as plt
    t = mat["t_epoch"].to_numpy(float)
    s1 = mat["s1_peak_to_trough"].to_numpy(float)
    s2 = mat["s2_peak_to_trough"].to_numpy(float)
    ppr = mat["ppr_peak_to_trough"].to_numpy(float)
    ons = np.sort(np.asarray(onsets, float))
    n = max(ons.size, 1)
    ncol = 2 if n <= 4 else 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.1 * nrow + 0.6),
                             facecolor=C.BG, squeeze=False)
    edges = np.arange(-win_min, bin_min, bin_min)
    cen = 0.5 * (edges[:-1] + edges[1:])
    nb = cen.size

    def _binned(x, mask, rel):
        arr = np.full(nb, np.nan)
        for b in range(nb):
            m = mask & (rel >= edges[b]) & (rel < edges[b + 1]) & np.isfinite(x)
            if m.sum() >= 2:
                arr[b] = np.median(x[m])
        return arr

    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i >= ons.size:
            ax.axis("off"); continue
        o = ons[i]
        rel = (t - o) / 60.0
        pi = np.searchsorted(ons, o, side="left") - 1          # approach-half, pre-only
        prev = ons[pi] if pi >= 0 else -np.inf
        base = (rel >= -win_min) & (rel < 0) & (t > 0.5 * (prev + o))
        b1, b2, bp = (_binned(s1, base, rel), _binned(s2, base, rel),
                      _binned(ppr, base, rel))
        far = slice(0, max(1, nb // 3))                         # far-edge baseline
        n1, n2 = np.nanmean(b1[far]), np.nanmean(b2[far])
        if np.isfinite(n1) and n1:
            ax.plot(cen, b1 / n1, color=C.S1_COLOR, lw=1.9, label="S1 (p2p)")
        if np.isfinite(n2) and n2:
            ax.plot(cen, b2 / n2, color=C.S2_COLOR, lw=1.9, label="S2 (p2p)")
        ax.axhline(1.0, color=C.MUTED, lw=0.6, ls=":")
        ax.axvspan(-2, 0, color=C.MUTED, alpha=0.13)
        ax.axvline(0, color=C.SEIZURE_COLOR, lw=1.0)
        axr = ax.twinx()
        axr.plot(cen, bp, color=C.SEIZURE_COLOR, lw=1.4, ls="--", label="PPR")
        axr.tick_params(colors=C.MUTED, labelsize=7)
        axr.set_ylabel("PPR", color=C.MUTED, fontsize=8)
        G._dark(ax, _dt.datetime.fromtimestamp(o).strftime("%m-%d %H:%M"))
        ax.set_ylabel("S1/S2 ÷ far baseline", color=C.TEXT, fontsize=8)
        if i // ncol == nrow - 1:
            ax.set_xlabel("minutes to onset", color=C.TEXT, fontsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C.S1_COLOR, lw=1.9),
               Line2D([], [], color=C.S2_COLOR, lw=1.9),
               Line2D([], [], color=C.SEIZURE_COLOR, lw=1.4, ls="--")]
    leg = fig.legend(handles, ["S1 (p2p)", "S2 (p2p)", "PPR"], fontsize=8,
                     framealpha=0.1, loc="upper right")
    for tt in leg.get_texts():
        tt.set_color(C.TEXT)
    fig.suptitle(title, color=C.TEXT, fontsize=12)
    G._footnote(fig, G._base_note(f"per-seizure S1/S2 p2p ÷ far-edge baseline + PPR; last "
                                  f"{win_min:.0f} min, {bin_min*60:.0f}-s bins; 2-min "
                                  f"window shaded; strict pre-ictal · {G._span_str(mat)}"))
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    G._savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---- #4: per-seizure trend, direction-consistency -------------------------------
def _per_seizure_rho(mat, onsets, feature, cap_sec):
    """Per-seizure Spearman ρ(feature, time-toward-onset) over strict pre-ictal epochs.
    Negative ρ = feature falls approaching onset. Returns (rhos[n], ns[n])."""
    from scipy.stats import spearmanr
    t = mat["t_epoch"].to_numpy(float)
    v = mat[feature].to_numpy(float)
    ons = np.sort(np.asarray(onsets, float))
    rhos, ns = [], []
    for o in ons:
        pi = np.searchsorted(ons, o, side="left") - 1
        prev = ons[pi] if pi >= 0 else -np.inf
        tto = o - t
        m = (tto > 0) & (tto <= cap_sec) & (t > 0.5 * (prev + o)) & np.isfinite(v)
        if m.sum() >= 20:
            rhos.append(float(spearmanr(v[m], -tto[m]).statistic))
            ns.append(int(m.sum()))
        else:
            rhos.append(np.nan); ns.append(int(m.sum()))
    return np.array(rhos), np.array(ns)


def per_seizure_stats_fig(mat, onsets, out_png, *, cap_h=2.0, title="") -> str:
    """Per-seizure near-onset trend ρ (feature falls toward onset = negative), one dot per
    seizure per feature, with the mean and a direction-consistency annotation (k/n same
    sign). n=4 lead seizures => a consistency claim, not a power claim."""
    import matplotlib.pyplot as plt
    ons = np.sort(np.asarray(onsets, float))
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=C.BG)
    cmap = G.TURBO_VIS
    xs = np.arange(len(_FEATS))
    summary = []
    for xi, (feat, lab, _c) in enumerate(_FEATS):
        if feat not in mat.columns:
            continue
        rhos, ns = _per_seizure_rho(mat, ons, feat, cap_h * 3600.0)
        good = np.isfinite(rhos)
        for si, r in enumerate(rhos):
            if np.isfinite(r):
                ax.scatter(xi + (si - ons.size / 2) * 0.04, r, s=42,
                           color=cmap(si / max(ons.size - 1, 1)), zorder=3,
                           edgecolors=C.BG, linewidths=0.5)
        if good.any():
            mr = np.nanmean(rhos)
            ax.plot([xi - 0.22, xi + 0.22], [mr, mr], color=C.SEIZURE_COLOR, lw=2.2)
            neg = int(np.sum(rhos[good] < 0)); tot = int(good.sum())
            ax.text(xi, 0.9, f"{neg}/{tot}↓", ha="center", va="top",
                    transform=ax.get_xaxis_transform(), color=C.TEXT, fontsize=8)
            summary.append(f"{lab}: mean ρ={mr:+.3f}, {neg}/{tot} fall")
    ax.axhline(0, color=C.MUTED, lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([l for _f, l, _c in _FEATS])
    G._dark(ax, title)
    ax.set_ylabel("Spearman ρ (feature vs time toward onset; <0 = falls)", color=C.TEXT)
    ax.set_xlim(-0.5, len(_FEATS) - 0.5)
    # per-seizure color legend
    for si, o in enumerate(ons):
        ax.scatter([], [], color=cmap(si / max(ons.size - 1, 1)),
                   label=_dt.datetime.fromtimestamp(o).strftime("%m-%d %H:%M"))
    leg = ax.legend(fontsize=7, framealpha=0.1, loc="center left",
                    bbox_to_anchor=(1.01, 0.5), title="seizure")
    for tt in leg.get_texts():
        tt.set_color(C.TEXT)
    leg.get_title().set_color(C.TEXT)
    G._footnote(fig, G._base_note(f"per-seizure Spearman over last {cap_h:.0f} h (strict "
                                  f"pre-ictal); white bar = mean; n={ons.size} => "
                                  f"consistency not power · {G._span_str(mat)}"))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    G._savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return summary, out_png


def periictal_zoom_fig(mat, onsets, out_png, *, feature="ppr_peak_to_trough",
                       pre_min=15.0, post_min=5.0, roll_min=1.0, step_min=0.5,
                       n_surr=400, title="") -> str:
    """Per-seizure zoom (−15→+5 min), one panel each: raw points + a 1-min rolling median
    over them + the circular-shift null 95% band behind. The clean visual test for a
    small near-onset sag -- if the rolling medians don't dip below the band near 0 in most
    panels, there's no pre-ictal drop."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    t = mat["t_epoch"].to_numpy(float); v = mat[feature].to_numpy(float)
    ok = np.isfinite(v); t, v = t[ok], v[ok]
    ons = np.sort(np.asarray(onsets, float))
    lo, hi = t.min(), t.max(); span = hi - lo; ms = 3 * 3600.0
    grid = np.arange(-pre_min, post_min + step_min, step_min)
    hw = roll_min / 2.0

    def _rollmed(center):
        a = np.searchsorted(t, center - (pre_min + 1) * 60)
        b = np.searchsorted(t, center + (post_min + 1) * 60)
        tl, vl = t[a:b], v[a:b]
        if tl.size < 3:
            return np.full(grid.size, np.nan)
        rel = (tl - center) / 60.0
        out = np.full(grid.size, np.nan)
        for i, g in enumerate(grid):
            msk = (rel >= g - hw) & (rel < g + hw)
            if msk.sum() >= 3:
                out[i] = np.median(vl[msk])
        return out

    n = ons.size; ncol = 3 if n > 4 else 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.0 * nrow + 0.6),
                             facecolor=C.BG, squeeze=False)
    rng = np.random.default_rng(0)
    for idx in range(nrow * ncol):
        ax = axes[idx // ncol][idx % ncol]
        if idx >= n:
            ax.axis("off"); continue
        o = ons[idx]
        rel = (t - o) / 60.0
        m = (rel >= -pre_min) & (rel <= post_min)
        null = np.full((n_surr, grid.size), np.nan)
        for s in range(n_surr):
            op = lo + ((o - lo + rng.uniform(ms, span - ms)) % span)
            null[s] = _rollmed(op)
        ax.fill_between(grid, np.nanpercentile(null, 2.5, axis=0),
                        np.nanpercentile(null, 97.5, axis=0), color=C.MUTED, alpha=0.28,
                        lw=0, label="shift-null 95%")
        if m.sum():
            ax.scatter(rel[m], v[m], s=6, color=C.ACCENT, alpha=0.30, linewidths=0,
                       rasterized=True)
            G._yclip(ax, v[m])
        ax.plot(grid, _rollmed(o), color=C.SEIZURE_COLOR, lw=2.2, label="1-min median")
        if feature.startswith("ppr"):
            ax.axhline(1.0, color=C.MUTED, lw=0.7, ls=":")
        ax.axvline(0, color=C.SEIZURE_COLOR, lw=1.1)
        G._dark(ax, _dt.datetime.fromtimestamp(o).strftime("%m-%d %H:%M"))
        if idx // ncol == nrow - 1:
            ax.set_xlabel("minutes from onset", color=C.TEXT, fontsize=8)
        if idx % ncol == 0:
            ax.set_ylabel(feature.replace("_", " "), color=C.TEXT, fontsize=8)
    handles = [Line2D([], [], color=C.SEIZURE_COLOR, lw=2.2, label="1-min rolling median"),
               Line2D([], [], color=C.MUTED, lw=7, alpha=0.4, label="shift-null 95%"),
               Line2D([], [], marker="o", color=C.ACCENT, lw=0, label="epoch")]
    leg = fig.legend(handles=handles, fontsize=8, framealpha=0.1, loc="upper right")
    for tt in leg.get_texts():
        tt.set_color(C.TEXT)
    fig.suptitle(title, color=C.TEXT, fontsize=12)
    G._footnote(fig, G._base_note(f"per-seizure −{pre_min:.0f}→+{post_min:.0f} min; "
                                  f"{roll_min:.0f}-min rolling median over points; shaded "
                                  "= circular-shift null 95% · " + G._span_str(mat)))
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    G._savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


_TREND_FEATS = [("ppr_peak_to_trough", "PPR p2p"), ("ppr_rms_amplitude", "PPR rms")]


def trend_statistic_fig(mat, lead, allon, out_png, *, n_surr=2000, title="") -> tuple:
    """Last-10-min minus baseline (30 min-2 h) contrast per onset-set, tested against BOTH
    a standard circular-shift null AND a TIME-OF-DAY-MATCHED (circadian) null -- surrogate
    onsets shifted by ~k·24 h (±2 h), so each keeps the original clock time / circadian
    phase. If the drop survives the circadian null it is beyond a vigilance-state confound.
    Per-seizure contrasts shown as coloured dots. Excludes the last 30 s (onset leakage)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    t = mat["t_epoch"].to_numpy(float)
    lo, hi = t.min(), t.max(); span = hi - lo; ms = 3 * 3600.0
    rng = np.random.default_rng(0)

    def _su():
        return rng.uniform(ms, span - ms)

    def _sc():                                              # circadian: ~k*24h +/- 2h
        kmax = int((span - ms) // 86400.0)
        if kmax < 1:
            return _su()
        for _ in range(200):
            s = int(rng.integers(1, kmax + 1)) * 86400.0 + rng.uniform(-7200.0, 7200.0)
            if ms <= s <= span - ms:
                return s
        return _su()

    def _contrast(ons, v):
        ons = np.sort(ons)
        i = np.searchsorted(ons, t, "left")
        nxt = np.where(i < ons.size, ons[np.clip(i, 0, ons.size - 1)], np.inf)
        tto = nxt - t
        pi = np.searchsorted(ons, t, "right") - 1
        prev = np.where(pi >= 0, ons[np.clip(pi, 0, ons.size - 1)], -np.inf)
        ok = (t - prev > 3600.0) & np.isfinite(v)
        n = v[ok & (tto > 30.0) & (tto <= 600.0)]
        b = v[ok & (tto > 1800.0) & (tto <= 7200.0)]
        return (np.mean(n) - np.mean(b)) if n.size >= 10 and b.size >= 10 else np.nan

    fig, ax = plt.subplots(figsize=(9.2, 5.3), facecolor=C.BG)
    cmap = G.TURBO_VIS
    xt, xl, xi, summ, allv = [], [], 0, [], []
    for feat, flab in _TREND_FEATS:
        v = mat[feat].to_numpy(float)
        for nm, ons in (("lead", lead), ("all", allon)):
            ons = np.sort(ons); obs = _contrast(ons, v)
            nu = np.array([_contrast(lo + ((ons - lo + _su()) % span), v)
                           for _ in range(n_surr)]); nu = nu[np.isfinite(nu)]
            nc = np.array([_contrast(lo + ((ons - lo + _sc()) % span), v)
                           for _ in range(n_surr)]); nc = nc[np.isfinite(nc)]

            def _p(nl, o=obs):
                m = np.median(nl)
                return (np.sum(np.abs(nl - m) >= abs(o - m)) + 1) / (nl.size + 1)
            pu, pc = _p(nu), _p(nc)
            ax.plot([xi - 0.16] * 2, np.percentile(nu, [2.5, 97.5]), color=C.MUTED,
                    lw=7, alpha=0.5, solid_capstyle="round")
            ax.plot([xi + 0.16] * 2, np.percentile(nc, [2.5, 97.5]), color=C.ACCENT,
                    lw=7, alpha=0.5, solid_capstyle="round")
            for j, o in enumerate(ons):
                tt = o - t
                n = v[(tt > 30.0) & (tt <= 600.0) & np.isfinite(v)]
                b = v[(tt > 1800.0) & (tt <= 7200.0) & np.isfinite(v)]
                if n.size >= 8 and b.size >= 8:
                    cv = np.mean(n) - np.mean(b); allv.append(cv)
                    ax.scatter(xi + rng.uniform(-0.05, 0.05), cv, s=26,
                               color=cmap(j / max(ons.size - 1, 1)), zorder=3,
                               edgecolors=C.BG, linewidths=0.4)
            ax.scatter(xi, obs, marker="D", s=95, color=C.SEIZURE_COLOR, zorder=5,
                       edgecolors=C.BG, linewidths=0.6)
            xt.append(xi); xl.append(f"{flab}\n{nm}")
            summ.append(f"{flab} {nm}: obs={obs:+.4f} p_shift={pu:.3f} p_circ={pc:.3f}")
            ax.annotate(f"p$_s$={pu:.2f}\np$_c$={pc:.2f}", (xi, 1.0),
                        xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=7, color=C.TEXT)
            xi += 1
    ax.axhline(0, color=C.MUTED, lw=0.9)
    if allv:
        G._yclip(ax, np.array(allv + [0.0]))
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_xlim(-0.5, xi - 0.5)
    G._dark(ax, title)
    ax.set_ylabel("last 10 min − baseline (30 min–2 h)   [<0 = PPR lower near onset]",
                  color=C.TEXT, fontsize=9)
    handles = [Line2D([], [], color=C.MUTED, lw=7, alpha=0.5, label="shift-null 95%"),
               Line2D([], [], color=C.ACCENT, lw=7, alpha=0.5, label="circadian-null 95%"),
               Line2D([], [], marker="D", color=C.SEIZURE_COLOR, lw=0, label="observed"),
               Line2D([], [], marker="o", color=C.MUTED, lw=0, label="per seizure")]
    leg = ax.legend(handles=handles, fontsize=7.5, framealpha=0.1, loc="lower left")
    for tt in leg.get_texts():
        tt.set_color(C.TEXT)
    G._footnote(fig, G._base_note("last-10-min−baseline contrast vs circular-shift AND "
                                  "time-of-day-matched (circadian) nulls; last 30 s "
                                  "excluded · " + G._span_str(mat)))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    G._savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return summ, out_png


def run(*, since=None, force=False, n_surr=1000) -> dict:
    """Generate the near-onset per-seizure analyses into periictal/near_onset/."""
    from src.preictal import isi as _isi
    from .run import _GROUPS, _GROUP_NAME, _ctx, _label
    store, ed = _ctx()
    mat = D.build_pp_matrix(store, ed, since=since, force=force)
    labels = {c: _label(c) for g in _GROUPS.values() for c in g}
    od = os.path.join(C.OUT_DIR, "periictal", "near_onset")
    os.makedirs(od, exist_ok=True)
    p = lambda n: os.path.join(od, n)                        # noqa: E731
    sz = _isi.scored_seizures(store, C.ANIMAL)
    t = mat["t_epoch"].to_numpy(float); lo, hi = t.min(), t.max()
    allon = np.sort(np.array([s.onset_epoch for s in sz], float))
    leadon = np.sort(np.array([s.onset_epoch for s in
                               _isi.leading_seizures(sz, 6 * 3600.0)], float))
    onsets = {"lead": leadon[(leadon >= lo) & (leadon <= hi)],
              "all": allon[(allon >= lo) & (allon <= hi)]}
    out = {}
    print(f"[paired_pulse.near_onset] {len(mat)} pairs; lead={onsets['lead'].size} "
          f"all={onsets['all'].size}", flush=True)

    # #1 S1 & S2 separately, per seizure
    for oset, ons in onsets.items():
        out[f"s1s2_{oset}"] = s1_s2_amplitude_fig(
            mat, ons, p(f"s1_s2_amplitude_{oset}.png"), win_min=10.0, bin_min=0.5,
            title=f"{C.ANIMAL} · {C.CHANNEL} · S1 & S2 amplitude approaching onset — "
            f"{oset} seizures")

    # #3 10-s linear bins over the last 5 min, with the shift null
    for g, cols in _GROUPS.items():
        cols = [c for c in cols if c in mat.columns]
        for oset, ons in onsets.items():
            res = {c: LB.trend_null(mat, ons, c, 10.0 / 60.0, cap_h=5.0 / 60.0,
                                    n_surr=n_surr) for c in cols}
            out[f"{g}_{oset}_near5min"] = G.trend_null_fig(
                res, p(f"{g}_{oset}_trend_null_10s_5min.png"), labels=labels,
                ref1=(g == "ppr"), logx=False, xlabel="min to onset (10-s linear bins)",
                title=f"{C.ANIMAL} · {C.CHANNEL} · last 5 min vs shift null "
                f"(10-s linear bins) — {_GROUP_NAME[g]} · {oset} seizures")

    # #4 per-seizure stats + direction-consistency
    for oset, ons in onsets.items():
        summ, fp = per_seizure_stats_fig(
            mat, ons, p(f"per_seizure_stats_{oset}.png"), cap_h=2.0,
            title=f"{C.ANIMAL} · {C.CHANNEL} · per-seizure near-onset trend — "
            f"{oset} seizures")
        out[f"perseiz_{oset}"] = fp
        print(f"[paired_pulse.near_onset] {oset} consistency: " + "; ".join(summ),
              flush=True)

    # #5 last-10-min trend statistic vs shift-null AND circadian (time-of-day) null
    tsumm, tfp = trend_statistic_fig(
        mat, onsets["lead"], onsets["all"], p("trend_statistic_dual_null.png"),
        n_surr=2000, title=f"{C.ANIMAL} · {C.CHANNEL} · last-10-min PPR drop vs shift "
        f"& circadian nulls")
    out["trend_statistic"] = tfp
    print("[paired_pulse.near_onset] trend-stat: " + "; ".join(tsumm), flush=True)

    # #6 per-seizure zoom (-15..+5 min): points + 1-min rolling median + shift-null band
    for feat, flab in (("ppr_peak_to_trough", "p2p"), ("ppr_rms_amplitude", "rms")):
        out[f"zoom_lead_{flab}"] = periictal_zoom_fig(
            mat, onsets["lead"], p(f"periictal_zoom_lead_{flab}.png"), feature=feat,
            title=f"{C.ANIMAL} · {C.CHANNEL} · per-seizure zoom (−15→+5 min) PPR {flab} "
            f"— lead seizures")

    for k, v in out.items():
        print(f"[paired_pulse.near_onset] {k} -> {v}", flush=True)
    return {"mat": mat, "out": out}
