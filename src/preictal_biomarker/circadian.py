"""Confirm (or reject) a circadian rhythm in the evoked response BEFORE treating
it as structure to model around. Rhythm must be distinguished from slow drift and
from lab-schedule artifacts; a prior (evoked responses track vigilance/circadian
state) is not a measurement in THIS animal.

Tests (each on PC1, selected features, and per-state occupancy over the full
24/7 record):
  1. Lomb-Scargle periodogram vs an AR(1)-surrogate null -> is there a 24 h (and
     12 h) peak ABOVE what slow drift alone produces? (rhythm vs trend)
  2. Cosinor (24 h) fit -> amplitude + FRACTION OF VARIANCE explained, with a
     block-bootstrap CI (a significant but tiny rhythm need not be modelled).
  3. Day-to-day repeatability -> clock-hour x day profile; mean inter-day
     correlation (a rhythm repeats, a trend does not).
  4. External anchor -> seizure onset clock-hour distribution (do seizures and the
     feature rhythm share a phase? if so the cycle is a CONFOUNDED covariate, not
     something to regress out).

Decision rule written into the figures: circadian is CONFIRMED for a signal only
if the 24 h LS peak clears the AR(1) null AND the daily profile repeats
(inter-day r well above its own shuffle). Otherwise it is drift / unconfirmed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

from . import config as C

_HOUR = 3600.0
_DAY = 86400.0


def _ar1_surrogate(x: np.ndarray, rng) -> np.ndarray:
    """AR(1) surrogate preserving mean, variance and lag-1 autocorr (a drift/
    red-noise null) but NO periodicity."""
    x = np.asarray(x, float)
    n = x.size
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    phi = float(np.dot(xc[:-1], xc[1:]) / denom) if denom > 0 else 0.0
    phi = max(min(phi, 0.999), -0.999)
    sd = float(np.std(xc) * np.sqrt(max(1e-9, 1 - phi ** 2)))
    e = rng.normal(0, sd, n)
    y = np.empty(n)
    y[0] = xc[0]
    for i in range(1, n):                                   # bounded loop
        y[i] = phi * y[i - 1] + e[i]
    return y + x.mean()


def _ls_power(t_sec, x, periods_h):
    """Lomb-Scargle power at each period (hours). t in seconds, irregular OK."""
    t = np.asarray(t_sec, float)
    x = np.asarray(x, float)
    m = np.isfinite(t) & np.isfinite(x)
    t, x = t[m], x[m]
    if t.size < 20:
        return np.full(len(periods_h), np.nan)
    x = x - x.mean()
    ang = 2.0 * np.pi / (np.asarray(periods_h, float) * _HOUR)   # rad/sec
    return lombscargle(t, x, ang, normalize=True)


def periodogram_test(t_sec, x, *, n_surr=200, seed=0,
                     periods_h=None) -> dict:
    """LS power spectrum + AR(1)-surrogate 95% null. p at 24 h = fraction of
    surrogates whose 24 h power >= observed."""
    periods_h = periods_h or list(np.geomspace(2, 72, 80))
    obs = _ls_power(t_sec, x, periods_h)
    # restrict surrogates to the regularly-ordered series (AR(1) on value order)
    x = np.asarray(x, float)
    order = np.argsort(np.asarray(t_sec, float))
    xo = x[order]
    rng = np.random.default_rng(seed)
    null = np.full((n_surr, len(periods_h)), np.nan)
    for i in range(n_surr):
        null[i] = _ls_power(np.asarray(t_sec, float)[order],
                            _ar1_surrogate(xo, rng), periods_h)
    hi = np.nanpercentile(null, 95, axis=0)
    p24 = _peak_p(periods_h, obs, null, 24.0)
    p12 = _peak_p(periods_h, obs, null, 12.0)
    return {"periods_h": np.asarray(periods_h, float), "power": obs,
            "null_hi": hi, "p24": p24, "p12": p12,
            "peak_period_h": float(periods_h[int(np.nanargmax(obs))])
            if np.isfinite(obs).any() else np.nan}


def _peak_p(periods_h, obs, null, target_h):
    j = int(np.argmin(np.abs(np.asarray(periods_h) - target_h)))
    o = obs[j]
    col = null[:, j]
    col = col[np.isfinite(col)]
    if not np.isfinite(o) or col.size == 0:
        return np.nan
    return float((np.sum(col >= o) + 1) / (col.size + 1))


def cosinor(t_sec, x, *, period_h=24.0, n_boot=1000, seed=0) -> dict:
    """24 h cosinor: x ~ M + A cos(2πt/P − φ). Returns mesor, amplitude, acrophase
    (clock h of peak), fraction-of-variance (R²) and a block-bootstrap amplitude CI."""
    t = np.asarray(t_sec, float); x = np.asarray(x, float)
    m = np.isfinite(t) & np.isfinite(x)
    t, x = t[m], x[m]
    w = 2 * np.pi / (period_h * _HOUR)
    A = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    fit = A @ beta
    r2 = 1.0 - np.var(x - fit) / (np.var(x) + 1e-12)
    amp = float(np.hypot(beta[1], beta[2]))
    acro = (-np.arctan2(beta[2], beta[1])) % (2 * np.pi)
    acro_h = float(acro / (2 * np.pi) * period_h)
    rng = np.random.default_rng(seed)
    amps = np.empty(n_boot)
    blk = max(50, t.size // 50)
    for b in range(n_boot):                                 # block bootstrap
        idx = _block_idx(t.size, blk, rng)
        bb, *_ = np.linalg.lstsq(A[idx], x[idx], rcond=None)
        amps[b] = np.hypot(bb[1], bb[2])
    return {"mesor": float(beta[0]), "amplitude": amp, "acrophase_h": acro_h,
            "r2": float(r2), "amp_lo": float(np.percentile(amps, 2.5)),
            "amp_hi": float(np.percentile(amps, 97.5))}


def _block_idx(n, blk, rng):
    out = []
    while len(out) < n:
        s = int(rng.integers(0, max(1, n - blk)))
        out.extend(range(s, min(n, s + blk)))
    return np.array(out[:n])


def daily_profile(t_sec, x, *, n_hours=24) -> dict:
    """[day x hour] mean matrix + mean inter-day correlation (repeatability)."""
    t = np.asarray(t_sec, float); x = np.asarray(x, float)
    m = np.isfinite(t) & np.isfinite(x)
    t, x = t[m], x[m]
    day = np.floor((t - t.min()) / _DAY).astype(int)
    hour = ((t % _DAY) / _HOUR).astype(int) % n_hours
    days = np.unique(day)
    M = np.full((days.size, n_hours), np.nan)
    for r, d in enumerate(days):
        for h in range(n_hours):
            v = x[(day == d) & (hour == h)]
            if v.size:
                M[r, h] = v.mean()
    cc = []
    for i in range(len(days)):
        for j in range(i + 1, len(days)):
            a, b = M[i], M[j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 6:
                cc.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return {"matrix": M, "days": days, "interday_r": float(np.nanmean(cc))
            if cc else np.nan, "n_pairs": len(cc)}


def seizure_clock(onsets) -> np.ndarray:
    """Clock-hour (0-23) of each seizure onset."""
    import datetime as _dt
    return np.array([_dt.datetime.fromtimestamp(float(o)).hour for o in onsets])


# --------------------------- figure + orchestration ------------------------- #

def circadian_report_fig(res: dict, clock_hours, out_png: str, *,
                         anchor="PC1") -> str:
    """4-panel confirmation report for the anchor signal + a cross-signal cosinor
    %-var summary + the seizure clock-hour distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .figures import _savefig, _dark
    a = res[anchor]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), facecolor=C.BG)
    # (1) periodogram
    p = a["periodo"]
    ax0 = ax[0, 0]
    ax0.semilogx(p["periods_h"], p["power"], color=C.ACCENT, lw=1.6, label=anchor)
    ax0.semilogx(p["periods_h"], p["null_hi"], color="#ff6b6b", lw=1.2, ls="--",
                 label="AR(1) null 95%")
    for ph in (24, 12):
        ax0.axvline(ph, color=C.MUTED, lw=0.8, ls=":")
    _dark(ax0, f"Lomb-Scargle periodogram ({anchor}) · p(24h)={p['p24']:.3f} "
               f"p(12h)={p['p12']:.3f}")
    ax0.set_xlabel("period (h)"); ax0.set_ylabel("LS power")
    leg = ax0.legend(fontsize=8, framealpha=0.1)
    for tx in leg.get_texts():
        tx.set_color(C.TEXT)
    # (2) cosinor %var across signals
    ax1 = ax[0, 1]
    names = list(res); r2 = [100 * res[n]["cosinor"]["r2"] for n in names]
    ax1.bar(range(len(names)), r2, color=C.ACCENT)
    _dark(ax1, "24h cosinor · fraction of variance explained")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace("_", " ") for n in names], rotation=25,
                        ha="right", fontsize=8)
    ax1.set_ylabel("% variance (R²)")
    for i, n in enumerate(names):
        ax1.text(i, r2[i] + 0.3, f"{r2[i]:.1f}%", ha="center", color=C.TEXT,
                 fontsize=7)
    # (3) daily profile heatmap
    ax2 = ax[1, 0]
    M = a["daily"]["matrix"]
    im = ax2.imshow(M, aspect="auto", cmap="magma", origin="lower",
                    extent=[0, 24, 0, M.shape[0]])
    _dark(ax2, f"daily profile ({anchor}) · inter-day r={a['daily']['interday_r']:.2f}"
               f" (n={a['daily']['n_pairs']} day-pairs)")
    ax2.set_xlabel("clock hour"); ax2.set_ylabel("day")
    cb = fig.colorbar(im, ax=ax2, fraction=0.045); cb.ax.tick_params(colors=C.MUTED)
    # (4) seizure clock distribution + anchor acrophase
    ax3 = ax[1, 1]
    ax3.hist(clock_hours, bins=np.arange(0, 25), color="#f2c744",
             edgecolor=C.BG, align="left")
    ax3.axvline(a["cosinor"]["acrophase_h"], color=C.ACCENT, lw=2.0,
                label=f"{anchor} acrophase ({a['cosinor']['acrophase_h']:.1f}h)")
    _dark(ax3, "seizure onset clock-hour (vs feature acrophase)")
    ax3.set_xlabel("clock hour"); ax3.set_ylabel("# seizures")
    leg = ax3.legend(fontsize=8, framealpha=0.1)
    for tx in leg.get_texts():
        tx.set_color(C.TEXT)
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · CIRCADIAN CONFIRMATION "
                 "(rhythm vs drift, before any regression)", color=C.TEXT,
                 fontsize=13)
    import os
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def run_confirmation(*, n_surr=200) -> dict:
    """Load the full-record matrix, fit states (for PC1), run the four tests on
    PC1 + key features + state occupancy, render the report, print the verdict."""
    import os
    from . import features as F, states as S, pipeline as P
    store, ed, db = P._ctx()
    df = F.build_feature_matrix(store, ed, db)             # full 24/7 record
    model = S.fit_states(df)
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    sig = {"PC1": model.df["pc1"].to_numpy(float)}
    for f in ("rms_amplitude", "autocorrelation", "csd_ar1"):
        if f in df.columns:
            sig[f] = df[f].to_numpy(float)
    res = {}
    for name, x in sig.items():
        _log(f"circadian tests on {name} ...")
        tb, xb = _bin_series(t, x, bin_sec=1800.0)        # 30-min means for the LS
        res[name] = {"periodo": periodogram_test(tb, xb, n_surr=n_surr),
                     "cosinor": cosinor(t, x), "daily": daily_profile(t, x)}
    clock = seizure_clock(F.scoped_onsets(store))
    out = os.path.join(C.EVOKED_DIR, "10_circadian_confirmation.png")
    circadian_report_fig(res, clock, out)
    print("\n=== CIRCADIAN CONFIRMATION VERDICT ===")
    for name, r in res.items():
        pr, co, da = r["periodo"], r["cosinor"], r["daily"]
        verdict = ("CONFIRMED" if (pr["p24"] < 0.05 and da["interday_r"] > 0.3)
                   else "drift/unconfirmed")
        print(f"  {name:16s} peak={pr['peak_period_h']:.1f}h p24={pr['p24']:.3f} "
              f"cosinorR2={100*co['r2']:.1f}% interdayR={da['interday_r']:.2f} "
              f"-> {verdict}")
    print(f"  seizure clock-hours: {np.bincount(clock, minlength=24).tolist()}")
    print(f"  figure -> {out}")
    return {"res": res, "clock": clock, "out": out}


def _bin_series(t, x, *, bin_sec=1800.0):
    """Regular-grid mean series (for the periodogram): bin by *bin_sec* and take
    the mean per occupied bin."""
    t = np.asarray(t, float); x = np.asarray(x, float)
    m = np.isfinite(t) & np.isfinite(x)
    t, x = t[m], x[m]
    b = ((t - t.min()) // bin_sec).astype(int)
    s = pd.DataFrame({"b": b, "x": x}).groupby("b")["x"].mean()
    tb = t.min() + s.index.to_numpy() * bin_sec + bin_sec / 2.0
    return tb, s.to_numpy()


def _log(m):
    print(f"[preictal_biomarker.circadian] {m}", flush=True)


if __name__ == "__main__":
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    run_confirmation()
