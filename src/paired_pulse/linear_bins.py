"""Linear-time-bin, STRICT pre-ictal variants of the binned / log-timescale paired-pulse
figures, swept over bin widths 1 / 5 / 10 / 30 min on a fixed pre-ictal window.

Two things differ from the main peri-ictal figures:

1.  **Linear bins.** The log-timescale trends here are binned with FIXED-WIDTH minute
    bins (not dyadic/log bins). Small shifts in the near-onset mean then read off at a
    constant resolution; the x-axis stays log for the "approach to onset" view.
2.  **Strict pre-ictal (blank onset + all post-onset).** An epoch counts for a seizure
    only if the next onset is ahead (time-to-onset > 0) AND that onset is nearer than
    the previous one (approach-half rule). This blanks the onset itself and every
    post-onset sample, including a prior seizure's recovery half -- no bleed.

Everything lands in a NEW folder: data/BCH111_paired_pulse/periictal/linear_bins/.
"""

from __future__ import annotations

import os

import numpy as np

from . import config as C, data as D, figures as G

BIN_MIN = (1.0, 5.0, 10.0, 30.0)        # minute bin widths to sweep
CAP_H = 2.0                             # pre-ictal window depth (hours)


def _lin_edges(bin_min, cap_h=CAP_H):
    """Linear seconds edges [0, bin, 2*bin, ..., cap]."""
    return np.arange(0.0, cap_h * 3600.0 + bin_min * 60.0, bin_min * 60.0)


def _preictal(t, onsets, cap_sec):
    """Strict pre-ictal mask + time-to-next-onset. Keep an epoch iff the next onset is
    ahead (tto > 0), within cap, AND nearer than the previous onset (tto <= tsl) -- which
    blanks the onset and ALL post-onset data (a prior seizure's recovery half included)."""
    ons = np.sort(np.asarray(onsets, float))
    if not ons.size:
        return np.zeros(t.size, bool), np.full(t.size, np.inf)
    idx = np.searchsorted(ons, t, side="left")
    nxt = np.where(idx < ons.size, ons[np.clip(idx, 0, ons.size - 1)], np.inf)
    tto = nxt - t
    pidx = np.searchsorted(ons, t, side="right") - 1
    prev = np.where(pidx >= 0, ons[np.clip(pidx, 0, ons.size - 1)], -np.inf)
    tsl = t - prev
    return (tto > 0) & (tto <= cap_sec) & (tto <= tsl), tto


def trend_null(mat, onsets, feature, bin_min, *, cap_h=CAP_H, n_surr=200,
               min_shift_h=3.0, seed=0) -> dict:
    """Pre-ictal trend of *feature* vs time-to-onset in LINEAR bins (strict pre-ictal),
    with a circular-shift null band. Same output contract as data.trend_shift_null, so
    figures.trend_null_fig renders it unchanged."""
    t = mat["t_epoch"].to_numpy(float)
    v = mat[feature].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    ons = np.sort(np.asarray(onsets, float))
    edges = _lin_edges(bin_min, cap_h)
    nb = edges.size - 1
    cen = 0.5 * (edges[:-1] + edges[1:]) / 60.0             # linear centers (min)
    cap = cap_h * 3600.0

    def _stat(o):
        m, tto = _preictal(t, o, cap)
        if not m.any():
            return np.full(nb, np.nan)
        bi = np.clip(np.searchsorted(edges, tto[m], "right") - 1, 0, nb - 1)
        sv = np.bincount(bi, weights=v[m], minlength=nb)
        cv = np.bincount(bi, minlength=nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            return sv / np.where(cv > 0, cv, np.nan)

    obs = _stat(ons)
    lo_t, hi_t = t.min(), t.max()
    span = hi_t - lo_t
    ms = min_shift_h * 3600.0
    rng = np.random.default_rng(seed)
    null = np.full((int(n_surr), nb), np.nan)
    for i in range(int(n_surr)):
        null[i] = _stat(lo_t + ((ons - lo_t + rng.uniform(ms, span - ms)) % span))
    lo, hi, med, p = D._band_p(null, obs)
    return {"centers": cen, "obs": obs, "lo": lo, "hi": hi, "med": med, "p": p,
            "n_surr": int(n_surr), "feature": feature}


def waveforms(store, evoked_dir, *, bin_sizes=BIN_MIN, cap_h=CAP_H, since=None,
              channel=None) -> dict:
    """Averaged S1/S2/residual waveforms per LINEAR lead-time bin (strict pre-ictal), for
    every bin width in ONE file-read pass. Returns {bin_min: wb} where wb has the same
    shape data.waveform_by_leadtime returns (twin, centers, edges, lead/all dicts)."""
    from src.preictal import isi as _isi
    channel = channel or C.CHANNEL
    sz = _isi.scored_seizures(store, C.ANIMAL)
    ons = {"all": np.sort(np.array([s.onset_epoch for s in sz], float)),
           "lead": np.sort(np.array([s.onset_epoch for s in
                                     _isi.leading_seizures(sz, 6 * 3600.0)], float))}
    edges = {bm: _lin_edges(bm, cap_h) for bm in bin_sizes}
    nb = {bm: edges[bm].size - 1 for bm in bin_sizes}
    acc = {bm: {k: {"s1": None, "s2": None, "cnt": np.zeros(nb[bm])} for k in ons}
           for bm in bin_sizes}
    twin = None
    cap = cap_h * 3600.0
    for fp, _d in D.list_pp_files(evoked_dir, since):
        r = D._read_file(fp, channel)
        if r is None:
            continue
        lfp, stim, tax, fs, st = r
        s1_on, s2_on, paired = D._pulse_onsets(stim, tax)
        if paired.sum() < 5:
            continue
        lfp_f = D._filter(lfp, fs)
        s1w, twin = D._window_array(lfp_f, tax, s1_on, fs)
        s2w, _ = D._window_array(lfp_f, tax, s2_on, fs)
        base = D._file_dt(fp).timestamp()
        te = base + st if (st is not None and st.size == lfp.shape[0]) else \
            np.full(lfp.shape[0], base)
        for k, o in ons.items():
            if not o.size:
                continue
            m0, tto = _preictal(te, o, cap)
            m0 = m0 & paired
            if not m0.any():
                continue
            for bm in bin_sizes:
                bi = np.clip(np.searchsorted(edges[bm], tto[m0], "right") - 1, 0,
                             nb[bm] - 1)
                if acc[bm][k]["s1"] is None:
                    acc[bm][k]["s1"] = np.zeros((nb[bm], s1w.shape[1]))
                    acc[bm][k]["s2"] = np.zeros((nb[bm], s2w.shape[1]))
                np.add.at(acc[bm][k]["s1"], bi, s1w[m0])
                np.add.at(acc[bm][k]["s2"], bi, s2w[m0])
                np.add.at(acc[bm][k]["cnt"], bi, 1)
    out = {}
    for bm in bin_sizes:
        cen = 0.5 * (edges[bm][:-1] + edges[bm][1:]) / 60.0
        d = {"twin": twin, "centers": cen, "edges": edges[bm]}
        for k in ons:
            c = acc[bm][k]["cnt"]
            dd = np.where(c[:, None] > 0, c[:, None], np.nan)
            s1m = acc[bm][k]["s1"] / dd if acc[bm][k]["s1"] is not None else None
            s2m = acc[bm][k]["s2"] / dd if acc[bm][k]["s2"] is not None else None
            d[k] = {"s1": s1m, "s2": s2m,
                    "resid": (s2m - s1m) if s2m is not None else None, "cnt": c}
        out[bm] = d
    return out


def run(*, since=None, force=False, n_surr=200, bin_sizes=BIN_MIN, cap_h=CAP_H) -> dict:
    """Generate the linear-bin, strict-pre-ictal figure sweep into periictal/linear_bins/:
    per bin width {1,5,10,30 min} -> trend+null (S1/S2/PPR x lead/all) + pre-ictal PPR
    trajectory; plus the S2/residual waveform-by-bin (one file pass); plus proximity ECDFs
    (binning-free). STRICT pre-ictal everywhere: the onset and all post-onset data are
    blanked."""
    from src.preictal import isi as _isi
    from .run import _GROUPS, _GROUP_NAME, _ctx, _label
    store, ed = _ctx()
    mat = D.build_pp_matrix(store, ed, since=since, force=force)
    labels = {c: _label(c) for g in _GROUPS.values() for c in g}
    od = os.path.join(C.OUT_DIR, "periictal", "linear_bins")
    os.makedirs(od, exist_ok=True)
    p = lambda n: os.path.join(od, n)                        # noqa: E731
    sz = _isi.scored_seizures(store, C.ANIMAL)
    t = mat["t_epoch"].to_numpy(float)
    lo, hi = t.min(), t.max()
    allon = np.sort(np.array([s.onset_epoch for s in sz], float))
    leadon = np.sort(np.array([s.onset_epoch for s in
                               _isi.leading_seizures(sz, 6 * 3600.0)], float))
    onsets = {"lead": leadon[(leadon >= lo) & (leadon <= hi)],
              "all": allon[(allon >= lo) & (allon <= hi)]}
    out = {}
    print(f"[paired_pulse.linbins] {len(mat)} pairs; bins {[int(b) for b in bin_sizes]} "
          f"min; cap {cap_h:.0f} h; n_surr {n_surr}", flush=True)

    # proximity ECDFs (no binning) -------------------------------------------------
    for oset, ons in onsets.items():
        out[f"ecdf_prox_{oset}"] = G.ppr_ecdf_proximity_fig(
            mat, ons, p(f"ppr_ecdf_proximity_{oset}.png"),
            title=f"{C.ANIMAL} · {C.CHANNEL} · PPR distribution shift by seizure "
            f"proximity (ECDF) — {oset} seizures")

    # trend + null and the pre-ictal trajectory, per bin width ---------------------
    for bm in bin_sizes:
        tag = f"{int(bm)}min"
        for g, cols in _GROUPS.items():
            cols = [c for c in cols if c in mat.columns]
            for oset, ons in onsets.items():
                res = {c: trend_null(mat, ons, c, bm, cap_h=cap_h, n_surr=n_surr)
                       for c in cols}
                out[f"{g}_{oset}_trend_null_{tag}"] = G.trend_null_fig(
                    res, p(f"{g}_{oset}_trend_null_{tag}.png"), labels=labels,
                    ref1=(g == "ppr"),
                    title=f"{C.ANIMAL} · {C.CHANNEL} · pre-ictal trend vs shift null "
                    f"({int(bm)}-min linear bins) — {_GROUP_NAME[g]} · {oset} seizures")
        for oset, ons in onsets.items():
            out[f"ppr_traj_{oset}_{tag}"] = G.periictal_trajectory_fig(
                mat, ons, p(f"ppr_traj_{oset}_{tag}.png"),
                feature="ppr_peak_to_trough", pre_h=cap_h, post_h=0.0, bin_min=bm,
                strict=True,
                title=f"{C.ANIMAL} · {C.CHANNEL} · pre-ictal PPR (p2p), {int(bm)}-min "
                f"linear bins — {oset} seizures (n={ons.size}; 1 = equal)")

    # waveform-by-bin for every bin width, in one file pass ------------------------
    wf = waveforms(store, ed, bin_sizes=bin_sizes, cap_h=cap_h, since=since)
    for bm in bin_sizes:
        tag = f"{int(bm)}min"
        for oset in ("lead", "all"):
            out[f"s2_wave_{oset}_{tag}"] = G.waveform_by_bin_fig(
                wf[bm], p(f"s2_waveform_by_bin_{oset}_{tag}.png"), onset_set=oset,
                which="s2", title=f"{C.ANIMAL} · {C.CHANNEL} · averaged S2 per pre-ictal "
                f"bin ({int(bm)}-min linear) — {oset} seizures")
            out[f"resid_wave_{oset}_{tag}"] = G.waveform_by_bin_fig(
                wf[bm], p(f"residual_waveform_by_bin_{oset}_{tag}.png"), onset_set=oset,
                which="resid", title=f"{C.ANIMAL} · {C.CHANNEL} · S2 − S1 residual per "
                f"pre-ictal bin ({int(bm)}-min linear) — {oset} seizures")

    for k, v in out.items():
        print(f"[paired_pulse.linbins] {k} -> {v}", flush=True)
    return {"mat": mat, "out": out}
