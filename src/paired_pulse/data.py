"""Paired-pulse extraction: per epoch, detect the two pulse onsets from the stim-copy
trace, window each pulse's LFP response 1-49 ms at its own onset, compute features,
and form PPR = feature(S2)/feature(S1). Build an over-time matrix + seizure join.

Decisive data facts (verified on real BCH111 recordings): each 50 ms pair is ONE
evokedOutput epoch; the dedup anchor is at t=0 and the partner at +50 ms (anchor=S1)
OR -50 ms (anchor=S2) -- the side varies per epoch, so it is detected per epoch.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from . import config as C
from src.notifications.evoked_stim_corr import _list_files, _file_dt, _match_key
from src.utils import evoked_features as _ef


def _log(m):
    print(f"[paired_pulse.data] {m}", flush=True)


def _filter(lfp, fs):
    """Apply the configured filter to the full trace. Default FILTER_MODE='lowpass' is a
    500 Hz low-pass ONLY (no high-pass corner -> no artifact ring / baseline droop)."""
    if not C.BANDPASS:
        return lfp
    if getattr(C, "FILTER_MODE", "bandpass") == "lowpass":
        return _ef._lowpass(lfp, fs, C.BP_HIGH_HZ)
    return _ef._bandpass(lfp, fs, C.BP_LOW_HZ, C.BP_HIGH_HZ)


def list_pp_files(evoked_dir, since=None):
    """Evoked files for the animal on/after *since* (default C.PP_START_DATE)."""
    since = since or C.PP_START_DATE
    out = [(f, _file_dt(f)) for f in _list_files(C.ANIMAL, evoked_dir, None)]
    out = [(f, d) for f, d in out if d is not None and d >= since]
    out.sort(key=lambda x: x[1])
    return out


def _read_file(fp, channel):
    """(lfp[ep,S], stim[ep,S], time_ms[S], fs, stim_times[ep]) or None."""
    import h5py
    try:
        with h5py.File(fp, "r") as g:
            grp = g.get("allAnimalResults")
            if grp is None:
                return None
            node = grp.get(channel) or grp.get(_match_key(list(grp.keys()), channel))
            if node is None or "evokedData" not in node or "stimulusTraces" not in node:
                return None
            t = np.asarray(node["timeAxis"][()], float).ravel()
            lfp = np.asarray(node["evokedData"][()], float)
            stim = np.asarray(node["stimulusTraces"][()], float)
            st = (np.asarray(node["stimulusTimes"][()], float).ravel()
                  if "stimulusTimes" in node else None)
    except (OSError, KeyError, ValueError, TypeError):
        return None
    if lfp.ndim != 2 or lfp.shape != stim.shape or lfp.shape[1] != t.size or t.size < 8:
        return None
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    return lfp, stim, t, 1000.0 / dt, st


def _amp_near(stim, t, center_ms, half=1.5):
    """Max |stim| in a small window around center_ms (per epoch)."""
    a = int(np.argmin(np.abs(t - (center_ms - half))))
    b = int(np.argmin(np.abs(t - (center_ms + half)))) + 1
    return np.max(np.abs(stim[:, a:b]), axis=1) if b > a else np.zeros(stim.shape[0])


def _pulse_onsets(stim, t):
    """Per epoch: (s1_onset_ms, s2_onset_ms, is_paired). Anchor at 0; partner at +ISI
    (anchor=S1) or -ISI (anchor=S2). Chronological S1=earlier, S2=later."""
    anchor = _amp_near(stim, t, 0.0)
    plus = _amp_near(stim, t, C.ISI_MS)
    minus = _amp_near(stim, t, -C.ISI_MS)
    partner = np.maximum(plus, minus)
    paired = (anchor > 0) & (partner >= C.SECOND_PULSE_FRAC * anchor)
    partner_plus = plus >= minus
    s1 = np.where(partner_plus, 0.0, -C.ISI_MS)          # earlier pulse
    s2 = np.where(partner_plus, C.ISI_MS, 0.0)           # later pulse
    return s1, s2, paired


def _window_array(lfp_f, t, onsets_ms, fs, demean=None):
    """[n_ep x NWIN] where each row is that epoch's LFP windowed C.WIN ms from its own
    onset, plus the rebased time axis (same for all rows). When demean (default
    C.BASELINE_SUBTRACT) each row has its own mean subtracted -- the low-pass keeps DC,
    so this baseline-removal restores DC-insensitive amplitude features."""
    demean = C.BASELINE_SUBTRACT if demean is None else demean
    nwin = int(round((C.WIN[1] - C.WIN[0]) / 1000.0 * fs))
    out = np.full((lfp_f.shape[0], nwin), np.nan)
    twin = None
    for on in np.unique(onsets_ms):
        m = onsets_ms == on
        a = int(np.argmin(np.abs(t - (on + C.WIN[0]))))
        seg = lfp_f[m, a:a + nwin]
        if seg.shape[1] == nwin:
            out[m] = seg
            if twin is None:
                twin = t[a:a + nwin] - on
    if demean:
        out = out - np.nanmean(out, axis=1, keepdims=True)
    return out, (twin if twin is not None else np.linspace(C.WIN[0], C.WIN[1], nwin))


def extract_pairs(fp, channel=None) -> pd.DataFrame | None:
    """Per-epoch S1/S2 features + PPR for one file. None if unreadable / not paired."""
    channel = channel or C.CHANNEL
    r = _read_file(fp, channel)
    if r is None:
        return None
    lfp, stim, t, fs, st = r
    s1_on, s2_on, paired = _pulse_onsets(stim, t)
    if paired.sum() < 5:
        return None                                      # not a paired-pulse file
    lfp_f = _filter(lfp, fs)
    s1w, twin = _window_array(lfp_f, t, s1_on, fs)
    s2w, _ = _window_array(lfp_f, t, s2_on, fs)
    f1 = _ef.compute_all(s1w, twin, fs, include_wavelet=False)
    f2 = _ef.compute_all(s2w, twin, fs, include_wavelet=False)
    cols = {}
    for feat in C.FEATURES:
        if feat in f1 and feat in f2:
            a, b = np.asarray(f1[feat], float), np.asarray(f2[feat], float)
            cols[f"s1_{feat}"] = a
            cols[f"s2_{feat}"] = b
            with np.errstate(divide="ignore", invalid="ignore"):
                cols[f"ppr_{feat}"] = b / np.where(a != 0, a, np.nan)
    fdt = _file_dt(fp)
    base = fdt.timestamp() if fdt is not None else np.nan
    t_epoch = base + st if (st is not None and st.size == lfp.shape[0]) else np.full(
        lfp.shape[0], base)
    df = pd.DataFrame(cols)
    df["t_epoch"] = t_epoch
    df["anchor_is_s1"] = (s1_on == 0.0)
    df["paired"] = paired
    df["file"] = os.path.basename(fp)
    return df[df["paired"]].reset_index(drop=True)


def mean_waveforms(evoked_dir, *, since=None, max_files=6, channel=None) -> dict:
    """Pooled S1 and S2 windowed LFP traces (1-49 ms from each pulse's own onset)
    across the most recent *max_files* paired-pulse files, for the overlay/residual
    figure. Returns {twin, s1[n,W], s2[n,W]}."""
    channel = channel or C.CHANNEL
    files = list_pp_files(evoked_dir, since)[-max_files:]
    s1s, s2s, twin = [], [], None
    for fp, _d in files:
        r = _read_file(fp, channel)
        if r is None:
            continue
        lfp, stim, t, fs, _st = r
        s1_on, s2_on, paired = _pulse_onsets(stim, t)
        if paired.sum() < 5:
            continue
        lfp_f = _filter(lfp, fs)
        a, twin = _window_array(lfp_f, t, s1_on, fs)
        b, _ = _window_array(lfp_f, t, s2_on, fs)
        s1s.append(a[paired]); s2s.append(b[paired])
    assert s1s, "no paired-pulse waveforms found"
    return {"twin": twin, "s1": np.vstack(s1s), "s2": np.vstack(s2s)}


def example_pairs(evoked_dir, *, since=None, n_files=3, n_per=4, channel=None,
                  seed=0) -> dict:
    """A handful of INDIVIDUAL paired-pulse epochs (S1 + S2 windowed traces) sampled
    from files spread across the window, for an example-evoked-responses figure."""
    channel = channel or C.CHANNEL
    files = list_pp_files(evoked_dir, since)
    if len(files) > n_files:
        picks = [files[int(k)] for k in np.linspace(0, len(files) - 1, n_files)]
    else:
        picks = files
    rng = np.random.default_rng(seed)
    out, twin = [], None
    for fp, _d in picks:
        r = _read_file(fp, channel)
        if r is None:
            continue
        lfp, stim, t, fs, _st = r
        s1_on, s2_on, paired = _pulse_onsets(stim, t)
        if paired.sum() < 5:
            continue
        lfp_f = _filter(lfp, fs)
        s1w, twin = _window_array(lfp_f, t, s1_on, fs)
        s2w, _ = _window_array(lfp_f, t, s2_on, fs)
        idx = np.flatnonzero(paired)
        for k in rng.choice(idx, size=min(n_per, idx.size), replace=False):
            p1 = float(np.ptp(s1w[k])); p2 = float(np.ptp(s2w[k]))
            out.append({"s1": s1w[k], "s2": s2w[k],
                        "ppr": p2 / p1 if p1 else np.nan,
                        "label": _file_dt(fp).strftime("%m-%d %H:%M")})
    return {"twin": twin, "pairs": out}


def waveform_by_leadtime(store, evoked_dir, *, since=None, cap_h=12.0,
                         channel=None) -> dict:
    """Averaged S1 & S2 evoked waveforms (and the S2−S1 residual) per LOG lead-time
    bin, pooled across epochs, for both lead and all seizures. One file-read pass;
    accumulates per-bin sums (memory-light). Bins match the pre-ictal trend edges."""
    from src.periictal import trajectory as _TR
    from src.preictal import isi as _isi
    channel = channel or C.CHANNEL
    edges = _TR.default_edges(cap_h * 3600, 2.0)
    nb = edges.size - 1
    cen = np.sqrt(edges[:-1] * edges[1:]) / 60.0          # geo centers (min)
    sz = _isi.scored_seizures(store, C.ANIMAL)
    ons = {"all": np.sort(np.array([s.onset_epoch for s in sz], float)),
           "lead": np.sort(np.array([s.onset_epoch for s in
                                     _isi.leading_seizures(sz, 6 * 3600.0)], float))}
    acc = {k: {"s1": None, "s2": None, "cnt": np.zeros(nb)} for k in ons}
    twin = None
    for fp, _d in list_pp_files(evoked_dir, since):
        r = _read_file(fp, channel)
        if r is None:
            continue
        lfp, stim, t, fs, st = r
        s1_on, s2_on, paired = _pulse_onsets(stim, t)
        if paired.sum() < 5:
            continue
        lfp_f = _filter(lfp, fs)
        s1w, twin = _window_array(lfp_f, t, s1_on, fs)
        s2w, _ = _window_array(lfp_f, t, s2_on, fs)
        base = _file_dt(fp).timestamp()
        te = base + st if (st is not None and st.size == lfp.shape[0]) else \
            np.full(lfp.shape[0], base)
        for k, o in ons.items():
            if not o.size:
                continue
            idx = np.searchsorted(o, te, side="left")
            nxt = np.where(idx < o.size, o[np.clip(idx, 0, o.size - 1)], np.inf)
            tto = nxt - te
            pidx = np.searchsorted(o, te, side="right") - 1
            prev = np.where(pidx >= 0, o[np.clip(pidx, 0, o.size - 1)], -np.inf)
            m = (paired & (tto > 0) & (tto <= edges[-1])    # pre-ictal only; exclude
                 & (te - prev > C.POSTICTAL_BUFFER_SEC))     # on/post prior onset
            if not m.any():
                continue
            bi = np.clip(np.searchsorted(edges, tto[m], "right") - 1, 0, nb - 1)
            if acc[k]["s1"] is None:
                acc[k]["s1"] = np.zeros((nb, s1w.shape[1]))
                acc[k]["s2"] = np.zeros((nb, s2w.shape[1]))
            np.add.at(acc[k]["s1"], bi, s1w[m])
            np.add.at(acc[k]["s2"], bi, s2w[m])
            np.add.at(acc[k]["cnt"], bi, 1)
    out = {"twin": twin, "centers": cen, "edges": edges}
    for k in ons:
        c = acc[k]["cnt"]
        d = np.where(c[:, None] > 0, c[:, None], np.nan)
        s1m = acc[k]["s1"] / d if acc[k]["s1"] is not None else None
        s2m = acc[k]["s2"] / d if acc[k]["s2"] is not None else None
        out[k] = {"s1": s1m, "s2": s2m,
                  "resid": (s2m - s1m) if s2m is not None else None, "cnt": c}
    return out


def preictal_traces(store, evoked_dir, *, since=None, cap_h=2.0, channel=None) -> dict:
    """EVERY individual STRICT pre-ictal windowed S2 and residual (S2−S1) trace + its
    time-to-onset (minutes), for lead and all seizures, in one file pass. Full density
    (not downsampled) for the 'each evoked response following the trend' overlay."""
    from src.preictal import isi as _isi
    from .linear_bins import _preictal
    channel = channel or C.CHANNEL
    sz = _isi.scored_seizures(store, C.ANIMAL)
    ons = {"all": np.sort(np.array([s.onset_epoch for s in sz], float)),
           "lead": np.sort(np.array([s.onset_epoch for s in
                                     _isi.leading_seizures(sz, 6 * 3600.0)], float))}
    cap = cap_h * 3600.0
    buf = {k: {"s2": [], "resid": [], "tto": []} for k in ons}
    twin = None
    for fp, _d in list_pp_files(evoked_dir, since):
        r = _read_file(fp, channel)
        if r is None:
            continue
        lfp, stim, t, fs, st = r
        s1_on, s2_on, paired = _pulse_onsets(stim, t)
        if paired.sum() < 5:
            continue
        lfp_f = _filter(lfp, fs)
        s1w, twin = _window_array(lfp_f, t, s1_on, fs)
        s2w, _ = _window_array(lfp_f, t, s2_on, fs)
        base = _file_dt(fp).timestamp()
        te = base + st if (st is not None and st.size == lfp.shape[0]) else \
            np.full(lfp.shape[0], base)
        for k, o in ons.items():
            if not o.size:
                continue
            m, tto = _preictal(te, o, cap)
            m = m & paired
            if not m.any():
                continue
            buf[k]["s2"].append(s2w[m])
            buf[k]["resid"].append(s2w[m] - s1w[m])
            buf[k]["tto"].append(tto[m] / 60.0)             # minutes
    out = {"twin": twin}
    for k in ons:
        if buf[k]["s2"]:
            out[k] = {"s2": np.vstack(buf[k]["s2"]), "resid": np.vstack(buf[k]["resid"]),
                      "tto": np.concatenate(buf[k]["tto"])}
        else:
            out[k] = {"s2": None, "resid": None, "tto": None}
    return out


def build_pp_matrix(store, evoked_dir, *, since=None, force=False) -> pd.DataFrame:
    """Concatenate per-epoch paired-pulse features across files + join seizure
    proximity (time_to_onset_sec vs lead onsets). Cached to C.PP_CACHE."""
    if not force and os.path.exists(C.PP_CACHE):
        _log(f"loading cached paired-pulse matrix: {C.PP_CACHE}")
        return pd.read_pickle(C.PP_CACHE)
    files = list_pp_files(evoked_dir, since)
    _log(f"scanning {len(files)} files since {(since or C.PP_START_DATE):%Y-%m-%d} ...")
    frames = []
    for i, (fp, _d) in enumerate(files):
        if i % 20 == 0:
            _log(f"  file {i}/{len(files)}")
        df = extract_pairs(fp)
        if df is not None:
            frames.append(df)
    assert frames, "no paired-pulse epochs found in range"
    mat = pd.concat(frames, ignore_index=True)
    mat = _attach_seizures(store, mat)
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    mat.to_pickle(C.PP_CACHE)
    _log(f"{len(mat)} paired epochs from {len(frames)} files -> {C.PP_CACHE}")
    return mat


def _band_p(null, obs):
    """Per-bin 2.5/97.5 band, median, and two-sided shift-null p (fraction of
    surrogates at least as far from the null median as the observed)."""
    lo = np.nanpercentile(null, 2.5, axis=0)
    hi = np.nanpercentile(null, 97.5, axis=0)
    med = np.nanmedian(null, axis=0)
    p = np.full(obs.size, np.nan)
    for b in range(obs.size):
        col = null[:, b][np.isfinite(null[:, b])]
        if col.size and np.isfinite(obs[b]):
            p[b] = (np.sum(np.abs(col - med[b]) >= abs(obs[b] - med[b])) + 1) / (col.size + 1)
    return lo, hi, med, p


def trend_shift_null(mat, onsets, *, feature="ppr_peak_to_trough", cap_h=12.0,
                     n_surr=500, min_shift_h=3.0, seed=0) -> dict:
    """Circular-shift null for the pre-ictal TREND: bin-mean of *feature* vs LOG
    time-to-next-onset, then shift the onsets (wrapped, >= min_shift_h) and recompute,
    n_surr times. Tests whether the near-onset change beats random alignment to the
    same (drifting) metric series."""
    from src.periictal import trajectory as _TR
    t = mat["t_epoch"].to_numpy(float); v = mat[feature].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(v); t, v = t[ok], v[ok]
    ons = np.sort(np.asarray(onsets, float))
    edges = _TR.default_edges(cap_h * 3600, 2.0)
    nb = edges.size - 1
    cen = np.sqrt(edges[:-1] * edges[1:]) / 60.0          # geo centers (min)

    def _stat(o):
        so = np.sort(o)
        idx = np.searchsorted(so, t, side="left")
        nxt = np.where(idx < so.size, so[np.clip(idx, 0, so.size - 1)], np.inf)
        tto = nxt - t
        pidx = np.searchsorted(so, t, side="right") - 1
        prev = np.where(pidx >= 0, so[np.clip(pidx, 0, so.size - 1)], -np.inf)
        m = (tto > 0) & (tto <= edges[-1]) & (t - prev > C.POSTICTAL_BUFFER_SEC)
        bi = np.clip(np.searchsorted(edges, tto[m], "right") - 1, 0, nb - 1)
        sv = np.bincount(bi, weights=v[m], minlength=nb)
        cv = np.bincount(bi, minlength=nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            return sv / np.where(cv > 0, cv, np.nan)

    obs = _stat(ons)
    lo_t, hi_t = t.min(), t.max(); span = hi_t - lo_t; ms = min_shift_h * 3600
    rng = np.random.default_rng(seed)
    null = np.full((int(n_surr), nb), np.nan)
    for i in range(int(n_surr)):
        null[i] = _stat(lo_t + ((ons - lo_t + rng.uniform(ms, span - ms)) % span))
    lo, hi, med, p = _band_p(null, obs)
    return {"centers": cen, "obs": obs, "lo": lo, "hi": hi, "med": med, "p": p,
            "n_surr": int(n_surr), "feature": feature}


def trajectory_shift_null(mat, onsets, *, feature="ppr_peak_to_trough", pre_h=2.0,
                          post_h=1.0, bin_min=10.0, n_surr=500, min_shift_h=3.0,
                          seed=0) -> dict:
    """Circular-shift null for the peri-ictal trajectory (-pre_h..+post_h around onset,
    per-onset bin-mean then mean across onsets). Shift onsets (wrapped) n_surr times."""
    t = mat["t_epoch"].to_numpy(float); v = mat[feature].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(v); t, v = t[ok], v[ok]
    ons = np.sort(np.asarray(onsets, float))
    edges = np.arange(-pre_h * 60, post_h * 60 + bin_min, bin_min)
    cen = 0.5 * (edges[:-1] + edges[1:]); nb = cen.size

    def _traj(o):
        acc = np.zeros(nb); cnt = np.zeros(nb)
        for on in o:
            rel = (t - on) / 60.0
            m = (rel >= edges[0]) & (rel < edges[-1])
            if m.sum() < 3:
                continue
            bi = np.clip(np.searchsorted(edges, rel[m], "right") - 1, 0, nb - 1)
            sv = np.bincount(bi, weights=v[m], minlength=nb)
            cv = np.bincount(bi, minlength=nb)
            with np.errstate(divide="ignore", invalid="ignore"):
                line = sv / np.where(cv > 0, cv, np.nan)
            acc = np.where(np.isfinite(line), acc + np.nan_to_num(line), acc)
            cnt = cnt + np.isfinite(line)
        return acc / np.where(cnt > 0, cnt, np.nan)

    obs = _traj(ons)
    lo_t, hi_t = t.min(), t.max(); span = hi_t - lo_t; ms = min_shift_h * 3600
    rng = np.random.default_rng(seed)
    null = np.full((int(n_surr), nb), np.nan)
    for i in range(int(n_surr)):
        null[i] = _traj(lo_t + ((ons - lo_t + rng.uniform(ms, span - ms)) % span))
    lo, hi, med, p = _band_p(null, obs)
    return {"centers": cen, "obs": obs, "lo": lo, "hi": hi, "med": med, "p": p,
            "n_surr": int(n_surr), "bin_min": bin_min, "n_onsets": int(ons.size),
            "feature": feature}


def _attach_seizures(store, mat) -> pd.DataFrame:
    """Join seizure proximity. time_to_onset_sec = time to the next LEAD onset (clean
    pre-ictal trend); tto_any_sec = time to the next ANY scored event (for the
    P(event within H) figure). Uses isi directly -- NOT the Sep13-Oct2 scoped
    preictal_biomarker.lead_onsets, which would exclude the paired-pulse window."""
    from src.preictal import isi as _isi
    sz = _isi.scored_seizures(store, C.ANIMAL)
    allon = np.sort(np.array([s.onset_epoch for s in sz], float))
    leadon = np.sort(np.array([s.onset_epoch for s in
                               _isi.leading_seizures(sz, 6 * 3600.0)], float))
    t = mat["t_epoch"].to_numpy(float)

    def _tto(ons):
        if not ons.size:
            return np.full(t.size, np.inf)
        idx = np.searchsorted(ons, t, side="left")
        return np.where(idx < ons.size, ons[np.clip(idx, 0, ons.size - 1)],
                        np.inf) - t

    mat["time_to_onset_sec"] = _tto(leadon)              # next LEAD
    mat["tto_any_sec"] = _tto(allon)                     # next ANY event
    lo, hi = t.min(), t.max()
    mat["n_lead_in_window"] = int(((leadon >= lo) & (leadon <= hi)).sum())
    mat["n_event_in_window"] = int(((allon >= lo) & (allon <= hi)).sum())
    return mat
