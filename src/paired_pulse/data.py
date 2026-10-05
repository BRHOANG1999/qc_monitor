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
from src.riding_event import primitives as _prim


def _log(m):
    print(f"[paired_pulse.data] {m}", flush=True)


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


def _window_array(lfp_f, t, onsets_ms, fs):
    """[n_ep x NWIN] where each row is that epoch's LFP windowed C.WIN ms from its own
    onset, plus the rebased time axis (same for all rows)."""
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
    return out, (twin if twin is not None else np.linspace(C.WIN[0], C.WIN[1], nwin))


def _phfo_present(win, twin, fs, band) -> np.ndarray:
    """Per-epoch pHFO on an already-onset-aligned window, via the riding_event
    primitives (robust-median template subtract -> gate the residual). 0/1."""
    if band is None or win.shape[0] == 0:
        return np.zeros(win.shape[0], np.int8)
    mask = np.ones(twin.size, bool)
    template, _k = _prim.robust_template(win, iters=1, k=4.0, win_mask=mask)
    resid = _prim.residuals(win, template)
    frac, prom, snr = _prim.phfo_metrics_epochs(resid, twin, fs, band,
                                                gate_ms=(float(twin[0]), float(twin[-1])))
    return _prim.phfo_gate(frac, prom, snr).astype(np.int8)


def extract_pairs(fp, channel=None, *, band=None) -> pd.DataFrame | None:
    """Per-epoch S1/S2 features + PPR for one file. None if unreadable / not paired.
    When *band* (the pHFO band) is given, also flags pHFO on each pulse."""
    channel = channel or C.CHANNEL
    r = _read_file(fp, channel)
    if r is None:
        return None
    lfp, stim, t, fs, st = r
    s1_on, s2_on, paired = _pulse_onsets(stim, t)
    if paired.sum() < 5:
        return None                                      # not a paired-pulse file
    lfp_f = _ef._bandpass(lfp, fs, C.BP_LOW_HZ, C.BP_HIGH_HZ) if C.BANDPASS else lfp
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
    if band is not None:                                 # pHFO per pulse (S2 is the ask)
        cols["s1_phfo_present"] = _phfo_present(s1w, twin, fs, band)
        cols["s2_phfo_present"] = _phfo_present(s2w, twin, fs, band)
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
        lfp_f = _ef._bandpass(lfp, fs, C.BP_LOW_HZ, C.BP_HIGH_HZ) if C.BANDPASS else lfp
        a, twin = _window_array(lfp_f, t, s1_on, fs)
        b, _ = _window_array(lfp_f, t, s2_on, fs)
        s1s.append(a[paired]); s2s.append(b[paired])
    assert s1s, "no paired-pulse waveforms found"
    return {"twin": twin, "s1": np.vstack(s1s), "s2": np.vstack(s2s)}


def build_pp_matrix(store, evoked_dir, *, since=None, force=False) -> pd.DataFrame:
    """Concatenate per-epoch paired-pulse features across files + join seizure
    proximity (time_to_onset_sec vs lead onsets). Cached to C.PP_CACHE."""
    if not force and os.path.exists(C.PP_CACHE):
        _log(f"loading cached paired-pulse matrix: {C.PP_CACHE}")
        return pd.read_pickle(C.PP_CACHE)
    files = list_pp_files(evoked_dir, since)
    band = None
    try:
        from src.riding_event.periictal import ensure_template
        band = ensure_template(store, C.ANIMAL)["band"]
        _log(f"pHFO band: {band[0]:.0f}-{band[1]:.0f} Hz")
    except Exception as e:                                # noqa: BLE001
        _log(f"pHFO band unavailable ({type(e).__name__}); skipping pHFO")
    _log(f"scanning {len(files)} files since {(since or C.PP_START_DATE):%Y-%m-%d} ...")
    frames = []
    for i, (fp, _d) in enumerate(files):
        if i % 20 == 0:
            _log(f"  file {i}/{len(files)}")
        df = extract_pairs(fp, band=band)
        if df is not None:
            frames.append(df)
    assert frames, "no paired-pulse epochs found in range"
    mat = pd.concat(frames, ignore_index=True)
    mat = _attach_seizures(store, mat)
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    mat.to_pickle(C.PP_CACHE)
    _log(f"{len(mat)} paired epochs from {len(frames)} files -> {C.PP_CACHE}")
    return mat


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
