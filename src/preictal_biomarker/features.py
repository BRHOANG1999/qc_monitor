"""Per-epoch evoked feature matrix for the pre-ictal biomarker rebuild.

Builds, for BCH111 channel BCH111SR over the analysis window, one row per lead-up
stimulus with: a per-epoch timestamp, the seizure-timing join (time_to_onset,
seizure index/onset, pre/post phase), and the 8 analysis features --
``line_length, rms_amplitude, variance, autocorrelation, peak_to_trough,
max_slope`` (per-waveform, 2-50 ms / LP500, gain-corrected, then Z_ss-detrended)
plus ``csd_variance, csd_ar1`` (rolling critical-slowing stats of the primary
feature). Reuses the committed ``periictal`` matrix machinery for the feature
extraction + seizure join; adds the Z_ss detrend and CSD features, which did not
survive in ``src/``.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import sqlite3

import numpy as np
import pandas as pd

from src.periictal import matrix as _mx, passive as _passive, config as _pcfg
from src.preictal import isi as _isi
from . import config as C

_DT_RE = re.compile(r"__(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})")


def _log(msg: str) -> None:
    print(f"[preictal_biomarker.features] {msg}", flush=True)


def lp_cfg():
    """FeatureConfig for the 2-50 ms / 1-500 Hz-bandpass evoked window."""
    return _passive.window_config(C.WINDOW_MS[0], C.WINDOW_MS[1],
                                  bandpass=C.BANDPASS, bp_low_hz=C.BP_LOW_HZ,
                                  bp_high_hz=C.BP_HIGH_HZ)


def scoped_onsets(store) -> np.ndarray:
    """All scored seizure onsets within the analysis date scope (epoch sec)."""
    s0 = C.ANALYSIS_START.timestamp()
    s1 = C.ANALYSIS_END.timestamp()
    on = np.array([s.onset_epoch for s in _isi.scored_seizures(store, C.ANIMAL)],
                  float)
    return np.sort(on[(on >= s0) & (on < s1)])


def lead_onsets(store) -> np.ndarray:
    """Lead (cluster-leader) seizure onsets within scope."""
    s0, s1 = C.ANALYSIS_START.timestamp(), C.ANALYSIS_END.timestamp()
    lead = _isi.leading_seizures(_isi.scored_seizures(store, C.ANIMAL),
                                 C.LEAD_GAP_H * 3600.0)
    on = np.array([s.onset_epoch for s in lead], float)
    return np.sort(on[(on >= s0) & (on < s1)])


def impedance_by_datetime(db_path: str) -> dict:
    """{recording datetime (sec, from filename stamp): (Ra_kOhm, Z_ss_kOhm)}."""
    assert db_path, "db_path required"
    conn = sqlite3.connect(db_path)
    out: dict = {}
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT pf.file_path fp, ci.access_r_kohm ra, ci.slow_ss_kohm z
               FROM channel_impedance ci JOIN processed_files pf ON pf.id=ci.file_id
               WHERE ci.channel_name LIKE ?
                 AND (ci.access_r_kohm IS NOT NULL OR ci.slow_ss_kohm IS NOT NULL)""",
            (f"%{C.ANIMAL}%",)).fetchall()
    finally:
        conn.close()
    for r in rows:
        m = _DT_RE.search(os.path.basename(str(r["fp"])))
        if m is None:
            continue
        key = _dt.datetime(*[int(x) for x in m.groups()]).timestamp()
        out[key] = (float(r["ra"]) if r["ra"] is not None else np.nan,
                    float(r["z"]) if r["z"] is not None else np.nan)
    return out


def _warm(evoked_dir: str, onsets: np.ndarray) -> None:
    """Warm the LP sidecars for files that can contribute a peri-ictal row."""
    filt = _mx._near_seizure_file_filter(onsets, C.LOOKBACK_SEC,
                                         _pcfg.PERIICTAL_PREFILTER_SLACK_SEC)
    _log("warming LP500 2-50 ms sidecars (~15 s/file, one-time) ...")

    def prog(i, n, fp):
        if i == 1 or i % 10 == 0 or i == n:
            _log(f"  sidecar {i}/{n}: {os.path.basename(str(fp))[:52]}")

    built, total = _passive.warm_variant(C.ANIMAL, evoked_dir, "evokedw",
                                         lp_cfg(), progress=prog, file_filter=filt)
    _log(f"sidecars ready: built {built} of {total} candidate files")


def build_raw_matrix(store, evoked_dir: str) -> pd.DataFrame:
    """Channel-filtered, in-scope peri-ictal matrix with the 6 waveform features
    (gain-corrected by the matrix). Seizure-join metadata included."""
    df = _mx.build_matrix(store, C.ANIMAL, evoked_dir, variant="evoked",
                          sidecar_variant="evokedw", feature_cfg=lp_cfg(),
                          window_sec=C.LOOKBACK_SEC, warm_missing=False)
    assert len(df), "empty matrix — were the LP sidecars warmed?"
    df = df[df["channel"] == C.CHANNEL].copy()
    s0, s1 = C.ANALYSIS_START.timestamp(), C.ANALYSIS_END.timestamp()
    df = df[df["seizure_onset_epoch"].between(s0, s1)]
    df = df[pd.to_numeric(df["t_epoch"], errors="coerce").between(s0, s1)]
    return df.reset_index(drop=True)


def _attach_zss(df: pd.DataFrame, db_path: str, *, tol_sec: float = 7200.0
                ) -> pd.DataFrame:
    """Join each row's recording Z_ss / Ra by matching its epoch time to the
    nearest impedance measurement. Impedance is measured per (hourly) recording
    and drifts slowly, so the nearest-in-time value is that recording's Z_ss. We
    match on ``abs_dt`` (epoch absolute time); ``rec`` is an internal recording
    start that can differ from the filename-stamped impedance time by tens of
    minutes, which is why a tight match on ``rec`` finds nothing."""
    imp = impedance_by_datetime(db_path)
    keys = np.array(sorted(imp), dtype=float)
    assert keys.size, "no impedance rows found for Z_ss join"
    t = pd.to_datetime(df["abs_dt"]).map(lambda d: d.timestamp()).to_numpy(float)
    ra = np.full(len(df), np.nan)
    zss = np.full(len(df), np.nan)
    pos = np.clip(np.searchsorted(keys, t), 1, keys.size - 1)
    for i, ti in enumerate(t):                   # nearest stamp (slowly-varying)
        j = pos[i]
        j = j if abs(keys[j] - ti) < abs(keys[j - 1] - ti) else j - 1
        if abs(keys[j] - ti) <= tol_sec:
            ra[i], zss[i] = imp[keys[j]]
    df = df.copy()
    df["ra"] = ra
    df["zss"] = zss
    return df


def _detrend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Regress Z_ss out of each waveform feature (residual + original mean), so
    impedance drift is removed but the feature keeps its scale. Rows without a
    Z_ss value keep their raw feature."""
    df = df.copy()
    z = df["zss"].to_numpy(float)
    ok = np.isfinite(z)
    for f in C.DETREND_FEATURES:
        if f not in df.columns:
            continue
        y = df[f].to_numpy(float)
        m = ok & np.isfinite(y)
        if m.sum() < 10:
            continue
        a, b = np.polyfit(z[m], y[m], 1)
        resid = y.copy()
        resid[m] = y[m] - (a * z[m] + b) + float(np.mean(y[m]))
        df[f] = resid
    return df


def _ar1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a (detrended) window; NaN if degenerate."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan
    x = x - x.mean()
    d = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / d) if d > 0 else np.nan


def _add_csd(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling critical-slowing stats of the primary feature over CSD_WIN trials
    (ordered by time): csd_variance (rolling var) and csd_ar1 (rolling lag-1
    autocorr of the locally-detrended window)."""
    df = df.sort_values("t_epoch").reset_index(drop=True)
    p = df[C.CSD_PRIMARY].to_numpy(float)
    w = int(C.CSD_WIN)
    s = pd.Series(p)
    mp = w // 2
    var = s.rolling(w, min_periods=mp).var(ddof=1)
    std = s.rolling(w, min_periods=mp).std(ddof=1)
    mean = s.rolling(w, min_periods=mp).mean()
    skew = s.rolling(w, min_periods=mp).skew()
    dvar = pd.Series(np.r_[np.nan, np.diff(p)]).rolling(w, min_periods=mp).var(ddof=1)
    df["csd_variance"] = var.to_numpy()                       # rising variance
    df["csd_skew"] = skew.to_numpy()                          # rising skewness
    df["csd_cv"] = (std / mean.abs().replace(0, np.nan)).to_numpy()   # coeff. of var
    df["csd_redden"] = (var / dvar.replace(0, np.nan)).to_numpy()     # low/high power
    ar1 = np.full(len(p), np.nan)                             # rising lag-1 autocorr
    for i in range(len(p)):
        lo = max(0, i - w + 1)
        if i - lo + 1 >= mp:
            ar1[i] = _ar1(p[lo:i + 1])
    df["csd_ar1"] = ar1
    return df


def build_feature_matrix(store, evoked_dir: str, db_path: str, *,
                         force: bool = False) -> pd.DataFrame:
    """Full per-epoch [t_epoch, meta, 8 features] matrix for BCH111SR. Cached to
    parquet; pass force=True to rebuild. Warms LP sidecars on a cold build."""
    if not force and os.path.exists(C.FEATURE_CACHE):
        _log(f"loading cached feature matrix: {C.FEATURE_CACHE}")
        return pd.read_pickle(C.FEATURE_CACHE)
    onsets = scoped_onsets(store)
    assert onsets.size >= 2, "need >= 2 in-scope seizures"
    _log(f"{onsets.size} in-scope seizures "
         f"({C.ANALYSIS_START:%Y-%m-%d}..{C.ANALYSIS_END:%Y-%m-%d})")
    _warm(evoked_dir, onsets)
    _log("building peri-ictal matrix (reading LP sidecars) ...")
    df = build_raw_matrix(store, evoked_dir)
    _log(f"{len(df)} BCH111SR rows; attaching Z_ss and detrending ...")
    df = _attach_zss(df, db_path)
    n_z = int(np.isfinite(df["zss"]).sum())
    _log(f"Z_ss matched for {n_z}/{len(df)} rows; CSD on '{C.CSD_PRIMARY}' ...")
    df = _detrend_features(df)
    df = _add_csd(df)
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    df.to_pickle(C.FEATURE_CACHE)
    _log(f"cached -> {C.FEATURE_CACHE}  ({len(df)} rows)")
    return df
