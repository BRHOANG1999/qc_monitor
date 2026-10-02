"""Continuous raw-LFP envelope around each seizure, LP500-filtered, for the
animation's "full LFP" panel -- so you can see the actual LFP over the -120->0 min
pre-ictal window and the seizure discharge at onset, on the SAME 1-500 Hz band as
the evoked features / residual branch.

Reads the raw multi-GB continuous recording (riding_event.detect.load_channel),
anti-alias decimates to ~1 kHz (this IS the <=500 Hz low-pass, and keeps memory
bounded), high-passes at 1 Hz to match the 1-500 Hz band, blanks stim, then builds
a min-max display envelope. Expensive (raw read per seizure) -> cached per onset.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from . import config as C

# Band-specific cache (do not mix with any earlier broadband envelope cache).
_CACHE = C.CACHE_DIR + "/BCH111_lfp_envelopes_lp500.pkl"
_KEYS = ("env_t", "env_lo", "env_hi", "onset_epoch")
_TARGET_FS = 1000.0            # decimate to ~1 kHz => anti-alias low-pass at ~500 Hz


def _log(m):
    print(f"[preictal_biomarker.continuous_lfp] {m}", flush=True)


def _key(o) -> str:
    return str(int(round(float(o))))


def _factor_le(q: int, cap: int = 10) -> list:
    """Factor q into decimation steps each <= cap (IIR decimate is unstable for big
    single-step q). Leftover prime kept as-is."""
    out, r = [], int(q)
    for d in range(cap, 1, -1):
        while r % d == 0:
            out.append(d); r //= d
    if r > 1:
        out.append(r)
    return out or [1]


def _lp_decimate(seg: np.ndarray, fs: float, target_fs: float = _TARGET_FS):
    """Anti-alias decimate seg to ~target_fs (the <=500 Hz low-pass). (seg_d, fs_d)."""
    from scipy.signal import decimate
    q_total = max(1, int(round(fs / target_fs)))
    if q_total <= 1:
        return seg.astype(np.float32), float(fs)
    s, applied = seg, 1
    for q in _factor_le(q_total, 10):
        if q > 1:
            s = decimate(s, q, ftype="iir"); applied *= q
    return s.astype(np.float32), float(fs) / applied


def _gather_lp_envelope(store, onset, *, pre_h, post_h, ppm=6.0) -> dict:
    """LP500 continuous-LFP min-max envelope around one onset (minutes rel onset)."""
    from src.riding_event import periictal as PE, detect as DET, seizure_lfp as SL
    from src.utils import chunk_cache as CC, evoked_features as EF
    t_lo, t_hi = onset - pre_h * 3600.0, onset + post_h * 3600.0
    recs = [r for r in PE.animal_recordings(store, C.ANIMAL)
            if r["start_epoch"] < t_hi and r["start_epoch"] + r["duration"] > t_lo]
    assert len(recs) < PE._MAX_RECS, "too many recordings in the window"
    env_t, env_lo, env_hi = [], [], []
    for r in recs:
        loaded = DET.load_channel(r["file_path"], C.ANIMAL)
        if loaded is None:
            continue
        signal, fs, _ch = loaded
        signal = np.asarray(signal, dtype=np.float32)
        try:
            ts, n = r["start_epoch"], signal.size
            i0 = max(0, int((t_lo - ts) * fs))
            i1 = min(n, int((t_hi - ts) * fs))
            if i1 <= i0:
                continue
            seg = signal[i0:i1].copy()
            seg, fs_d = _lp_decimate(seg, fs)                 # <=500 Hz low-pass
            hi_hz = min(C.BP_HIGH_HZ, 0.49 * fs_d)
            seg = EF._bandpass(seg[None, :], fs_d, C.BP_LOW_HZ, hi_hz)[0]  # 1-500 Hz
            stim = PE._recording_stim_times(store, C.ANIMAL, r["file_id"])
            if stim is not None:
                SL._blank_stim(seg, fs_d, np.asarray(stim, float) - i0 / fs,
                               pre_ms=1.0, post_ms=12.0)
            t0m = (ts + i0 / fs - onset) / 60.0
            tc, lo, hi = SL._decimate_minmax(seg, fs_d, t0m, px_per_min=ppm)
            env_t.append(tc); env_lo.append(lo); env_hi.append(hi)
        finally:
            del signal
            CC.clear()
    if not env_t:
        return None
    et = np.concatenate(env_t); o = np.argsort(et)
    return {"env_t": et[o], "env_lo": np.concatenate(env_lo)[o],
            "env_hi": np.concatenate(env_hi)[o], "onset_epoch": float(onset)}


def build_lfp_envelopes(store, onsets, *, pre_h=2.0, post_h=0.35,
                        force=False) -> dict:
    """{onset-key: LP500 envelope dict} per onset; raw read only for uncached ones."""
    cached = {}
    if not force and os.path.exists(_CACHE):
        with open(_CACHE, "rb") as f:
            cached = pickle.load(f)
    todo = [o for o in onsets if _key(o) not in cached]
    _log(f"{len(onsets)} onsets; {len(todo)} need an LP500 raw-LFP read "
         f"(pre {pre_h} h, post {post_h} h) ...")
    for i, o in enumerate(todo):
        _log(f"  [{i+1}/{len(todo)}] reading + LP500 raw LFP around onset {_key(o)} ...")
        try:
            cached[_key(o)] = _gather_lp_envelope(store, float(o), pre_h=pre_h,
                                                  post_h=post_h)
        except Exception as e:                            # noqa: BLE001
            _log(f"    skip {_key(o)}: {type(e).__name__}: {e}")
            cached[_key(o)] = None
        os.makedirs(C.CACHE_DIR, exist_ok=True)
        with open(_CACHE, "wb") as f:                     # incremental / resumable
            pickle.dump(cached, f)
    return {_key(o): cached.get(_key(o)) for o in onsets}
