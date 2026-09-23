"""Prong A: isolate the riding event by subtracting the average evoked response.

For one recording + channel: read the evoked epochs, build a per-recording
robust-median template (the *average evoked response*), subtract it to get
per-epoch residuals, flag the epochs that carry the riding event by residual
energy, and compute the power spectra that separate event windows from clean
responses. Pure of plotting — ``render`` draws the returned arrays; heavy only in
that it reads one recording's traces once (never on a dashboard thread).
"""

from __future__ import annotations

import logging

import numpy as np

from src.riding_event import primitives as _p
from src.riding_event import spectra as _s
from src.utils.animal import is_animal_channel, split_animal_electrode
from src.utils.evoked_output import read_file_evoked

logger = logging.getLogger("qc_monitor.riding_event.residual")

# Post-stim windows (ms). Flagging uses the tight window where the event sits;
# the spectrum uses a wider window for finer frequency resolution. Both start
# past the artifact (>~2 ms) so the stim transient never dominates.
DEFAULT_FLAG_WIN_MS = (2.0, 180.0)         # wide: the ripple onsets 5-50 ms out
DEFAULT_SPEC_WIN_MS = (2.0, 200.0)
_DEFAULT_HF_BAND = (150.0, 800.0)          # fallback ripple band if no fundamental


def _hf_band(f0: float, fs: float) -> tuple:
    """The ripple band to flag on: centred on the discovered fundamental *f0*
    (the event is a fast oscillation ~f0), else a sensible HF default. Always
    high-pass enough (>=100 Hz) to reject the smooth low-frequency evoked
    response, so only the HF ripple is scored."""
    nyq = 0.49 * float(fs)
    if f0 and np.isfinite(f0) and f0 > 0:
        return (max(100.0, f0 * 0.6), min(f0 * 2.5, nyq))
    return (_DEFAULT_HF_BAND[0], min(_DEFAULT_HF_BAND[1], nyq))


def pick_channel(chans: dict, animal: str, prefer: str | None = None) -> str | None:
    """Choose the animal's electrode channel from a ``read_file_evoked`` dict.

    *prefer* (an electrode or full channel name) wins when it matches; otherwise
    the animal channel with the most epochs is used. Returns the channel key or
    None when the file carries no usable animal channel."""
    cand = []
    for ch, rec in chans.items():
        a, elec = split_animal_electrode(ch)
        if a != animal or not is_animal_channel(ch):
            continue
        tr = rec.get("traces")
        if tr is None or getattr(tr, "ndim", 0) != 2 or tr.shape[0] < 1:
            continue
        cand.append((ch, elec, int(tr.shape[0])))
    if not cand:
        return None
    if prefer:
        for ch, elec, _n in cand:
            if prefer in (ch, elec):
                return ch
    return max(cand, key=lambda t: t[2])[0]


def analyze_recording(evoked_path: str, animal: str, *,
                      prefer_channel: str | None = None,
                      flag_win_ms=DEFAULT_FLAG_WIN_MS,
                      spec_win_ms=DEFAULT_SPEC_WIN_MS,
                      template_iters: int = 1, k: float = 4.0,
                      noise_k: float | None = None, flag_pct: float = 95.0,
                      band=None,
                      artifact_ms=_p.DEFAULT_ARTIFACT_MS) -> dict | None:
    """Full Prong-A analysis for one ``*_evoked.mat`` + *animal*. Returns a result
    dict (see keys below) or None when the recording has no usable channel. The
    stim artifact ``[-artifact_ms[0], +artifact_ms[1]]`` ms is blanked
    (interpolated) FIRST, before the template / residual / spectra."""
    assert evoked_path and animal, "evoked_path and animal required"
    chans = read_file_evoked(evoked_path, only_animals=[animal])
    ch = pick_channel(chans, animal, prefer_channel)
    if ch is None:
        return None
    rec = chans[ch]
    traces = np.asarray(rec["traces"], dtype=np.float64)
    time_ms = np.asarray(rec["time_ms"], dtype=np.float64)
    times = np.asarray(rec.get("times") or [], dtype=np.float64)
    assert traces.ndim == 2 and time_ms.size == traces.shape[1], "shape mismatch"
    fs = 1000.0 / float(np.mean(np.diff(time_ms)))
    traces = _p.blank_artifact_epochs(traces, time_ms, pre_ms=artifact_ms[0],
                                      post_ms=artifact_ms[1])

    post = (time_ms >= flag_win_ms[0]) & (time_ms <= flag_win_ms[1])
    template, keep = _p.robust_template(traces, iters=template_iters, k=k,
                                        win_mask=post)
    resid = _p.residuals(traces, template)
    # Pass 1: a broad provisional flag, only to find the ripple band from the
    # event group's spectrum. Pass 2: the REAL flag on the high-frequency burst
    # (the ripple), so a smooth-response wobble is not mistaken for an event.
    e0 = _p.event_energy(resid, time_ms, win_ms=flag_win_ms, fs=fs, band=band)
    mask0 = _p.flag_events(e0, k=k, noise_k=noise_k)
    clean0 = (~mask0) & keep
    if clean0.sum() < 1:
        clean0 = keep
    if band is not None:
        hf_band = (float(band[0]), float(band[1]))
    else:
        sp0 = _spectra(traces, resid, template, time_ms, fs, spec_win_ms,
                       mask0, clean0)
        hf_band = _hf_band(sp0["fundamental"].get("fundamental_hz"), fs)
    # HF-band RMS = sustained ripple power over the window: a smooth-response
    # wobble has ~none, a transient blip is diluted, only a sustained ripple scores.
    score = _p.event_energy(resid, time_ms, win_ms=flag_win_ms, fs=fs, band=hf_band)
    event_mask = _p.flag_events(score, k=k, noise_k=noise_k, min_pct=flag_pct)
    clean_mask = (~event_mask) & keep            # kept = template's clean cohort
    return _assemble(evoked_path, animal, ch, traces, resid, time_ms, times, fs,
                     template, score, event_mask, clean_mask, keep,
                     flag_win_ms, spec_win_ms, hf_band)


def _assemble(path, animal, ch, traces, resid, time_ms, times, fs, template,
              energy, event_mask, clean_mask, keep, flag_win_ms, spec_win_ms,
              hf_band=None) -> dict:
    """Package the arrays + the five-group spectral comparison."""
    if clean_mask.sum() < 1:                      # degenerate: nothing clean
        clean_mask = keep
    spectra = _spectra(traces, resid, template, time_ms, fs, spec_win_ms,
                       event_mask, clean_mask)
    n = traces.shape[0]
    return {"path": path, "animal": animal, "channel": ch, "fs": fs,
            "time_ms": time_ms, "times": times, "traces": traces,
            "template": template, "resid": resid, "energy": energy,
            "event_mask": event_mask, "clean_mask": clean_mask, "hf_band": hf_band,
            "event_rate": float(event_mask.sum()) / max(1, n), "n": int(n),
            "n_event": int(event_mask.sum()), "n_clean": int(clean_mask.sum()),
            "flag_win_ms": flag_win_ms, "spec_win_ms": spec_win_ms, **spectra}


def _spectra(traces, resid, template, time_ms, fs, win_ms, event_mask,
             clean_mask) -> dict:
    """The five PSD curves + the excess-power readout. Residual-vs-residual
    isolates the event's own spectrum (both had the template removed);
    raw-vs-raw shows it in the untouched signal."""
    ev, cl = event_mask, clean_mask
    f, psd_ev_r, _ = _s.welch_psd(resid[ev], fs, time_ms=time_ms, win_ms=win_ms) \
        if ev.any() else (np.empty(0), np.empty(0), 0)
    _, psd_cl_r, _ = _s.welch_psd(resid[cl], fs, time_ms=time_ms, win_ms=win_ms) \
        if cl.any() else (f, np.empty(0), 0)
    _, psd_ev_raw, _ = _s.welch_psd(traces[ev], fs, time_ms=time_ms, win_ms=win_ms) \
        if ev.any() else (f, np.empty(0), 0)
    _, psd_cl_raw, _ = _s.welch_psd(traces[cl], fs, time_ms=time_ms, win_ms=win_ms) \
        if cl.any() else (f, np.empty(0), 0)
    _, psd_tmpl, _ = _s.welch_psd(template[None, :], fs, time_ms=time_ms,
                                  win_ms=win_ms)
    exc = (_s.excess_db(psd_ev_r, psd_cl_r)
           if psd_ev_r.size and psd_ev_r.size == psd_cl_r.size else np.empty(0))
    line_hz = _s.line_noise_freqs(fs)
    fund = (_s.find_fundamental(f, exc, line_hz=line_hz) if exc.size
            else {"fundamental_hz": float("nan"), "harmonics": [],
                  "n_harmonics": 0, "peak_db": float("nan")})
    eb = _s.excess_band(f, exc) if exc.size else (float("nan"), float("nan"))
    return {"freqs": f, "psd_event_resid": psd_ev_r, "psd_clean_resid": psd_cl_r,
            "psd_event_raw": psd_ev_raw, "psd_clean_raw": psd_cl_raw,
            "psd_template": psd_tmpl, "excess_db": exc, "fundamental": fund,
            "excess_band": eb, "line_hz": line_hz}


def event_prevalence(evoked_path: str, animal: str, *, max_epochs: int = 400,
                     flag_win_ms=DEFAULT_FLAG_WIN_MS, k: float = 4.0) -> dict:
    """Cheap event-rate probe for the auto-pick scan: reads one recording, builds
    the template on a strided sub-sample of epochs (bounded cost), returns
    ``{path, channel, n, n_event, event_rate}`` (event_rate NaN when unusable)."""
    assert evoked_path and animal, "evoked_path and animal required"
    miss = {"path": evoked_path, "channel": None, "n": 0, "n_event": 0,
            "event_rate": float("nan")}
    try:
        chans = read_file_evoked(evoked_path, only_animals=[animal])
    except Exception:                             # noqa: BLE001 -- bad file, skip
        return miss
    ch = pick_channel(chans, animal)
    if ch is None:
        return miss
    tr = np.asarray(chans[ch]["traces"], dtype=np.float64)
    tms = np.asarray(chans[ch]["time_ms"], dtype=np.float64)
    tr = _p.blank_artifact_epochs(tr, tms, pre_ms=_p.DEFAULT_ARTIFACT_MS[0],
                                  post_ms=_p.DEFAULT_ARTIFACT_MS[1])
    step = max(1, tr.shape[0] // int(max_epochs))
    sub = tr[::step]
    fs = 1000.0 / float(np.mean(np.diff(tms)))
    post = (tms >= flag_win_ms[0]) & (tms <= flag_win_ms[1])
    template, _keep = _p.robust_template(sub, iters=1, k=k, win_mask=post)
    hf_band = _hf_band(None, fs)                  # default ripple band for ranking
    score = _p.event_energy(_p.residuals(sub, template), tms, win_ms=flag_win_ms,
                            fs=fs, band=hf_band)
    mask = _p.flag_events(score, k=k, noise_k=None)
    return {"path": evoked_path, "channel": ch, "n": int(sub.shape[0]),
            "n_event": int(mask.sum()),
            "event_rate": float(mask.sum()) / max(1, sub.shape[0])}
