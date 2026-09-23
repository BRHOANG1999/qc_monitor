"""Plant-and-recover tests for the riding-event core.

Run: pytest tests/test_riding_event.py -q

Each test plants a known signal (a template + a subset of event-carrying epochs,
or a continuous stream with events at known times) and asserts the pure
primitives recover it, plus graceful degeneracy on empty/degenerate input.
"""

import os
import sys
from os.path import abspath, dirname, join

import numpy as np

_ROOT = abspath(join(dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.riding_event import primitives as p          # noqa: E402
from src.riding_event import spectra as sp            # noqa: E402
from src.riding_event import detect as det            # noqa: E402

_FS = 20000.0
_DT = 1000.0 / _FS


def _time_ms():
    return np.arange(-10.0, 100.0, _DT)


def _template_wave(t_ms):
    """A smooth stereotyped evoked response: a decaying deflection for t>0."""
    tt = np.maximum(t_ms, 0.0)
    return 5.0 * np.exp(-tt / 15.0) * np.cos(2 * np.pi * 40.0 * tt / 1000.0)


def _make_epochs(n=120, event_idx=(10, 30, 55, 80), f0=137.0, amp=6.0, seed=0):
    """n epochs = template + noise; the event epochs also carry an f0 burst over
    5-40 ms. Returns (traces, time_ms, event_idx set)."""
    rng = np.random.default_rng(seed)
    t = _time_ms()
    base = _template_wave(t)
    traces = base[None, :] + rng.normal(0, 0.25, size=(n, t.size))
    win = ((t >= 5.0) & (t <= 40.0)).astype(float)
    burst = amp * np.sin(2 * np.pi * f0 * t / 1000.0) * win
    for j in event_idx:
        traces[j] += burst
    return traces, t, set(event_idx)


def test_robust_template_recovers_clean_shape():
    traces, t, ev = _make_epochs()
    post = (t >= 2.0) & (t <= 100.0)
    template, keep = p.robust_template(traces, iters=2, k=4.0, win_mask=post)
    base = _template_wave(t)
    assert np.max(np.abs(template - base)) < 0.6, "template should recover base"
    for j in ev:                                       # event epochs excluded
        assert not keep[j], f"event epoch {j} should be dropped from the template"
    assert keep.sum() >= traces.shape[0] - 2 * len(ev), "kept too few clean epochs"


def test_flag_events_recovers_planted():
    traces, t, ev = _make_epochs()
    template, _ = p.robust_template(traces, iters=1, win_mask=(t >= 2) & (t <= 100))
    resid = p.residuals(traces, template)
    energy = p.event_energy(resid, t, win_ms=(2.0, 100.0), fs=_FS)
    mask = p.flag_events(energy, k=4.0)
    assert set(np.flatnonzero(mask)) == ev, "flagged set must equal planted events"


def test_spectra_recovers_fundamental():
    f0 = 137.0
    traces, t, ev = _make_epochs(f0=f0)
    template, keep = p.robust_template(traces, iters=1,
                                       win_mask=(t >= 2) & (t <= 100))
    resid = p.residuals(traces, template)
    energy = p.event_energy(resid, t, win_ms=(2.0, 100.0), fs=_FS)
    mask = p.flag_events(energy, k=4.0)
    clean = (~mask) & keep
    f, psd_ev, _ = sp.welch_psd(resid[mask], _FS, time_ms=t, win_ms=(2.0, 100.0))
    _, psd_cl, _ = sp.welch_psd(resid[clean], _FS, time_ms=t, win_ms=(2.0, 100.0))
    exc = sp.excess_db(psd_ev, psd_cl)
    fund = sp.find_fundamental(f, exc, line_hz=sp.line_noise_freqs(_FS))
    assert abs(fund["fundamental_hz"] - f0) < 20.0, "fundamental should be ~f0"
    assert fund["peak_db"] > 6.0, "event group should show clear excess power"


def test_rising_edge_align_locks_the_edge():
    rng = np.random.default_rng(1)
    fs = 2000.0
    shape = np.concatenate([np.zeros(20), np.linspace(0, 5, 8),
                            5 * np.exp(-np.arange(40) / 10.0)])
    snips = []
    for _ in range(60):
        jit = int(rng.integers(-6, 7))
        pad = np.zeros(30)
        s = np.concatenate([pad, shape, pad])
        s = np.roll(s, jit) + rng.normal(0, 0.05, size=s.size)
        snips.append(s)
    aligned, _fid = p.rising_edge_align(np.asarray(snips), fs, search_ms=30.0,
                                        pre_ms=5.0, post_ms=15.0)
    pre = max(1, int(round(5.0e-3 * fs)))
    edges = np.argmax(np.diff(aligned, axis=1), axis=1)
    assert aligned.shape[0] >= 55, "most snippets should survive alignment"
    assert np.median(np.abs(edges - pre)) <= 1, "edges should lock near `pre`"


def test_matched_filter_recovers_events():
    rng = np.random.default_rng(2)
    fs = 2000.0
    n = int(20 * fs)
    template = np.sin(2 * np.pi * 8.0 * np.arange(40) / fs) * \
        np.hanning(40) * 5.0
    signal = rng.normal(0, 0.5, size=n)
    planted = np.arange(1000, n - 1000, 2200)
    for loc in planted:
        signal[loc:loc + template.size] += template
    locs, scores, r = p.matched_filter_detect(signal, template, fs, thresh=0.6,
                                               refractory_sec=0.3)
    hit = sum(np.any(np.abs(locs - loc) <= 5) for loc in planted)
    assert hit >= planted.size - 1, "matched filter should recover planted events"
    assert locs.size <= planted.size + 2, "few false positives expected"


def test_detect_candidates_finds_bursts():
    rng = np.random.default_rng(3)
    fs = 2000.0
    n = int(30 * fs)
    t = np.arange(n) / fs
    signal = rng.normal(0, 0.3, size=n)
    centers = np.arange(2 * int(fs), n - 2 * int(fs), 5 * int(fs))
    for c in centers:
        w = np.zeros(n)
        seg = slice(c, c + int(0.2 * fs))
        w[seg] = np.hanning(int(0.2 * fs))
        signal += 4.0 * np.sin(2 * np.pi * 120.0 * t) * w
    locs, env, thr, thr_hi = p.detect_candidates(signal, fs, band=(20.0, 200.0),
                                                 min_dist_sec=1.0, k=4.0)
    for c in centers:
        assert np.any(np.abs(locs - c) <= int(0.3 * fs)), "should detect each burst"


def test_noise_threshold_rejects_giant_transient():
    rng = np.random.default_rng(8)
    fs = 2000.0
    n = int(30 * fs)
    t = np.arange(n) / fs
    signal = rng.normal(0, 0.3, size=n)
    centers = np.arange(2 * int(fs), n - 2 * int(fs), 6 * int(fs))
    for c in centers:                                  # modest-amplitude events
        w = np.zeros(n)
        w[c:c + int(0.2 * fs)] = np.hanning(int(0.2 * fs))
        signal += 0.6 * np.sin(2 * np.pi * 120.0 * t) * w
    glitch = 10000                                      # between centers (not
    signal[glitch:glitch + 4] += 400.0                  # near any real event)
    lo_only, _, _, _ = p.detect_candidates(signal, fs, band=(20.0, 200.0),
                                           min_dist_sec=1.0, k=4.0)
    gated, _, _, thr_hi = p.detect_candidates(signal, fs, band=(20.0, 200.0),
                                              min_dist_sec=1.0, k=4.0, noise_k=20.0)
    assert np.any(np.abs(lo_only - glitch) <= int(0.3 * fs)), "glitch is a candidate"
    assert not np.any(np.abs(gated - glitch) <= int(0.3 * fs)), "ceiling drops glitch"
    for c in centers:                                  # real events survive
        assert np.any(np.abs(gated - c) <= int(0.3 * fs)), "real event kept"


def test_artifact_blanking_removes_spike_keeps_event():
    t = _time_ms()
    base = _template_wave(t)
    rng = np.random.default_rng(7)
    traces = base[None, :] + rng.normal(0, 0.2, size=(20, t.size))
    art = (t >= -1.0) & (t <= 2.0)
    traces[:, art] += 60.0                              # huge stim artifact
    b = p.blank_artifact_epochs(traces, t, pre_ms=1.0, post_ms=2.0)
    assert np.nanmax(np.abs(b[:, art])) < 8.0, "artifact should be interpolated out"
    post = t > 3.0
    assert np.allclose(b[:, post], traces[:, post]), "post-stim event untouched"

    fs = 2000.0
    n = int(10 * fs)
    x = rng.normal(0, 0.3, size=n)
    stim = np.arange(1000, n - 1000, 2000)
    for s in stim:
        x[s - 2:s + 4] += 90.0                          # artifact spikes
    ev0 = stim[0] + 500
    x[ev0:ev0 + 40] += 5.0
    keep = x[ev0:ev0 + 40].copy()
    xb = p.blank_artifact_continuous(x, stim, pre=2, post=4)
    for s in stim:
        assert abs(xb[s]) < 15.0, "continuous artifact should be interpolated out"
    assert np.allclose(xb[ev0:ev0 + 40], keep), "distant event untouched"


def test_stim_blanking_and_amplitude_gate():
    rng = np.random.default_rng(4)
    fs = 2000.0
    n = int(24 * fs)
    template = np.sin(2 * np.pi * 120.0 * np.arange(40) / fs) * np.hanning(40) * 6.0
    signal = rng.normal(0, 0.4, size=n)
    planted = np.arange(2000, n - 2000, 4000)
    for loc in planted:
        signal[loc:loc + template.size] += template
    stim_idx = [0, 2, 4]                                # 3 planted events are "stim"
    stim_times = [planted[i] / fs for i in stim_idx]
    d = det.run_detector(signal, fs, band=(20.0, 200.0), min_dist_sec=0.3,
                         thresh=0.5, refractory_sec=0.3, template_override=template,
                         exclude_times_sec=stim_times, exclude_pad_sec=0.15,
                         noise_k=None)                  # isolate blanking + amp gate
    dl = np.asarray(d["det_locs"])
    for i in stim_idx:                                  # stim-coincident excluded
        assert not np.any(np.abs(dl - planted[i]) <= 200), "stim event not blanked"
    spont = [i for i in range(planted.size) if i not in stim_idx]
    hit = sum(np.any(np.abs(dl - planted[i]) <= 20) for i in spont)
    assert hit >= len(spont) - 1, "spontaneous events should survive the gate"


def test_degenerate_inputs():
    z = np.zeros((10, 200))
    template, keep = p.robust_template(z, iters=1)
    assert np.allclose(template, 0.0) and keep.all(), "zeros -> zero template"
    energy = p.event_energy(p.residuals(z, template), np.arange(200) * _DT,
                            win_ms=(0.0, 5.0), fs=_FS)
    assert not p.flag_events(energy).any(), "no events in flat input"
    r = p.matched_filter_series(np.zeros(100), np.zeros(20))
    assert np.allclose(r, 0.0), "zero template -> zero correlation"
    assert det.build_template(np.zeros(1000), _FS, np.array([1, 2])) == {}, \
        "too few candidates -> empty template"
