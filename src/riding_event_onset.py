"""Do the RIDING EVENTS have archetypes, and do some precede seizures?

Combines three existing subsystems:
  * ``riding_event`` (Prong A) isolates the oscillatory event riding the evoked
    response: a per-recording robust-median template is subtracted, the residual
    IS the event, and epochs are flagged by band-limited residual energy.
  * ``evoked_typology`` archetypes those event residuals (NMF -> a few prototype
    event shapes + each event's non-negative mixture).
  * the peri-ictal join (``matrix.time_to_onset_for`` + ``isi``) gives each epoch
    its time-to-next-seizure-onset.

Then we ask: does the event RATE rise as onset approaches, and is any event
ARCHETYPE enriched pre-ictally? Coverage-limited (needs recordings inside a
seizure's look-back window) -- the figures state n.

CLI: ``python -m src.riding_event_onset --animal BCH111 --date 2026-09-20 --days 14``
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from src.riding_event import primitives as _p            # noqa: E402
from src.riding_event.residual import pick_channel        # noqa: E402
from src.evoked_typology import normalize_shapes, nmf_prototypes, _dark  # noqa: E402
from src.notifications import evoked_digest as _ed        # noqa: E402
from src.periictal import config as _pcfg                 # noqa: E402
from src.periictal.matrix import time_to_onset_for        # noqa: E402
from src.preictal.isi import included_seizures, lookback_ceilings  # noqa: E402
from src.utils.evoked_output import read_file_evoked, parse_recording_dt  # noqa: E402

_BG, _TEXT, _MUTED, _ACC = "#1e1e2f", "#f0f0f5", "#9a9ab0", "#5e7ce2"
_EVENT_BAND = (150.0, 700.0)             # the known ~200-600 Hz event band + margin
_FLAG_WIN = (2.0, 120.0)


def gather_events(store, animal, evoked_dir, end_day, days, *, band=_EVENT_BAND,
                  flag_win=_FLAG_WIN, k=4.0, cap=8000, seed=0, progress=None):
    """Run Prong-A per recording; collect event-residual waveforms + every epoch's
    (abs time, event flag), then attach time-to-onset. Returns
    ``(R[n_ev,T], tm, ev_tto, all_tto, all_flag, window_sec)``."""
    files = _ed.list_window_files(animal, evoked_dir, end_day, days)
    seizures = included_seizures(store, animal)
    onsets = np.array([s.onset_epoch for s in seizures], dtype=float)
    window_sec = _pcfg.DEFAULT_WINDOW_SEC
    ceilings = np.array(
        [c if c is not None else np.nan for c in
         lookback_ceilings(seizures, _pcfg.DEFAULT_POST_ICTAL_BUFFER_SEC,
                           window_sec)], dtype=float)
    rng = np.random.default_rng(seed)
    R, ev_abs, all_abs, all_flag, seen, tmw = [], [], [], [], 0, [None]
    for i, path in enumerate(files):
        if progress and i % 5 == 0:
            progress(f"prong-A over recordings… ({i}/{len(files)})")
        rec_dt = parse_recording_dt(os.path.basename(path))
        if rec_dt is None:
            continue
        try:
            chans = read_file_evoked(path, only_animals=[animal])
        except Exception:                            # noqa: BLE001
            continue
        ch = pick_channel(chans, animal)
        if ch is None or chans[ch].get("traces") is None:
            continue
        traces = np.asarray(chans[ch]["traces"], dtype=np.float64)
        time_ms = np.asarray(chans[ch]["time_ms"], dtype=np.float64)
        times = np.asarray(chans[ch].get("times") or [], dtype=np.float64)
        if traces.ndim != 2 or times.size != traces.shape[0]:
            continue
        fs = 1000.0 / float(np.mean(np.diff(time_ms)))
        post = (time_ms >= flag_win[0]) & (time_ms <= flag_win[1])
        if post.sum() < 8:
            continue
        template, _keep = _p.robust_template(traces, iters=1, k=k, win_mask=post)
        resid = _p.residuals(traces, template)
        energy = _p.event_energy(resid, time_ms, win_ms=flag_win, fs=fs, band=band)
        mask = _p.flag_events(energy, k=k)
        abs_ep = rec_dt.timestamp() + times          # abs epoch time (s)
        if tmw[0] is None:
            tmw[0] = time_ms[post]
        all_abs.extend(abs_ep.tolist())
        all_flag.extend(mask.tolist())
        for j in np.flatnonzero(mask):
            seen += 1
            row = resid[j][post]
            if row.size != tmw[0].size or not np.all(np.isfinite(row)):
                continue
            if len(R) < cap:
                R.append(row); ev_abs.append(abs_ep[j])
            else:
                r_ = int(rng.integers(0, seen))
                if r_ < cap:
                    R[r_], ev_abs[r_] = row, abs_ep[j]
    all_abs = np.asarray(all_abs)
    all_flag = np.asarray(all_flag, dtype=bool)
    all_tto = time_to_onset_for(all_abs, onsets, ceilings)
    ev_tto = time_to_onset_for(np.asarray(ev_abs), onsets, ceilings) \
        if ev_abs else np.empty(0)
    return (np.asarray(R) if R else None, tmw[0], ev_tto, all_tto, all_flag,
            window_sec)


# --------------------------------------------------------------------- #
#  pure aggregation: rate / archetype prevalence vs time-to-onset
# --------------------------------------------------------------------- #
def rate_by_tto(all_tto, all_flag, window_sec, n_bins=10):
    """Event rate in each equal time-to-onset bin over (0, window]. Returns
    ``(centers_min, rate, n, baseline_rate)`` where baseline = interictal (NaN tto)
    epochs (outside any look-back)."""
    tto = np.asarray(all_tto, dtype=float)
    flag = np.asarray(all_flag, dtype=bool)
    pre = np.isfinite(tto)
    baseline = float(flag[~pre].mean()) if (~pre).any() else float("nan")
    edges = np.linspace(0.0, window_sec, n_bins + 1)
    centers, rate, n = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = pre & (tto >= a) & (tto < b)
        centers.append((a + b) / 2.0 / 60.0)          # minutes to onset
        rate.append(float(flag[m].mean()) if m.any() else np.nan)
        n.append(int(m.sum()))
    return np.array(centers), np.array(rate), np.array(n), baseline


def archetype_by_tto(ev_tto, dom, k, window_sec, n_bins=6):
    """Fraction of EVENT epochs of each archetype in each time-to-onset bin.
    Returns ``(centers_min, frac[k, n_bins], n[n_bins])``."""
    tto = np.asarray(ev_tto, dtype=float)
    dom = np.asarray(dom, dtype=int)
    pre = np.isfinite(tto)
    edges = np.linspace(0.0, window_sec, n_bins + 1)
    centers, n = [], []
    frac = np.full((k, n_bins), np.nan)
    for bi, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = pre & (tto >= a) & (tto < b)
        centers.append((a + b) / 2.0 / 60.0)
        n.append(int(m.sum()))
        if m.any():
            counts = np.bincount(dom[m], minlength=k)
            frac[:, bi] = counts / counts.sum()
    return np.array(centers), frac, np.array(n)


# --------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------- #
def fig_event_archetypes(R, tm, nmf, out):
    """The event ARCHETYPES: NMF signed prototype shapes of the isolated event
    residuals + how often each is the dominant mix."""
    comps, Wt = nmf["components"], nmf["weights"]
    k = comps.shape[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, k))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=_BG)
    _dark(axes[0]); _dark(axes[1])
    for c in range(k):
        axes[0].plot(tm, comps[c], color=colors[c], lw=1.8,
                     label=f"event archetype {c + 1}")
    axes[0].axhline(0, color=_MUTED, lw=0.5, alpha=0.4)
    axes[0].set_title(f"Riding-event ARCHETYPES (NMF of {len(R)} event residuals)",
                      color=_TEXT, fontsize=11)
    axes[0].set_xlabel("ms since stim", color=_MUTED, fontsize=9)
    leg = axes[0].legend(fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(_TEXT)
    dom = np.argmax(Wt, axis=1)
    axes[1].bar(range(1, k + 1), np.bincount(dom, minlength=k) / len(dom),
                color=colors)
    axes[1].set_title("dominant-archetype prevalence", color=_TEXT, fontsize=11)
    axes[1].set_xlabel("archetype", color=_MUTED, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=125, facecolor=_BG); plt.close(fig)
    return out


def fig_rate_vs_tto(all_tto, all_flag, window_sec, out):
    """Event RATE vs time-to-seizure-onset (onset at LEFT) with the interictal
    baseline -- the headline: does the riding event get more common pre-ictally?"""
    centers, rate, n, base = rate_by_tto(all_tto, all_flag, window_sec, n_bins=10)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_BG)
    _dark(ax)
    ok = np.isfinite(rate)
    ax.plot(centers[ok], rate[ok] * 100, color=_ACC, lw=2, marker="o", ms=5,
            label="event rate (pre-ictal window)")
    if np.isfinite(base):
        ax.axhline(base * 100, color="#e08a3c", lw=1.4, ls="--",
                   label=f"interictal baseline ({base * 100:.1f}%)")
    for x, y, nn in zip(centers, rate, n):            # annotate n per bin
        if np.isfinite(y):
            ax.annotate(str(nn), (x, y * 100), color=_MUTED, fontsize=7,
                        ha="center", va="bottom")
    ax.invert_xaxis()                                 # onset at the RIGHT-> left=far
    ax.set_xlabel("minutes to seizure onset (0 = onset, at right)", color=_TEXT,
                  fontsize=10)
    ax.set_ylabel("% of epochs flagged as riding-event", color=_TEXT, fontsize=10)
    npre = int(np.isfinite(np.asarray(all_tto)).sum())
    ax.set_title(f"Riding-event rate vs seizure proximity  (pre-ictal epochs "
                 f"n={npre})", color=_TEXT, fontsize=12)
    leg = ax.legend(fontsize=9, frameon=False)
    for t in leg.get_texts():
        t.set_color(_TEXT)
    fig.tight_layout(); fig.savefig(out, dpi=125, facecolor=_BG); plt.close(fig)
    return out


def fig_archetype_vs_tto(ev_tto, dom, k, window_sec, out):
    """Per-archetype prevalence across time-to-onset bins -- is any event archetype
    enriched approaching onset?"""
    centers, frac, n = archetype_by_tto(ev_tto, dom, k, window_sec, n_bins=6)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, k))
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_BG)
    _dark(ax)
    for c in range(k):
        ok = np.isfinite(frac[c])
        ax.plot(centers[ok], frac[c][ok] * 100, color=colors[c], lw=1.8,
                marker="o", ms=4, label=f"archetype {c + 1}")
    ax.invert_xaxis()
    ax.set_xlabel("minutes to seizure onset (0 = onset, at right)", color=_TEXT,
                  fontsize=10)
    ax.set_ylabel("% of pre-ictal events", color=_TEXT, fontsize=10)
    ax.set_title("Event ARCHETYPE composition vs seizure proximity", color=_TEXT,
                 fontsize=12)
    leg = ax.legend(fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(_TEXT)
    fig.tight_layout(); fig.savefig(out, dpi=125, facecolor=_BG); plt.close(fig)
    return out


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", default="BCH111")
    ap.add_argument("--date", default=None)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--nmf-k", type=int, default=4)
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import yaml
    from src.db.store import Store
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    evoked_dir = cfg["chronic_evoked"]["evoked_output_dir"]
    end = datetime.fromisoformat(a.date) if a.date else datetime.now()
    out = a.out or os.path.join("data", "riding_event_onset")
    os.makedirs(out, exist_ok=True)
    prog = lambda m: print("  ", m)                          # noqa: E731
    print("opening store (slow)…")
    store = Store(cfg["database"]["path"])
    R, tm, ev_tto, all_tto, all_flag, window_sec = gather_events(
        store, a.animal, evoked_dir, end, a.days, cap=a.cap, progress=prog)
    n_all = len(all_flag)
    n_ev = int(all_flag.sum()) if n_all else 0
    n_pre = int(np.isfinite(all_tto).sum()) if n_all else 0
    print(f"epochs={n_all}  events={n_ev} ({100*n_ev/max(1,n_all):.1f}%)  "
          f"pre-ictal epochs={n_pre}")
    if R is None or len(R) < 20:
        print("too few events to archetype"); return 1
    nmf = nmf_prototypes(normalize_shapes(R), k=a.nmf_k)
    dom = np.argmax(nmf["weights"], axis=1)
    p1 = fig_event_archetypes(R, tm, nmf, os.path.join(out, "event_archetypes.png"))
    p2 = fig_rate_vs_tto(all_tto, all_flag, window_sec,
                         os.path.join(out, "rate_vs_onset.png"))
    p3 = fig_archetype_vs_tto(ev_tto, dom, a.nmf_k, window_sec,
                              os.path.join(out, "archetype_vs_onset.png"))
    print("SAVED:", p1); print("SAVED:", p2); print("SAVED:", p3)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
