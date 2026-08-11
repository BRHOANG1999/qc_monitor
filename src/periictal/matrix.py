"""Join per-stimulus evoked features to the scored-seizure timeline.

Produces a tidy ``event x metric`` DataFrame in which every stimulus that falls
in a lead-up window before a seizure onset carries a CONTINUOUS
``time_to_onset_sec`` (the label the explorer colours by). The join is computed
at QUERY TIME, never baked into the sidecars: scoring one new seizure retroact-
ively rewrites the lead-time of every preceding stimulus, so the label must be
derived from the current seizure list on each build.

Reuses:
  * ``src.evoked_figures.data.iter_animal_sidecars`` -- fresh-sidecar rows.
  * ``src.preictal.isi.scored_seizures`` -- the EO+Racine seizure timeline.
  * ``src.preictal.isi.lookback_ceilings`` -- per-seizure ISI ceiling (with a
    post-ictal buffer) so a lead-up window can't reach into the prior seizure's
    post-ictal tail; also caps the window length.
  * ``src.preictal.isi.leadtime_bins`` -- dyadic lead-time bins for stratified
    subsampling.
  * ``src.periictal.stim_map`` -- the STIM_REPORT hardware fingerprint.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.evoked_figures.data import _nan, iter_animal_sidecars, parse_iso
from src.periictal import config as _cfg
from src.periictal import passive as _passive
from src.periictal import stim_map as _sm
from src.preictal.isi import (included_seizures, leadtime_bins,
                              lookback_ceilings)
from src.utils.evoked_output import parse_recording_dt

_MAX_FILES = 2_000_000        # NASA Rule 2: explicit scan bound.


def _near_seizure_file_filter(onsets, window_sec, slack):
    """Filename-time predicate: keep a recording only if its start could contain
    a stimulus within *window_sec* of some onset (+ *slack* for the chunk's own
    length). A PROVABLE SUPERSET of files that yield a kept row -- every kept
    stimulus lies within +/- window_sec of its onset (pre ceiling <= window_sec;
    post window == window_sec) and a file starts no later than its stimuli -- so
    the matrix is identical, only files whose every stimulus is dropped anyway
    are skipped. Unparseable filenames are KEPT (never drop on uncertainty)."""
    lo = np.asarray(onsets, dtype=np.float64) - float(window_sec) - float(slack)
    hi = np.asarray(onsets, dtype=np.float64) + float(window_sec) + float(slack)

    def _keep(fp) -> bool:
        dt = parse_recording_dt(fp)
        if dt is None:
            return True
        ts = dt.timestamp()
        return bool(np.any((ts >= lo) & (ts <= hi)))
    return _keep


def _sidecar_iter(animal: str, evoked_dir: str, sidecar_variant: str, feature_cfg,
                  warm_missing: bool = False, file_filter=None):
    """The (mat, sidecar, rows) generator for a sidecar variant. The shared
    default 'evoked' sidecar (full trace, no window) is read directly; any other
    variant ('passive', 'evokedw', …) is a config-signed windowed sidecar and
    needs its FeatureConfig to read the right file. *warm_missing* recomputes a
    stale/missing default sidecar on the fly (self-heals a version bump)."""
    if sidecar_variant in (None, "evoked"):
        return iter_animal_sidecars(animal, evoked_dir, compute_missing=warm_missing,
                                    file_filter=file_filter)
    cfg = feature_cfg or _passive.passive_config()
    # The windowed variants (passive/evokedw) don't take the prefilter yet -- the
    # common evoked path above does; passive builds are correct, just unfiltered.
    return _passive.iter_variant_sidecars(animal, evoked_dir, sidecar_variant, cfg)


def _epoch_columns(animal: str, evoked_dir: str, metrics: list[str],
                   sidecar_variant: str, feature_cfg, warm_missing: bool = False,
                   file_filter=None):
    """One pass over *animal*'s fresh sidecars -> parallel arrays:
    (t_epoch[N], metric_arrays{m: f32[N]}, channel[N] obj, session[N] obj).
    Session/channel are kept per row so the protocol filter + fingerprint can
    attach; a stimulus with no parseable abs_dt is dropped here."""
    assert animal, "animal required"
    t_parts, ch_parts, se_parts, rec_parts = [], [], [], []
    m_parts: dict = {m: [] for m in metrics}
    for i, (_fp, _sp, rows) in enumerate(
            _sidecar_iter(animal, evoked_dir, sidecar_variant, feature_cfg,
                          warm_missing, file_filter=file_filter)):
        assert i < _MAX_FILES, "sidecar scan runaway"
        if not rows:
            continue
        secs = np.fromiter(
            ((parse_iso(r.get("abs_dt")) or datetime.min).timestamp()
             if r.get("abs_dt") else np.nan for r in rows),
            dtype=np.float64, count=len(rows))
        t_parts.append(secs)
        ch_parts.append(np.array([r.get("channel") or "" for r in rows], dtype=object))
        se_parts.append(np.array([r.get("session") or "" for r in rows], dtype=object))
        # rec_dt (recording start) is the natural per-recording id for drill-down.
        rec_parts.append(np.array([r.get("rec_dt") or "" for r in rows], dtype=object))
        for m in metrics:
            m_parts[m].append(
                np.fromiter((_nan(r.get(m)) for r in rows),
                            dtype=np.float32, count=len(rows)))
    if not t_parts:
        return (np.empty(0), {m: np.empty(0, np.float32) for m in metrics},
                np.empty(0, object), np.empty(0, object), np.empty(0, object))
    t = np.concatenate(t_parts)
    mcols = {m: np.concatenate(m_parts[m]) for m in metrics}
    return (t, mcols, np.concatenate(ch_parts), np.concatenate(se_parts),
            np.concatenate(rec_parts))


def _assign_next_onset(t: np.ndarray, onsets: np.ndarray,
                       ceilings: np.ndarray):
    """Vectorized nearest-UPCOMING-onset assignment.

    Returns (idx[N], tto[N], keep[N]) where idx = index of the first onset
    strictly after each t, tto = that onset minus t, and keep marks stimuli that
    are (a) before some onset and (b) within that seizure's lookback ceiling
    (which already excludes the previous seizure's post-ictal tail and caps the
    window). NaN ceilings (first-of-animal / buffer-ate-the-ISI) are dropped."""
    n_sz = onsets.size
    idx = np.searchsorted(onsets, t, side="right")   # first onset > t
    has_next = idx < n_sz
    safe = np.where(has_next, idx, 0)
    ceil = ceilings[safe]
    tto = np.where(has_next, onsets[safe] - t, np.nan)
    keep = has_next & np.isfinite(ceil) & (tto > 0.0) & (tto <= ceil)
    return safe, tto, keep


def _assign_prev_onset(t: np.ndarray, onsets: np.ndarray,
                       ceilings: np.ndarray):
    """Vectorized nearest-PAST-onset assignment for the post-ictal window (the
    positive-control substrate — symmetric to ``_assign_next_onset``).

    Returns (idx[N], tto[N], keep[N]) where idx = index of the LAST onset
    strictly before each t, tto = that onset minus t (NEGATIVE — a post-ictal row
    reads as time PAST the onset), and keep marks stimuli within that seizure's
    post-window ceiling. Nearest-previous-onset attribution keeps each post row
    tied to its immediately preceding seizure, so the ceiling is a plain window
    length (no reach into the next seizure)."""
    assert onsets.ndim == 1, "onsets must be 1-D"
    assert ceilings.shape == onsets.shape, "ceilings/onsets shape mismatch"
    idx = np.searchsorted(onsets, t, side="left") - 1   # last onset < t
    has_prev = idx >= 0
    safe = np.where(has_prev, idx, 0)
    ceil = ceilings[safe]
    since = np.where(has_prev, t - onsets[safe], np.nan)   # time since onset, >0
    tto = -since                                           # negative
    keep = has_prev & np.isfinite(ceil) & (since > 0.0) & (since <= ceil)
    return safe, tto, keep


def _lead_bins_for(ceilings: np.ndarray, min_leadtime_sec: float) -> dict:
    """Per-seizure dyadic bin edges, indexed by seizure position (skips NaN
    ceilings). Built once so the row-wise bin lookup is a cheap searchsorted."""
    out: dict = {}
    for j, c in enumerate(ceilings):
        if np.isfinite(c) and c > min_leadtime_sec:
            out[j] = np.asarray(leadtime_bins(float(c), min_leadtime_sec).edges_sec)
    return out


def build_matrix(store, animal: str, evoked_dir: str, *,
                 protocol: str | None = None,
                 window_sec: float = _cfg.DEFAULT_WINDOW_SEC,
                 post_ictal_buffer_sec: float = _cfg.DEFAULT_POST_ICTAL_BUFFER_SEC,
                 min_leadtime_sec: float = _cfg.DEFAULT_MIN_LEADTIME_SEC,
                 metrics: list[str] | None = None,
                 variant: str = "evoked",
                 feature_cfg=None,
                 sidecar_variant: str | None = None,
                 passive_cfg=None,
                 attach_fingerprint: bool = True,
                 warm_missing: bool = False) -> pd.DataFrame:
    """Tidy ``event x metric`` frame for *animal*: one row per lead-up stimulus,
    with continuous ``time_to_onset_sec``, the seizure it precedes, its lead-time
    bin, hour-of-day, protocol token and stim fingerprint. Empty frame when the
    animal has <2 seizures or no fresh sidecars. See module docstring."""
    assert animal and evoked_dir, "animal and evoked_dir required"
    assert window_sec > 0, "window_sec must be > 0"
    metrics = metrics or _cfg.metrics_for_variant(variant)
    # sidecar_variant selects which sidecar to read (defaults from the metric
    # variant); feature_cfg is its windowing (passive_cfg kept as a legacy alias).
    feature_cfg = feature_cfg if feature_cfg is not None else passive_cfg
    if sidecar_variant is None:
        sidecar_variant = "passive" if variant == "passive" else "evoked"
    seizures = included_seizures(store, animal)
    onsets = np.array([s.onset_epoch for s in seizures], dtype=np.float64)
    if onsets.size < 2:
        return _empty_frame(metrics)
    ceilings = np.array(
        [c if c is not None else np.nan
         for c in lookback_ceilings(seizures, post_ictal_buffer_sec, window_sec)],
        dtype=np.float64)

    # Near-seizure prefilter: skip files that can't contribute a kept row (only
    # the evoked path; a provable superset -> matrix unchanged). Big win on a
    # cold build (BCH111: ~127 of 351 files read instead of all).
    file_filter = None
    if (_cfg.PERIICTAL_PREFILTER_NEAR_SEIZURE
            and sidecar_variant in (None, "evoked")):
        file_filter = _near_seizure_file_filter(
            onsets, window_sec, _cfg.PERIICTAL_PREFILTER_SLACK_SEC)

    t, mcols, chan, sess, rec = _epoch_columns(animal, evoked_dir, metrics,
                                               sidecar_variant, feature_cfg,
                                               warm_missing, file_filter=file_filter)
    if t.size == 0:
        return _empty_frame(metrics)
    idx_pre, tto_pre, keep_pre = _assign_next_onset(t, onsets, ceilings)
    # Symmetric post-onset window (phase="post") — the positive-control substrate.
    # A stimulus may legitimately be BOTH a pre-row (upcoming seizure) and a
    # post-row (previous seizure); the two phases are analysed separately.
    post_ceil = np.full(onsets.size, float(window_sec), dtype=np.float64)
    idx_post, tto_post, keep_post = _assign_prev_onset(t, onsets, post_ceil)
    if protocol:
        # Keep a stimulus only when BOTH it and the seizure it flanks are in the
        # requested protocol -- a clean within-protocol scope (no mixing stim
        # conditions across the join).
        epoch_ok = np.array([protocol in s for s in sess], dtype=bool)
        keep_pre &= epoch_ok & np.array(
            [protocol in _token(seizures[j].session_dir) for j in idx_pre], bool)
        keep_post &= epoch_ok & np.array(
            [protocol in _token(seizures[j].session_dir) for j in idx_post], bool)
    if not (keep_pre.any() or keep_post.any()):
        return _empty_frame(metrics)

    frames = []
    sel_pre = np.flatnonzero(keep_pre)
    if sel_pre.size:
        fp = _assemble(sel_pre, t, tto_pre, idx_pre, mcols, chan, sess, rec,
                       seizures, metrics, "pre")
        _add_lead_bins(fp, idx_pre[sel_pre], ceilings, min_leadtime_sec)
        frames.append(fp)
    sel_post = np.flatnonzero(keep_post)
    if sel_post.size:
        fq = _assemble(sel_post, t, tto_post, idx_post, mcols, chan, sess, rec,
                       seizures, metrics, "post")
        fq["lead_bin"] = np.full(len(fq), -1, dtype=np.int32)   # n/a post-onset
        frames.append(fq)
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if attach_fingerprint:
        _attach_fingerprint(frame, store, animal)
    return frame


# --------------------------------------------------------------------- #
#  Assembly helpers
# --------------------------------------------------------------------- #

# The non-metric columns the frame always carries.
_META = ["t_epoch", "abs_dt", "channel", "session", "rec", "time_to_onset_sec",
         "seizure_idx", "seizure_onset_epoch", "seizure_racine", "hour_of_day",
         "phase", "lead_bin", "stim_key", "stim_status"]


def _token(session_dir: str) -> str:
    """Protocol-folder token: basename up to the first ``__``."""
    return os.path.basename(session_dir or "").split("__", 1)[0]


def _empty_frame(metrics: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in
                         _META + list(metrics)})


def _assemble(sel, t, tto, idx, mcols, chan, sess, rec, seizures, metrics,
              phase):
    """Build the DataFrame for the kept stimuli (index array *sel*), tagged with
    the peri-onset *phase* ('pre' | 'post')."""
    sz_idx = idx[sel]
    data = {
        "t_epoch": t[sel],
        "abs_dt": [datetime.fromtimestamp(x).isoformat() for x in t[sel]],
        "channel": chan[sel],
        "session": sess[sel],
        "rec": rec[sel],
        "time_to_onset_sec": tto[sel],
        "seizure_idx": sz_idx.astype(np.int32),
        "seizure_onset_epoch": np.array([seizures[j].onset_epoch for j in sz_idx]),
        "seizure_racine": np.array([_int_or(seizures[j].racine) for j in sz_idx]),
        "hour_of_day": np.array([_hour_of_day(x) for x in t[sel]]),
        "phase": np.array([phase] * sz_idx.size, dtype=object),
    }
    for m in metrics:
        data[m] = mcols[m][sel]
    return pd.DataFrame(data)


def _int_or(v, default=-1) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _hour_of_day(epoch_sec: float) -> float:
    d = datetime.fromtimestamp(epoch_sec)
    return d.hour + d.minute / 60.0 + d.second / 3600.0


def _add_lead_bins(frame: pd.DataFrame, sz_idx: np.ndarray,
                   ceilings: np.ndarray, min_leadtime_sec: float) -> None:
    """Assign each row its dyadic lead-time bin within its seizure's window."""
    edges_by_sz = _lead_bins_for(ceilings, min_leadtime_sec)
    tto = frame["time_to_onset_sec"].to_numpy()
    bins = np.full(len(frame), -1, dtype=np.int32)
    for k in range(len(frame)):
        edges = edges_by_sz.get(int(sz_idx[k]))
        if edges is None or edges.size < 2:
            continue
        b = int(np.searchsorted(edges, tto[k], side="right") - 1)
        bins[k] = min(max(b, 0), edges.size - 2)
    frame["lead_bin"] = bins


def _attach_fingerprint(frame: pd.DataFrame, store, animal: str) -> None:
    """Attach the STIM_REPORT hardware fingerprint (key + status) per row,
    resolved once per protocol token. Rows whose token has no session (or an
    unreachable share) get status 'unknown' and key = the token itself."""
    tok2dir = _sm.session_dir_by_token(store, animal)
    cache: dict = {}
    keys, stats = [], []
    for tok in frame["session"]:
        if tok not in cache:
            sd = tok2dir.get(tok)
            fp = _sm.resolve_fingerprint(store, sd, animal) if sd else \
                _sm.StimFingerprint("unknown")
            cache[tok] = (fp.key() if fp.status == "mapped" else tok, fp.status)
        keys.append(cache[tok][0])
        stats.append(cache[tok][1])
    frame["stim_key"] = keys
    frame["stim_status"] = stats
