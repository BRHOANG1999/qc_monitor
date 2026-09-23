"""Prong B: a stim-agnostic ictal-event detector.

Detect candidate events on the continuous LFP (band envelope -> peak detection),
align them by their rising edge, build a robust median template, then detect
further events with a normalized-correlation (matched-filter) threshold. The same
template also re-detects the stim-locked events isolated in Prong A (internal
cross-check), and Prong A's artifact-free residual windows can seed the template.

I/O boundary: ``load_channel`` reads one continuous recording via ``get_chunk``
(off-GIL only on dashboard threads; in-process in the CLI). Everything else is
pure over arrays from ``primitives``.
"""

from __future__ import annotations

import logging

import numpy as np

from src.chronic_stability import metrics as _m
from src.riding_event import primitives as _p
from src.utils.animal import is_animal_channel, split_animal_electrode

logger = logging.getLogger("qc_monitor.riding_event.detect")

_MIN_TEMPLATE_SNIPS = 12         # below this a template is noise, not a shape.


def load_channel(file_path: str, animal: str,
                 prefer: str | None = None) -> tuple | None:
    """(signal_1d, fs, channel_name) for *animal*'s electrode in a raw recording.

    Reads the continuous ``.mat`` via the shared chunk cache and selects the
    animal's channel (``prefer`` electrode/name wins; else the first animal
    channel that follows a stimCopy, matching the dashboard's display channel).
    Returns None when the file carries no animal channel."""
    from src.utils.chunk_cache import get_chunk
    assert file_path and animal, "file_path and animal required"
    chunk = get_chunk(file_path)
    names = list(getattr(chunk, "channel_names", []) or [])
    idx = _animal_channel_index(names, animal, prefer)
    if idx is None:
        return None
    return np.asarray(chunk.signal[:, idx], dtype=np.float64), float(chunk.fs), \
        names[idx]


def _animal_channel_index(channel_names, animal: str,
                          prefer: str | None = None) -> int | None:
    """Index of *animal*'s electrode: ``prefer`` first, else the animal channel
    that immediately follows a stimCopy (the lab's display channel), else the
    first animal channel. None when the animal isn't present."""
    names = list(channel_names or [])
    mine = [i for i, n in enumerate(names)
            if isinstance(n, str) and split_animal_electrode(n)[0] == animal
            and is_animal_channel(n)]
    if not mine:
        return None
    if prefer:
        for i in mine:
            ch = names[i]
            if prefer in (ch, split_animal_electrode(ch)[1]):
                return i
    for i in mine:                                # prefer the post-stimCopy one
        if i > 0 and isinstance(names[i - 1], str) and "stimcopy" in names[i - 1].lower():
            return i
    return mine[0]


def build_template(signal, fs: float, locs, *, pre_ms: float = 5.0,
                   post_ms: float = 25.0, search_ms: float = 10.0,
                   refit_iters: int = 2, corr_keep: float = 0.5) -> dict:
    """Rising-edge-aligned robust median template from candidate *locs*.

    Snippets around each candidate -> ``rising_edge_align`` -> median template,
    then *refit_iters* passes keeping only snippets that correlate >= ``corr_keep``
    to the running template (``metrics.template_correlation``) and re-medianing.
    Returns ``{template, aligned, corr, kept, n}`` ({} when too few snippets)."""
    assert fs and fs > 0, "fs must be positive"
    pre = max(1, int(round(pre_ms * 1e-3 * fs)))
    post = max(1, int(round(post_ms * 1e-3 * fs)))
    search = max(2, int(round((pre_ms + search_ms) * 1e-3 * fs)))
    snips, _kept = _p.snippets_around(signal, locs, pre=pre, post=post + search)
    if snips.shape[0] < _MIN_TEMPLATE_SNIPS:
        return {}
    aligned, _fid = _p.rising_edge_align(snips, fs, search_ms=search_ms,
                                         pre_ms=pre_ms, post_ms=post_ms)
    if aligned.shape[0] < _MIN_TEMPLATE_SNIPS:
        return {}
    return _refit(aligned, refit_iters, corr_keep)


def _refit(aligned, refit_iters: int, corr_keep: float) -> dict:
    """Median template + correlation-gated robust refit over aligned snippets."""
    template = np.median(aligned, axis=0)
    keep = np.ones(aligned.shape[0], dtype=bool)
    corr = _m.template_correlation(aligned, template)
    for _ in range(int(min(refit_iters, 10))):    # NASA Rule 2: bounded
        corr = _m.template_correlation(aligned, template)
        new_keep = np.isfinite(corr) & (corr >= float(corr_keep))
        if new_keep.sum() < _MIN_TEMPLATE_SNIPS or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= _MIN_TEMPLATE_SNIPS else keep
            break
        keep = new_keep
        template = np.median(aligned[keep], axis=0)
    return {"template": template, "aligned": aligned, "corr": corr,
            "kept": keep, "n": int(keep.sum())}


def harvest_from_prongA(res_a: dict, *, search_ms: float = 10.0,
                        pre_ms: float = 5.0, post_ms: float = 25.0) -> dict:
    """Rising-edge-aligned template from Prong A's stim-locked event residuals —
    artifact-free examples (the stim artifact was removed by template
    subtraction). ``res_a`` is ``residual.analyze_recording``'s result. Returns
    the same shape as ``build_template`` ({} when too few event epochs)."""
    assert res_a and "resid" in res_a, "need a Prong-A result"
    ev = res_a["event_mask"]
    if int(ev.sum()) < _MIN_TEMPLATE_SNIPS:
        return {}
    resid = res_a["resid"][ev]
    time_ms = np.asarray(res_a["time_ms"], dtype=np.float64)
    fs = float(res_a["fs"])
    post = (time_ms >= res_a["flag_win_ms"][0]) & (time_ms <= res_a["flag_win_ms"][1])
    if post.sum() < 4:
        return {}
    aligned, _fid = _p.rising_edge_align(resid[:, post], fs, search_ms=search_ms,
                                         pre_ms=pre_ms, post_ms=post_ms)
    if aligned.shape[0] < _MIN_TEMPLATE_SNIPS:
        return {}
    return _refit(aligned, 2, 0.5)


def run_detector(signal, fs: float, *, band=(20.0, 200.0),
                 min_dist_sec: float = 0.05, k: float = 4.0,
                 pre_ms: float = 5.0, post_ms: float = 25.0,
                 search_ms: float = 10.0, thresh: float = 0.7,
                 refractory_sec: float = 0.05,
                 template_override=None) -> dict:
    """End-to-end Prong B on one continuous channel: candidates -> template ->
    matched-filter detections. Returns everything the renderer needs, including
    the intermediate envelope/threshold and the correlation series.

    ``template_override`` (e.g. Prong A's clean, artifact-removed stim-locked
    template) is used for the matched filter when given; the self-bootstrapped
    continuous template is still built (and returned for the alignment figure)."""
    x = np.asarray(signal, dtype=np.float64)
    assert x.ndim == 1 and fs and fs > 0, "signal 1-D, fs > 0"
    locs, env, thr = _p.detect_candidates(x, fs, band=band,
                                          min_dist_sec=min_dist_sec, k=k)
    tmpl = build_template(x, fs, locs, pre_ms=pre_ms, post_ms=post_ms,
                          search_ms=search_ms)
    out = {"fs": fs, "band": band, "cand_locs": locs, "env": env, "cand_thr": thr,
           "template": tmpl, "det_locs": np.empty(0, dtype=np.int64),
           "det_scores": np.empty(0), "r_series": np.empty(0), "thresh": thresh,
           "template_source": None}
    mf = np.asarray(template_override, dtype=np.float64) \
        if template_override is not None else None
    if mf is None and tmpl and tmpl.get("template") is not None:
        mf = tmpl["template"]
        out["template_source"] = "continuous"
    elif mf is not None:
        out["template_source"] = "prongA"
    if mf is not None and mf.size >= 2:
        det_locs, scores, r = _p.matched_filter_detect(
            x, mf, fs, thresh=thresh, refractory_sec=refractory_sec)
        out.update(det_locs=det_locs, det_scores=scores, r_series=r)
    return out


def scores_by_proximity(det_locs_sec, onsets_sec, *, window_sec: float = 300.0
                        ) -> dict:
    """Split detection times into near vs far from scored seizure onsets (for the
    validation histogram). ``det_locs_sec`` and ``onsets_sec`` are absolute
    seconds. Returns ``{near_frac, n_near, n_far, n_onsets}``."""
    d = np.asarray(det_locs_sec, dtype=np.float64)
    o = np.asarray(onsets_sec, dtype=np.float64)
    if d.size == 0 or o.size == 0:
        return {"near_frac": float("nan"), "n_near": 0, "n_far": int(d.size),
                "n_onsets": int(o.size)}
    near = np.zeros(d.shape[0], dtype=bool)
    for on in o:                                  # bounded by onset count
        near |= np.abs(d - on) <= float(window_sec)
    n_near = int(near.sum())
    return {"near_frac": n_near / max(1, d.size), "n_near": n_near,
            "n_far": int(d.size - n_near), "n_onsets": int(o.size)}
