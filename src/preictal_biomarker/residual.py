"""Residual-waveform variant (separate analysis branch).

Instead of 8 engineered features, represent each epoch by its **residual evoked
waveform** = the epoch's 2-50 ms trace minus the GLOBAL AVERAGE evoked waveform
(mean across all epochs). PCA is run directly on those residual waveforms; the
k-means states, occupancy + circular-shift null, per-seizure trajectory, timeline
and animations then reuse the representation-agnostic backend unchanged.

Design notes
- Traces are read FULL (all post-stim samples), 1-500 Hz bandpassed, then windowed
  to 2-50 ms and decimated by ``DECIM`` -- the SAME band as the main branch's LP500
  features (via ``evoked_features.preprocess`` / ``_bandpass``), so the residual
  representation and the engineered features are on one band (no HF-noise confound).
- Per-epoch time (``abs_dt``) is taken from the warmed feature sidecar, paired to
  the windowed traces by index (validated per file); the seizure join reuses the
  committed ``periictal.matrix`` helpers.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.notifications.evoked_stim_corr import (_file_dt, _list_files, _match_key)
from src.utils.evoked_output import read_feature_sidecar
from src.utils import evoked_features as _ef
from src.periictal import matrix as _mx, passive as _passive, config as _pcfg
from src.preictal import isi as _isi
from . import config as C, features as F, states as S

DECIM = 4                                   # keep every 4th sample (~5 kHz)
# LP500 preprocess config (1-500 Hz bandpass on the full trace, then 2-50 ms crop)
# -- identical band to the engineered-feature sidecars (evoked_windowed._bandpass).
_LP_CFG = _ef.FeatureConfig(window_start_ms=C.WINDOW_MS[0],
                            window_end_ms=C.WINDOW_MS[1], bandpass=C.BANDPASS,
                            bp_low_hz=C.BP_LOW_HZ, bp_high_hz=C.BP_HIGH_HZ)
_RES_CACHE = C.CACHE_DIR + "/BCH111_residual_matrix.npz"
_RES_CACHE_FULL = C.CACHE_DIR + "/BCH111_residual_matrix_full.npz"


def _log(m):
    print(f"[preictal_biomarker.residual] {m}", flush=True)


def _epoch_abs_dt(fp) -> np.ndarray | None:
    """Per-epoch abs_dt (epoch sec) for BCH111SR from the warmed LP sidecar."""
    rows = read_feature_sidecar(fp, C.ANIMAL, "evokedw",
                                _passive.config_sig(F.lp_cfg()))
    if not rows:
        return None
    from src.evoked_figures.data import parse_iso
    out = []
    for r in rows:
        if (r.get("channel") or "") != C.CHANNEL:
            continue
        d = parse_iso(r.get("abs_dt")) if r.get("abs_dt") else None
        out.append(d.timestamp() if d else np.nan)
    return np.asarray(out, float) if out else None


def _full_traces(path, channel):
    """(traces[:, all_samples], time_ms, fs) -- FULL evokedData read so the 1-500 Hz
    bandpass can run on the whole post-stim trace BEFORE windowing (a 1 Hz high-pass
    is meaningless on a 48 ms crop). fs from the median time step. None on failure."""
    import h5py
    try:
        with h5py.File(path, "r") as g:
            grp = g.get("allAnimalResults")
            if grp is None:
                return None
            node = grp.get(channel) or grp.get(_match_key(list(grp.keys()), channel))
            if node is None or "evokedData" not in node or "timeAxis" not in node:
                return None
            t = np.asarray(node["timeAxis"][()], float).ravel()
            tr = np.asarray(node["evokedData"][()], float)
    except (OSError, KeyError, ValueError, TypeError):
        return None
    if tr.ndim != 2 or tr.shape[1] != t.size or t.size < 4:
        return None
    dt = float(np.median(np.diff(t)))                    # ms
    if not np.isfinite(dt) or dt <= 0:
        return None
    return tr, t, 1000.0 / dt


def build_residual_matrix(store, evoked_dir: str, *, force=False,
                          full_record=False) -> dict:
    """Residual waveform matrix for BCH111SR. Returns dict(res[N,S], t[N],
    rec[N], time_ms[S], mean_wave[S]) and caches to npz.

    full_record=False (default): peri-ictal prefilter (epochs within ~8 h of a
    seizure) -- the scope for the occupancy/null analysis. full_record=True: ALL
    in-scope-date epochs (no peri-ictal filter) -- so the continuous timeline has
    no artificial gaps on days far from any seizure."""
    cache = _RES_CACHE_FULL if full_record else _RES_CACHE
    if not force and os.path.exists(cache):
        _log(f"loading cached residual matrix: {cache}")
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    if full_record:
        s0, s1 = C.ANALYSIS_START, C.ANALYSIS_END
        files = [f for f in _list_files(C.ANIMAL, evoked_dir, None)
                 if (_file_dt(f) is not None and s0 <= _file_dt(f) < s1)]
    else:
        onsets = F.scoped_onsets(store)
        filt = _mx._near_seizure_file_filter(onsets, C.LOOKBACK_SEC,
                                             _pcfg.PERIICTAL_PREFILTER_SLACK_SEC)
        files = [f for f in _list_files(C.ANIMAL, evoked_dir, None) if filt(f)]
    _log(f"extracting 2-50 ms waveforms from {len(files)} files "
         f"(full read -> 1-500 Hz bandpass -> window) ...")
    waves, ts, recs = [], [], []
    tw_keep = None
    for i, fp in enumerate(files):
        if i % 20 == 0:
            _log(f"  file {i}/{len(files)}")
        ft = _full_traces(fp, C.CHANNEL)
        adt = _epoch_abs_dt(fp)
        if ft is None or adt is None:
            continue
        tr_full, t_full, fs = ft
        if tr_full.shape[0] != adt.size:       # order/count must line up
            continue
        tr, tw = _ef.preprocess(tr_full, t_full, fs, _LP_CFG)   # LP500 then 2-50 ms
        waves.append(tr[:, ::DECIM].astype(np.float32))
        ts.append(adt)
        recs.append(np.full(adt.size, _file_dt(fp).timestamp()))
        tw_keep = tw[::DECIM]
    assert waves, "no waveforms extracted"
    W = np.concatenate(waves, 0)
    t = np.concatenate(ts)
    rec = np.concatenate(recs)
    good = np.all(np.isfinite(W), axis=1) & np.isfinite(t)
    W, t, rec = W[good], t[good], rec[good]
    mean_wave = W.mean(0)
    res = (W - mean_wave).astype(np.float32)
    _log(f"{res.shape[0]} epochs x {res.shape[1]} samples; global mean subtracted")
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, res=res, t=t, rec=rec,
                        time_ms=tw_keep, mean_wave=mean_wave)
    return {"res": res, "t": t, "rec": rec, "time_ms": tw_keep,
            "mean_wave": mean_wave}


def _seizure_join(t: np.ndarray, store) -> pd.DataFrame:
    """Build the meta frame (t_epoch, abs_dt, seizure join) for residual epochs,
    reusing the committed matrix assignment helpers."""
    seiz = _mx.included_seizures(store, C.ANIMAL)
    onsets = np.array([s.onset_epoch for s in seiz], float)
    ceil = np.array([c if c is not None else np.nan for c in
                     _mx.lookback_ceilings(seiz, _pcfg.DEFAULT_POST_ICTAL_BUFFER_SEC,
                                           C.LOOKBACK_SEC)], float)
    order = np.argsort(t)
    ts = t[order]
    idx_pre, tto_pre, keep_pre = _mx._assign_next_onset(ts, onsets, ceil)
    post_ceil = np.full(onsets.size, float(C.LOOKBACK_SEC))
    idx_post, tto_post, keep_post = _mx._assign_prev_onset(ts, onsets, post_ceil)
    # pre takes precedence; else post; else drop (outside any window)
    phase = np.where(keep_pre, "pre", np.where(keep_post, "post", "none"))
    tto = np.where(keep_pre, tto_pre, np.where(keep_post, tto_post, np.nan))
    sidx = np.where(keep_pre, idx_pre, idx_post)
    son = np.where(np.isfinite(sidx), onsets[np.clip(sidx, 0, onsets.size - 1)], np.nan)
    df = pd.DataFrame({"t_epoch": ts, "abs_dt": pd.to_datetime(ts, unit="s"),
                       "time_to_onset_sec": tto, "phase": phase,
                       "seizure_idx": sidx, "seizure_onset_epoch": son})
    return df, order


def fit_residual_states(data: dict, store, *, n_pcs=None, k=None, peri_only=True):
    """PCA on residual waveforms -> k-means states. Returns (StateModel, extras)
    where extras has per-state mean residual waveform + the time axis.
    peri_only=True keeps only peri-ictal epochs (occupancy scope); False keeps
    every epoch (for a gap-free continuous timeline)."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    k = int(k or C.K_STATES)
    res, t = data["res"], data["t"]
    df, order = _seizure_join(t, store)
    X = res[order]
    if peri_only:
        keep = df["phase"].to_numpy() != "none"   # only epochs in a peri-ictal window
        df = df[keep].reset_index(drop=True); X = X[keep]
    if n_pcs is None:                              # #PCs to reach VAR_TARGET var
        full = PCA(random_state=C.SEED).fit(X)
        n_pcs = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_),
                                    C.VAR_TARGET) + 1)
        n_pcs = max(2, min(n_pcs, X.shape[1]))
    else:
        n_pcs = int(n_pcs)
    pca = PCA(n_components=n_pcs, random_state=C.SEED).fit(X)
    emb = pca.transform(X)
    km = KMeans(n_clusters=k, random_state=C.SEED, n_init=10).fit(emb)
    for j in range(n_pcs):
        df[f"pc{j + 1}"] = emb[:, j]
    df["state"] = km.labels_
    state_means = np.array([X[km.labels_ == s].mean(0) if np.any(km.labels_ == s)
                            else np.full(X.shape[1], np.nan) for s in range(k)])
    rms = np.sqrt(np.nanmean(state_means ** 2, axis=1))
    order_rms = np.argsort(-rms)
    labels = {int(s): ("largest-deviation" if s == order_rms[0] else
                       "near-average" if s == order_rms[-1] else "mid-deviation")
              for s in range(k)}
    model = S.StateModel(df=df, features=[], scaler=None, pca=pca, kmeans=km,
                         explained_var=pca.explained_variance_ratio_,
                         profiles=pd.DataFrame(), labels=labels,
                         fit_mask=np.ones(len(df), bool))
    return model, {"state_means": state_means, "time_ms": data["time_ms"],
                   "mean_wave": data["mean_wave"]}


# --------------------------- rendering / orchestration ---------------------- #

def _state_waveforms_fig(extras, labels, out_png):
    """Per-state MEAN RESIDUAL waveform (deviation from the global evoked mean),
    with the global mean waveform shown faint for reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tm, sm, mw = extras["time_ms"], extras["state_means"], extras["mean_wave"]
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=C.BG)
    ax.set_facecolor(C.PANEL)
    ax.axhline(0, color=C.MUTED, lw=0.6)
    for s in range(sm.shape[0]):
        ax.plot(tm, sm[s], color=C.STATE_COLORS[s], lw=2.0,
                label=f"state {s} · {labels.get(s,'')}")
    ax2 = ax.twinx()
    ax2.plot(tm, mw, color=C.MUTED, lw=1.0, ls="--", alpha=0.7)
    ax2.set_ylabel("global mean evoked (µV, dashed)", color=C.MUTED, fontsize=8)
    ax2.tick_params(colors=C.MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(C.MUTED)
    ax.tick_params(colors=C.MUTED, labelsize=8)
    ax.set_xlabel("ms since stim", color=C.TEXT)
    ax.set_ylabel("mean RESIDUAL (epoch − global mean)", color=C.TEXT)
    ax.set_title("residual-branch states: mean residual evoked waveform per state",
                 color=C.TEXT, fontsize=10, loc="left")
    leg = ax.legend(fontsize=7.5, framealpha=0.1, loc="best")
    for t in leg.get_texts():
        t.set_color(C.TEXT)
    import os as _os
    _os.makedirs(_os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=130, facecolor=C.BG,
                                    bbox_inches="tight")
    plt.close(fig)
    return out_png


def run_residual(*, force=False, with_anim=True) -> dict:
    """Full residual-branch run: build residual matrix, fit states, render the
    figure set (+ animations), all into OUT_DIR/residual/."""
    from . import figures as G, occupancy as O, trajectory as TR, timeline as TL
    from src.db.store import Store
    from src.dashboard.data_helpers import load_config
    cfg = load_config(); store = Store(cfg["database"]["path"])
    evoked_dir = cfg["chronic_evoked"]["evoked_output_dir"]
    data = build_residual_matrix(store, evoked_dir, force=force)
    model, extras = fit_residual_states(data, store)
    ev = model.explained_var
    _log(f"{len(model.df)} peri-ictal epochs | PC var%="
         f"{[round(100*x,1) for x in ev]} cum={round(100*ev.sum(),1)}")
    _log(f"state counts: {model.df['state'].value_counts().sort_index().to_dict()}")
    od = C.RESID_WAVE_DIR; os.makedirs(od, exist_ok=True)
    p = lambda n: os.path.join(od, n)
    pre = S.preictal_mask(model.df); base = S.baseline_mask(model.df)
    occ = O.observed(model.df, pre, base)
    out = {}
    out["state_waveforms"] = _state_waveforms_fig(extras, model.labels,
                                                  p("resid_01_state_waveforms.png"))
    # anchor: PCA scatter + occupancy bars
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 4.8), facecolor=C.BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.3)
    G.pca_scatter(fig.add_subplot(gs[0, 0]), model, pre)
    G.occupancy_bars(fig.add_subplot(gs[0, 1]), occ, model)
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · RESIDUAL-waveform PCA state space "
                 f"({C.N_PCS} PCs, {100*ev.sum():.0f}% var)", color=C.TEXT, fontsize=12)
    fig.savefig(p("resid_00_anchor.png"), dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig); out["anchor"] = p("resid_00_anchor.png")
    lead = F.lead_onsets(store); allon = F.scoped_onsets(store)
    null = O.circular_shift_null(model.df, lead, n_surr=2000, seed=0)
    out["null_compare"] = G.occupancy_null_compare(
        null, p("resid_05_occupancy_null_compare.png"), baseline=occ["baseline"])
    out["traj_lead"] = G.per_seizure_trajectory_fig(
        TR.per_seizure_trajectory(model.df, lead), p("resid_06_traj_lead.png"),
        title_suffix=" — lead (residual branch)")
    out["traj_all"] = G.per_seizure_trajectory_fig(
        TR.per_seizure_trajectory(model.df, allon), p("resid_06_traj_all.png"),
        title_suffix=" — all (residual branch)")
    out["timeline"] = TL.render_timeline(model.df, lead, p("resid_07_timeline.png"))
    for j in range(C.K_STATES):
        _log(f"  state {j} [{model.labels[j]}]: obs={null['observed'][j]:.1f}% "
             f"null med={null['median'][j]:.1f}% p={null['p'][j]:.2f}")
    if with_anim:
        from . import anim as A
        out["anim_lead"] = A.animate_set(model, lead, A.seizure_labels(lead),
                                         os.path.join(od, "anim", "resid_traj_lead"),
                                         step_min=2.0, progress=lambda i, n: None)
    for k_, v in out.items():
        _log(f"{k_} -> {v}")
    return {"model": model, "extras": extras, "out": out}


_RM_CACHE = C.CACHE_DIR + "/BCH111_resid_metrics.pkl"


def residual_metric_matrix(store, db, *, force=False) -> pd.DataFrame:
    """The SAME expanded metric set as the feature branch, but computed on the
    RESIDUAL waveform (epoch - global mean): compute_all -> Z_ss-detrend -> CSD,
    with the seizure join. So the two branches differ only by input (evoked vs
    residual), enabling a like-for-like comparison. Cached."""
    if not force and os.path.exists(_RM_CACHE):
        _log(f"loading cached residual-metrics: {_RM_CACHE}")
        return pd.read_pickle(_RM_CACHE)
    from src.utils import evoked_features as ef
    from src.db.store import Store  # noqa: F401 (store passed in)
    from src.dashboard.data_helpers import load_config
    ed = load_config()["chronic_evoked"]["evoked_output_dir"]
    data = build_residual_matrix(store, ed)
    res, t, tm = data["res"], data["t"], data["time_ms"]
    fs = 1000.0 / float(np.mean(np.diff(tm)))
    _log(f"compute_all on {res.shape[0]} residual epochs (~12 min) ...")
    feats = ef.compute_all(res, tm, fs, expensive=False, include_wavelet=True)
    df = pd.DataFrame({"t_epoch": t,
                       "abs_dt": pd.to_datetime(t, unit="s")})
    for c in set(C.FEATURES) | {C.CSD_PRIMARY}:
        if c in feats:
            df[c] = np.asarray(feats[c], float)
    df = df.sort_values("t_epoch").reset_index(drop=True)
    jdf, _order = _seizure_join(t, store)              # jdf sorted ascending by t
    for col in ("time_to_onset_sec", "phase", "seizure_idx",
                "seizure_onset_epoch"):
        df[col] = jdf[col].to_numpy()
    _log("attaching Z_ss, detrending, adding CSD ...")
    df = F._attach_zss(df, db)
    df = F._detrend_features(df)
    df = F._add_csd(df)
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    df.to_pickle(_RM_CACHE)
    _log(f"cached -> {_RM_CACHE}  ({len(df)} rows)")
    return df


def run_residual_metrics(*, force=False, n_surr=500) -> dict:
    """Residual-branch analysis on the SAME metric set as the feature branch:
    fit states, render anchor + occupancy-null, run the classifier-vs-null."""
    from . import figures as G, occupancy as O, classify as CL
    from src.db.store import Store
    from src.dashboard.data_helpers import load_config
    cfg = load_config(); store = Store(cfg["database"]["path"]); db = cfg["database"]["path"]
    df = F._with_phfo(residual_metric_matrix(store, db, force=force))
    model = S.fit_states(df)
    lead = F.lead_onsets(store)
    pre = S.preictal_mask(model.df); base = S.baseline_mask(model.df)
    occ = O.observed(model.df, pre, base)
    od = C.RESID_METRIC_DIR; os.makedirs(od, exist_ok=True)
    p = lambda n: os.path.join(od, n)
    G.anchor_figure(model, occ, pre, p("residm_00_anchor.png"))
    null = O.circular_shift_null(model.df, lead, n_surr=n_surr)
    G.occupancy_null_compare(null, p("residm_05_occupancy_null_compare.png"),
                             baseline=occ["baseline"])
    clf = CL.classify_null(df, lead, n_surr=n_surr)
    G.classifier_null_fig(clf, p("residm_08_classifier_null.png"))
    ev = model.explained_var
    _log(f"PC var%={[round(100*x,1) for x in ev]} cum={round(100*ev.sum(),1)}")
    _log(f"classifier AUC={clf['auc']:.3f} p={clf['p']:.3f} "
         f"(null med {clf['null_median']:.3f}, 95th {clf['null_hi']:.3f})")
    for j in range(C.K_STATES):
        _log(f"  occ state {j}: obs={null['observed'][j]:.1f}% p={null['p'][j]:.2f}")
    return {"model": model, "occ": occ, "null": null, "clf": clf}


def _states_by_time(model_df, t_w):
    """State label for each waveform epoch (time t_w) by nearest feature-epoch."""
    tf = pd.to_numeric(model_df["t_epoch"], errors="coerce").to_numpy(float)
    sf = model_df["state"].to_numpy()
    order = np.argsort(tf); tf, sf = tf[order], sf[order]
    pos = np.clip(np.searchsorted(tf, t_w), 1, tf.size - 1)
    near = np.where(np.abs(tf[pos] - t_w) < np.abs(tf[pos - 1] - t_w), pos, pos - 1)
    st = sf[near]
    st[np.abs(tf[near] - t_w) > 5.0] = -1                  # no match within 5 s
    return st


def evoked_and_states(store, ed, db):
    """Reconstruct raw 2-50 ms (stim-blanked) evoked waveforms + the feature-branch
    state per waveform epoch + the time axis. Shared by the cluster-traces figure
    and the animation LFP panel."""
    from . import features as F, states as S
    data = build_residual_matrix(store, ed, full_record=True)
    evoked = data["res"] + data["mean_wave"]              # residual + global mean
    model = S.fit_states(F.build_feature_matrix(store, ed, db))
    st = _states_by_time(model.df, data["t"])
    return evoked, data["time_ms"], st, data["t"], model


def run_cluster_traces(*, n_traces=25, seed=0) -> str:
    """Per-state evoked waveform (mean + n_traces random traces) for the feature-
    branch states, on the full record."""
    from . import figures as G
    from src.db.store import Store
    from src.dashboard.data_helpers import load_config
    cfg = load_config(); store = Store(cfg["database"]["path"])
    ed = cfg["chronic_evoked"]["evoked_output_dir"]; db = cfg["database"]["path"]
    evoked, tm, st, _t, _m = evoked_and_states(store, ed, db)
    out = os.path.join(C.EVOKED_DIR, "12_cluster_traces.png")
    G.cluster_traces_fig(evoked, tm, st, out, n_traces=n_traces, seed=seed)
    _log(f"cluster traces -> {out}  (state counts "
         f"{dict(zip(*np.unique(st[st>=0], return_counts=True)))})")
    return out


def run_anim_lfp(*, which="lead", step_min=2.0, fps=12, with_env=True,
                 frames=True) -> dict:
    """Full trajectory animation: PCA comet + per-epoch evoked LFP + CONTINUOUS
    LFP envelope (−120→0 min, seizure at 0), all time-locked; + per-frame PNGs and
    a step-through HTML viewer. which='lead' (12 raw reads) or 'all' (43, slow)."""
    from . import anim as A, continuous_lfp as CL
    from src.db.store import Store
    from src.dashboard.data_helpers import load_config
    cfg = load_config(); store = Store(cfg["database"]["path"])
    ed = cfg["chronic_evoked"]["evoked_output_dir"]; db = cfg["database"]["path"]
    evoked, tm, _st, t_w, model = evoked_and_states(store, ed, db)
    onsets = F.lead_onsets(store) if which == "lead" else F.scoped_onsets(store)
    lfp_env = None
    if with_env:
        envd = CL.build_lfp_envelopes(store, onsets, pre_h=2.0, post_h=0.35)
        lfp_env = [envd.get(CL._key(o)) for o in onsets]
    adir = os.path.join(C.EVOKED_DIR, "anim")
    fdir = os.path.join(adir, f"{which}_frames") if frames else None
    pr = lambda i, n: print(f"  frame {i}/{n}", flush=True) if i % 80 == 0 else None
    _log(f"rendering {which} full animation ({onsets.size} seizures) ...")
    out = A.animate_set(model, onsets, A.seizure_labels(onsets),
                        os.path.join(adir, f"preictal_traj_{which}_full"),
                        step_min=step_min, fps=fps, waveforms=evoked, wave_t=t_w,
                        wave_time_ms=tm, lfp_env=lfp_env, frames_dir=fdir,
                        progress=pr)
    _log(f"done: {out}")
    return out


def run_horizons(bins_min=(30.0, 10.0, 5.0, 1.0, 0.5)) -> dict:
    """Re-render the residual per-seizure trajectory (lead + all) and the state
    timeline at several dominant-state bin widths. Finer bins surface the rare
    deviation states (gold=2 ~1%, red=3 ~4%) that a 10-min plurality washes out
    (e.g. the post-ictal red state in the minutes just after onset)."""
    from . import figures as G, trajectory as TR, timeline as TL
    from src.db.store import Store
    from src.dashboard.data_helpers import load_config
    cfg = load_config(); store = Store(cfg["database"]["path"])
    data = build_residual_matrix(store, cfg["chronic_evoked"]["evoked_output_dir"])
    model, _ = fit_residual_states(data, store)
    lead = F.lead_onsets(store); allon = F.scoped_onsets(store)
    od = os.path.join(C.RESID_WAVE_DIR, "horizons"); os.makedirs(od, exist_ok=True)
    out = {}
    for b in bins_min:
        tag = f"{int(b)}min" if b >= 1 else f"{int(round(b * 60))}s"
        out[f"traj_lead_{tag}"] = G.per_seizure_trajectory_fig(
            TR.per_seizure_trajectory(model.df, lead, bin_min=b),
            os.path.join(od, f"resid_traj_lead_{tag}.png"),
            title_suffix=f" — lead (residual, {tag} bins)")
        out[f"traj_all_{tag}"] = G.per_seizure_trajectory_fig(
            TR.per_seizure_trajectory(model.df, allon, bin_min=b),
            os.path.join(od, f"resid_traj_all_{tag}.png"),
            title_suffix=f" — all (residual, {tag} bins)")
        out[f"timeline_{tag}"] = TL.render_timeline(
            model.df, lead, os.path.join(od, f"resid_timeline_{tag}.png"),
            bin_sec=b * 60.0)
        _log(f"horizon {tag}: rendered trajectory (lead/all) + timeline")
    for kk, vv in out.items():
        _log(f"{kk} -> {vv}")
    return out


def run_full_timeline(bins_min=(30.0, 10.0, 5.0, 1.0), *, force=False) -> dict:
    """Continuous FULL-RECORD state timeline (no peri-ictal gaps): fit states on
    every in-scope-date epoch (not just near-seizure ones) and render the
    24h/2d/1wk timeline at several bin widths."""
    from . import timeline as TL
    from src.db.store import Store
    from src.dashboard.data_helpers import load_config
    cfg = load_config(); store = Store(cfg["database"]["path"])
    ed = cfg["chronic_evoked"]["evoked_output_dir"]
    data = build_residual_matrix(store, ed, full_record=True, force=force)
    model, _ = fit_residual_states(data, store, peri_only=False)
    _log(f"{len(model.df)} full-record epochs; states "
         f"{model.df['state'].value_counts().sort_index().to_dict()}")
    lead = F.lead_onsets(store)
    od = os.path.join(C.RESID_WAVE_DIR, "full_record"); os.makedirs(od, exist_ok=True)
    out = {}
    for b in bins_min:
        tag = f"{int(b)}min" if b >= 1 else f"{int(round(b * 60))}s"
        out[tag] = TL.render_timeline(
            model.df, lead, os.path.join(od, f"resid_full_timeline_{tag}.png"),
            bin_sec=b * 60.0)
        _log(f"full-record timeline {tag} -> {out[tag]}")
    return out


if __name__ == "__main__":
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    if "--anim-lfp" in sys.argv:
        run_anim_lfp(which="all" if "--all" in sys.argv else "lead")
    elif "--cluster-traces" in sys.argv:
        run_cluster_traces()
    elif "--metrics" in sys.argv:
        run_residual_metrics(force="--force" in sys.argv)
    elif "--full-timeline" in sys.argv:
        run_full_timeline(force="--force" in sys.argv)
    elif "--horizons" in sys.argv:
        run_horizons()
    else:
        run_residual(force="--force" in sys.argv,
                     with_anim="--no-anim" not in sys.argv)
