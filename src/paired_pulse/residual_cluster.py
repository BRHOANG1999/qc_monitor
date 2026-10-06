"""Residual-waveform clustering branch for the paired-pulse analysis.

Represent each paired-pulse epoch by its **S2 − S1 residual waveform** (the 3-45 ms,
≤500 Hz, baseline-subtracted difference), run PCA → k-means on those residual vectors,
then reproduce the SAME pre-ictal clustering analysis as the preictal_biomarker
``evoked_branch`` (occupancy + circular-shift null, per-seizure state trajectory, state
timeline, P(seizure | state) + null, state-risk shift null, state-1 occurrence, cluster
traces, cluster-validity). The whole analysis backend
(``preictal_biomarker.{figures,occupancy,trajectory,timeline,states,validity}``) is
representation-agnostic (it consumes ``df.t_epoch`` / ``df.state`` / ``df.pcN`` + onset
arrays), so it is reused unchanged.

Two deliberate differences from a naive copy, both important here:
 * **Full-record fit** -- PCA/k-means are fit on EVERY paired-pulse epoch, not just
   near-seizure ones (state-space analyses must use all evoked responses).
 * **PP-window onsets via ``isi``** -- the preictal_biomarker ``scoped_onsets`` are scoped
   to 2026-09-13..2026-10-02 and would EXCLUDE the paired-pulse window, so onsets are
   taken straight from ``isi.scored_seizures`` / ``leading_seizures``.

Output: ``data/BCH111_paired_pulse_lp500/residual_cluster/``. CLI: ``--rescluster``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as C, data as D
from src.preictal import isi as _isi
from src.preictal_biomarker import (config as PBC, states as S, figures as G,
                                     occupancy as O, trajectory as TR, timeline as TL,
                                     validity as V)
from src.preictal_biomarker.residual import _seizure_join

DECIM = 4                                   # keep every 4th sample (~5 kHz)
_CACHE = os.path.join(C.CACHE_DIR, "BCH111_pp_residual_matrix.npz")
OUT = os.path.join(C.OUT_DIR, "residual_cluster")


def _log(m):
    print(f"[paired_pulse.rescluster] {m}", flush=True)


def build_residual_matrix(store, evoked_dir, *, force=False) -> dict:
    """Per-epoch S2 − S1 residual waveform across the FULL paired-pulse record. Returns
    dict(res[N,S], t[N], rec[N], time_ms[S], mean_wave[S]); res = each epoch's S2 − S1
    residual MINUS the global-mean S2 − S1 (deviation). Cached to npz."""
    if not force and os.path.exists(_CACHE):
        _log(f"loading cached residual matrix: {_CACHE}")
        z = np.load(_CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    files = D.list_pp_files(evoked_dir)
    _log(f"extracting S2 − S1 residual waveforms from {len(files)} files ...")
    waves, ts, recs, twin = [], [], [], None
    for i, (fp, _d) in enumerate(files):
        if i % 20 == 0:
            _log(f"  file {i}/{len(files)}")
        r = D._read_file(fp, C.CHANNEL)
        if r is None:
            continue
        lfp, stim, t, fs, st = r
        s1_on, s2_on, paired = D._pulse_onsets(stim, t)
        if paired.sum() < 5:
            continue
        lfp_f = D._filter(lfp, fs)
        s1w, tw = D._window_array(lfp_f, t, s1_on, fs)
        s2w, _ = D._window_array(lfp_f, t, s2_on, fs)
        resid = (s2w - s1w)[paired]
        base = D._file_dt(fp).timestamp()
        te = base + st if (st is not None and st.size == lfp.shape[0]) else \
            np.full(lfp.shape[0], base)
        te = te[paired]
        waves.append(resid[:, ::DECIM].astype(np.float32))
        ts.append(te)
        recs.append(np.full(te.size, base))
        twin = tw[::DECIM]
    assert waves, "no paired-pulse epochs found"
    W = np.concatenate(waves, 0)
    t = np.concatenate(ts)
    rec = np.concatenate(recs)
    good = np.all(np.isfinite(W), axis=1) & np.isfinite(t)
    W, t, rec = W[good], t[good], rec[good]
    mean_wave = W.mean(0)
    res = (W - mean_wave).astype(np.float32)
    _log(f"{res.shape[0]} epochs x {res.shape[1]} samples; global-mean S2−S1 subtracted")
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    np.savez_compressed(_CACHE, res=res, t=t, rec=rec, time_ms=twin, mean_wave=mean_wave)
    return {"res": res, "t": t, "rec": rec, "time_ms": twin, "mean_wave": mean_wave}


def fit_states(data: dict, store, *, peri_only=False, n_pcs=None, k=None):
    """PCA on S2 − S1 residual waveforms → k-means states (full-record fit by default).
    Returns (StateModel, extras) with per-state mean residual + the aligned X matrix."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    k = int(k or PBC.K_STATES)
    res, t = data["res"], data["t"]
    df, order = _seizure_join(t, store)
    X = res[order]
    if peri_only:
        keep = df["phase"].to_numpy() != "none"
        df = df[keep].reset_index(drop=True); X = X[keep]
    if n_pcs is None:
        full = PCA(random_state=PBC.SEED).fit(X)
        n_pcs = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_),
                                    PBC.VAR_TARGET) + 1)
        n_pcs = max(2, min(n_pcs, X.shape[1]))
    else:
        n_pcs = int(n_pcs)
    pca = PCA(n_components=n_pcs, random_state=PBC.SEED).fit(X)
    emb = pca.transform(X)
    km = KMeans(n_clusters=k, random_state=PBC.SEED, n_init=10).fit(emb)
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
                   "mean_wave": data["mean_wave"], "X": X}


def _state_waveforms_fig(extras, labels, out_png):
    """Per-state MEAN S2 − S1 residual (deviation from the global-mean residual), with the
    global-mean S2 − S1 shown faint for reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tm, sm, mw = extras["time_ms"], extras["state_means"], extras["mean_wave"]
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=C.BG)
    ax.set_facecolor(C.PANEL)
    ax.axhline(0, color=C.MUTED, lw=0.6)
    for s in range(sm.shape[0]):
        ax.plot(tm, sm[s], color=PBC.STATE_COLORS[s], lw=2.0,
                label=f"state {s} · {labels.get(s, '')}")
    ax2 = ax.twinx()
    ax2.plot(tm, mw, color=C.MUTED, lw=1.0, ls="--", alpha=0.7)
    ax2.set_ylabel("global mean S2 − S1 (dashed)", color=C.MUTED, fontsize=8)
    ax2.tick_params(colors=C.MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(C.MUTED)
    ax.tick_params(colors=C.MUTED, labelsize=8)
    ax.set_xlabel("ms from pulse onset", color=C.TEXT)
    ax.set_ylabel("mean residual (epoch S2−S1 − global mean)", color=C.TEXT)
    ax.set_title("residual-cluster states: mean S2 − S1 residual per state",
                 color=C.TEXT, fontsize=10, loc="left")
    leg = ax.legend(fontsize=7.5, framealpha=0.1, loc="best")
    for tt in leg.get_texts():
        tt.set_color(C.TEXT)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _pp_onsets(store, lo, hi):
    """PP-window lead & all onsets from isi (NOT the Sep-scoped preictal_biomarker ones)."""
    sz = _isi.scored_seizures(store, C.ANIMAL)
    allon = np.sort(np.array([s.onset_epoch for s in sz], float))
    leadon = np.sort(np.array([s.onset_epoch for s in
                               _isi.leading_seizures(sz, 6 * 3600.0)], float))
    return (leadon[(leadon >= lo) & (leadon <= hi)],
            allon[(allon >= lo) & (allon <= hi)])


def run(*, force=False, n_surr=2000) -> dict:
    """Full residual-cluster run into residual_cluster/ (fit on the full record)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .run import _ctx
    store, ed = _ctx()
    data = build_residual_matrix(store, ed, force=force)
    model, extras = fit_states(data, store, peri_only=False)
    ev = model.explained_var
    _log(f"{len(model.df)} full-record epochs | PCs={ev.size} var%="
         f"{[round(100 * x, 1) for x in ev[:6]]} cum={round(100 * ev.sum(), 1)}")
    _log(f"state counts: {model.df['state'].value_counts().sort_index().to_dict()}")
    t = model.df["t_epoch"].to_numpy(float)
    lo, hi = t.min(), t.max()
    lead, allon = _pp_onsets(store, lo, hi)
    _log(f"PP-window onsets: lead={lead.size} all={allon.size}")
    od = OUT
    os.makedirs(od, exist_ok=True)
    p = lambda n: os.path.join(od, n)                        # noqa: E731
    pre = S.preictal_mask(model.df); base = S.baseline_mask(model.df)
    occ = O.observed(model.df, pre, base)
    out = {}

    # 01 per-state mean residual waveform
    out["state_waveforms"] = _state_waveforms_fig(extras, model.labels,
                                                  p("rc_01_state_waveforms.png"))
    # 00 anchor: PCA scatter + occupancy bars
    fig = plt.figure(figsize=(13, 4.8), facecolor=C.BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.3)
    G.pca_scatter(fig.add_subplot(gs[0, 0]), model, pre)
    G.occupancy_bars(fig.add_subplot(gs[0, 1]), occ, model)
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · S2−S1 RESIDUAL PCA state space "
                 f"({ev.size} PCs, {100 * ev.sum():.0f}% var) — pre-ictal highlighted",
                 color=C.TEXT, fontsize=12)
    fig.savefig(p("rc_00_anchor.png"), dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig); out["anchor"] = p("rc_00_anchor.png")

    # 11 cluster validity (discrete states vs continuum) on the residual space
    try:
        cols = [f"s{i}" for i in range(data["res"].shape[1])]
        vdf = pd.DataFrame(data["res"], columns=cols)
        vres = V.cluster_validity(vdf, features=cols)
        out["validity"] = G.cluster_validity_fig(
            vres, p("rc_11_cluster_validity.png"),
            title_suffix=" — S2−S1 residual space")
        _log(f"cluster validity: discrete={vres['discrete']} "
             f"(PCs@95%={vres['ncomp']}, bic_min_k={vres.get('bic_min_k')})")
    except Exception as e:                                   # noqa: BLE001
        _log(f"cluster validity skipped: {type(e).__name__}: {e}")

    # 12 cluster traces (per-state example raw S2−S1 residuals)
    raw = extras["X"] + extras["mean_wave"]
    out["cluster_traces"] = G.cluster_traces_fig(
        raw, extras["time_ms"], model.df["state"].to_numpy(),
        p("rc_12_cluster_traces.png"), n_traces=25,
        title_suffix=" — S2−S1 residual")

    # 05 occupancy vs circular-shift null (lead)
    null = O.circular_shift_null(model.df, lead, n_surr=n_surr, seed=0)
    out["null_compare"] = G.occupancy_null_compare(
        null, p("rc_05_occupancy_null_compare.png"), baseline=occ["baseline"])
    for j in range(PBC.K_STATES):
        _log(f"  occ state {j} [{model.labels[j]}]: obs={null['observed'][j]:.1f}% "
             f"null med={null['median'][j]:.1f}% p={null['p'][j]:.2f}")

    # 06 per-seizure state trajectory (lead + all)
    out["traj_lead"] = G.per_seizure_trajectory_fig(
        TR.per_seizure_trajectory(model.df, lead), p("rc_06_traj_lead.png"),
        title_suffix=" — lead (S2−S1 residual states)")
    out["traj_all"] = G.per_seizure_trajectory_fig(
        TR.per_seizure_trajectory(model.df, allon), p("rc_06_traj_all.png"),
        title_suffix=" — all (S2−S1 residual states)")
    # 07 state timeline
    out["timeline"] = TL.render_timeline(model.df, lead, p("rc_07_timeline.png"))

    # 09 P(seizure | state) + shift null, and 10 state-risk shift null (lead + all)
    for nm, ons in (("lead", lead), ("all", allon)):
        out[f"prob_{nm}"] = G.seizure_prob_fig(
            O.seizure_prob_by_state(model.df, ons), p(f"rc_09_seizure_prob_{nm}.png"),
            title_suffix=f" — {nm} (S2−S1 residual)")
        out[f"prob_null_{nm}"] = G.seizure_prob_null_fig(
            O.seizure_prob_shift_null(model.df, ons, n_surr=n_surr),
            p(f"rc_09_seizure_prob_null_{nm}.png"), labels=model.labels,
            title_suffix=f" — {nm} (S2−S1 residual)")
        out[f"risk_null_{nm}"] = G.state_risk_bin_null_fig(
            O.state_risk_bin_null(model.df, ons, n_surr=n_surr),
            p(f"rc_10_state_risk_shift_null_{nm}.png"), labels=model.labels,
            title_suffix=f" — {nm} (S2−S1 residual)")

    # 13 state-1 occurrence (trajectory + timeline, lead + all)
    for nm, ons in (("lead", lead), ("all", allon)):
        out[f"state1_traj_{nm}"] = G.state_fraction_trajectory_fig(
            TR.per_seizure_state_fraction(model.df, ons, target=1),
            p(f"rc_13_state1_occurrence_traj_{nm}.png"))
    out["state1_timeline"] = TL.render_state_fraction_timeline(
        model.df, lead, p("rc_13_state1_occurrence_timeline.png"), target=1)

    for k_, v in out.items():
        _log(f"{k_} -> {v}")
    return {"model": model, "extras": extras, "occ": occ, "null": null, "out": out}
