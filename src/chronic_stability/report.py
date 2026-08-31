"""Orchestration for the BCH111 chronic-stim stability map.

``run()`` executes the full read-only pipeline: detect the stable Rₐ run per
configuration, build a fixed artifact template over the stable window, compute
per-trial metrics across the whole record, count seizures, run distributional
comparisons + collinearity, render figures, and write CSV + a markdown report.
Mapping only -- no experimental recommendations.
"""

from __future__ import annotations

import json
import os

import numpy as np

from src.preictal.isi import parse_chunk_datetime, scored_seizures
from src.utils import evoked_output as eo
from . import compare, config_timeline as ct, maintenance, metrics as M
from . import ra_flatrun as fr, render

_ART_LO, _ART_HI = -1.0, 2.0        # artifact window (ms), operator-confirmed.
_METRIC_KEYS = ("amplitude", "corr", "nrmse")


def _log(msg: str) -> None:
    print(f"[chronic_stability] {msg}", flush=True)


def _ra_series(store, animal: str, channel: str):
    """(t_epoch, r_kohm) for one record channel's dominant-charge Rₐ series."""
    ser = store.impedance_series_by_channel()
    rows = ser.get((animal, channel), [])
    te, r = [], []
    for row in rows:
        dt = parse_chunk_datetime(row["chunk_datetime"])
        if dt is not None and row["access_r_kohm"] is not None:
            te.append(dt.timestamp())
            r.append(float(row["access_r_kohm"]))
    return np.asarray(te, float), np.asarray(r, float)


def _detect_stable(store, animal: str, intervals: list, params: dict):
    """Detect the longest flat Rₐ run per configuration; return the global
    longest (the stable window) with its config index and per-config series."""
    ra = {}
    best = None
    for k, iv in enumerate(intervals):
        te, r = _ra_series(store, animal, iv["record_channel"])
        ra[k] = (te, r)
        if r.size < 8:
            continue
        th = (te - te[0]) / 3600.0
        run = fr.detect_longest_flat_run(th, r, te, **params)
        if run.get("found") and (best is None
                                 or run["duration_h"] > best[1]["duration_h"]):
            best = (k, run)
    return best, ra


def run(store, animal: str, evoked_dir: str, out_dir: str, *,
        flat_params: dict | None = None, n_boot: int = 2000) -> dict:
    """Execute the full mapping pipeline and write all artifacts to *out_dir*."""
    assert animal and evoked_dir and out_dir, "animal/evoked_dir/out_dir req'd"
    os.makedirs(out_dir, exist_ok=True)
    flat_params = flat_params or {}
    _log(f"enumerating configurations for {animal} ...")
    sessions = ct.bch111_sessions(store, animal)
    intervals = ct.config_intervals(sessions)
    events = ct.detect_config_events(sessions)
    assert intervals, "no configurations found"
    _log(f"{len(intervals)} configuration(s); detecting stable Rₐ run ...")
    best, ra = _detect_stable(store, animal, intervals, flat_params)
    assert best is not None, "no flat Rₐ run detected in any configuration"
    stable_cfg, run_ = best
    files = sorted(f for f in eo.list_evoked_files(evoked_dir)
                   if animal in os.path.basename(f))
    _log(f"{len(files)} evoked files; building template over stable window ...")
    tmpl = _template(files, animal, run_)
    _log("streaming per-trial metrics across the whole record ...")
    mets = M.stream_metrics(files, animal, _ART_LO, _ART_HI, tmpl)
    seiz = scored_seizures(store, animal)
    sz_ep = np.asarray([s.onset_epoch for s in seiz], float)
    _log("sampling raw waveforms for trace figures ...")
    sample = M.gather_sample_trials(files, animal, _ART_LO, _ART_HI,
                                    max_trials=8000)
    ctx = _assemble(store, animal, intervals, events, ra, stable_cfg, run_,
                    mets, seiz, sz_ep, tmpl, n_boot)
    ctx["sample"] = sample
    ctx["flat_params"] = flat_params
    ctx["evoked_files"] = files
    ctx["maintenance"] = _load_maintenance(mets)
    ctx["fig_lines"] = maintenance.figure_lines(ctx["maintenance"])
    _write_all(out_dir, animal, ctx)
    _log(f"done -> {out_dir}")
    return ctx


def _load_maintenance(mets) -> dict:
    """Battery / cage events over the record span (degrades cleanly if the
    maintenance sheets are unreachable)."""
    from src.dashboard.data_helpers import load_config
    lo, hi = float(mets["epoch"].min()), float(mets["epoch"].max())
    ev = maintenance.load_events(load_config(), lo, hi)
    if ev["available"]:
        _log(f"maintenance: {len(ev['software'])} software stop/start events, "
             f"{len(ev['battery_human'])} battery + {len(ev['cage_human'])} "
             f"cage human-reported")
    else:
        _log(f"maintenance overlay unavailable ({ev['error']})")
    return ev


def _template(files, animal, run_):
    """Build the fixed artifact template from stable-window segments."""
    seg = M.collect_segments(files, animal, _ART_LO, _ART_HI,
                             run_["epoch_start"], run_["epoch_end"])
    assert seg["seg"].shape[0] >= 50, "too few stable-window trials for template"
    return M.build_template(seg["seg"])


def _assemble(store, animal, intervals, events, ra, stable_cfg, run_, mets,
              seiz, sz_ep, tmpl, n_boot) -> dict:
    """Assign trials to configs, count seizures, run comparisons + collinearity."""
    cfg_idx = ct.assign_epochs_to_configs(mets["epoch"], intervals)
    in_stable = (mets["epoch"] >= run_["epoch_start"]) \
        & (mets["epoch"] <= run_["epoch_end"])
    per_cfg = _per_config_summary(intervals, cfg_idx, mets, sz_ep, ra)
    run_["seizures_in_window"] = int(((sz_ep >= run_["epoch_start"])
                                      & (sz_ep <= run_["epoch_end"])).sum())
    comps = _comparisons(intervals, stable_cfg, cfg_idx, in_stable, mets, n_boot)
    collin = _collinearity(intervals, stable_cfg, cfg_idx, in_stable, mets)
    return {"animal": animal, "intervals": intervals, "events": events,
            "stable_cfg": stable_cfg, "run": run_,
            "per_config": per_cfg, "comparisons": comps, "collinearity": collin,
            "metrics": mets, "cfg_idx": cfg_idx, "in_stable": in_stable,
            "ra": ra, "seizure_epochs": sz_ep,
            "n_seizures_total": len(seiz), "template": tmpl}


def _per_config_summary(intervals, cfg_idx, mets, sz_ep, ra) -> list:
    """Per configuration: span, trials, seizures, and metric medians + Rₐ."""
    out = []
    for k, iv in enumerate(intervals):
        m = cfg_idx == k
        ep = mets["epoch"][m]
        hi = iv["end_epoch"] if np.isfinite(iv["end_epoch"]) \
            else (ep.max() if ep.size else iv["start_epoch"])
        nsz = int(((sz_ep >= iv["start_epoch"]) & (sz_ep < hi)).sum())
        te, r = ra.get(k, (np.empty(0), np.empty(0)))
        out.append({**{key: iv[key] for key in
                       ("config_label", "stim_site", "record_site",
                        "start_epoch", "end_epoch", "modal_charge_nC")},
                    "n_trials": int(m.sum()), "n_seizures": nsz,
                    "duration_h": float((hi - iv["start_epoch"]) / 3600.0),
                    "amp_median": _med(mets["amplitude"][m]),
                    "corr_median": _med(mets["corr"][m]),
                    "nrmse_median": _med(mets["nrmse"][m]),
                    "polarity_switch": M.switch_count(mets["polarity"][m]),
                    "ra_median": _med(r), "ra_n": int(r.size)})
    return out


def _med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _cfg_name(intervals, k: int) -> str:
    """Readable configuration name, e.g. 'Config B (SLM record, SR stim)'."""
    if k < 0 or k >= len(intervals):
        return "pre-config (record-only)"
    iv = intervals[k]
    return (f"Config {chr(65 + k)} ({iv['record_site']} record, "
            f"{iv['stim_site']} stim)")


def _comparisons(intervals, stable_cfg, cfg_idx, in_stable, mets, n_boot) -> list:
    """Stable-vs-own-config, stable-config-vs-other, stable-window-vs-other."""
    sc = cfg_idx == stable_cfg
    others = [k for k in range(len(intervals)) if k != stable_cfg]
    sname = _cfg_name(intervals, stable_cfg)
    out = []
    for key in _METRIC_KEYS:
        v = mets[key]
        out.append(compare.compare_two_groups(
            v[in_stable], v[sc & ~in_stable],
            name=f"{key}: stable window vs {sname} outside the stable window",
            n_boot=n_boot))
        for o in others:
            oc = cfg_idx == o
            oname = _cfg_name(intervals, o)
            out.append(compare.compare_two_groups(
                v[sc], v[oc], name=f"{key}: {sname} vs {oname}",
                n_boot=n_boot))
            out.append(compare.compare_two_groups(
                v[in_stable], v[oc],
                name=f"{key}: stable window vs {oname}", n_boot=n_boot))
    out.append(compare.compare_polarity(
        mets["polarity"][sc], mets["polarity"][cfg_idx == others[0]],
        name=f"polarity: {sname} vs {_cfg_name(intervals, others[0])}")
        if others else {})
    return [c for c in out if c]


def _collinearity(intervals, stable_cfg, cfg_idx, in_stable, mets) -> dict:
    """Spearman age-vs-metric per config + the stable window's age band."""
    out = {"per_config": {}, "stable_age_band": {}}
    for k, iv in enumerate(intervals):
        m = cfg_idx == k
        if m.sum() < 3:
            continue
        out["per_config"][iv["config_label"]] = {
            key: compare.age_confound(mets["epoch"][m], mets[key][m],
                                      iv["start_epoch"]) for key in _METRIC_KEYS}
    sc = cfg_idx == stable_cfg
    out["stable_age_band"] = compare.stable_age_band(
        mets["epoch"][in_stable], mets["epoch"][sc],
        intervals[stable_cfg]["start_epoch"])
    return out


# --------------------------- output writers --------------------------------- #

def _write_all(out_dir, animal, ctx) -> None:
    """Write all CSV / PNG / markdown artifacts."""
    _write_csvs(out_dir, animal, ctx)
    _write_figures(out_dir, animal, ctx)
    _write_markdown(out_dir, animal, ctx)


def _write_csvs(out_dir, animal, ctx) -> None:
    import pandas as pd
    m = ctx["metrics"]
    df = pd.DataFrame({
        "epoch": m["epoch"], "channel": m["channel"],
        "config_idx": ctx["cfg_idx"], "in_stable_window": ctx["in_stable"],
        "polarity": m["polarity"], "polarity_abs": m["polarity_abs"],
        "amplitude": m["amplitude"], "corr": m["corr"], "nrmse": m["nrmse"]})
    df.to_csv(os.path.join(out_dir, f"{animal}_per_trial_metrics.csv.gz"),
              index=False, compression="gzip")
    pd.DataFrame(ctx["per_config"]).to_csv(
        os.path.join(out_dir, f"{animal}_config_timeline.csv"), index=False)
    pd.DataFrame(ctx["comparisons"]).to_csv(
        os.path.join(out_dir, f"{animal}_comparisons.csv"), index=False)
    with open(os.path.join(out_dir, f"{animal}_config_events.json"), "w") as f:
        json.dump(ctx["events"], f, indent=2, default=str)
    _write_maintenance_csv(out_dir, animal, ctx, pd)


def _write_maintenance_csv(out_dir, animal, ctx, pd) -> None:
    """Software + human-reported battery/cage events as one tidy CSV."""
    mev = ctx.get("maintenance") or {}
    if not mev.get("available"):
        return
    rows = [{**e, "source": "software"} for e in mev["software"]]
    rows += [{**e, "source": "battery_human", "kind": "battery"}
             for e in mev["battery_human"]]
    rows += [{**e, "source": "cage_human", "kind": "cage"}
             for e in mev["cage_human"]]
    if rows:
        pd.DataFrame(rows).sort_values("epoch").to_csv(
            os.path.join(out_dir, f"{animal}_maintenance_events.csv"),
            index=False)


def _write_figures(out_dir, animal, ctx) -> None:
    p = lambda n: os.path.join(out_dir, f"{animal}_{n}")
    rec_t0 = float(ctx["metrics"]["epoch"].min())     # day 0 shared by all figs
    ctx["record_t0"] = rec_t0
    lines = ctx.get("fig_lines")
    render.ra_over_time_fig(ctx["ra"], ctx["intervals"], ctx["run"],
                            ctx["seizure_epochs"], p("ra_over_time.png"),
                            t0=rec_t0, events=lines)
    render.ra_aligned_fig(ctx["ra"], ctx["intervals"], ctx["run"],
                          p("ra_aligned.png"))
    _sensitivity_fig(out_dir, animal, ctx)
    m = ctx["metrics"]
    render.metric_timeseries_fig(
        m["epoch"], m, ctx["ra"], ctx["intervals"], ctx["run"],
        ctx["seizure_epochs"], p("metric_timeseries.png"), events=lines)
    _trace_figs(out_dir, animal, ctx)
    _distribution_figs(out_dir, animal, ctx)


def _sensitivity_fig(out_dir, animal, ctx) -> None:
    """Threshold-robustness sweep on the stable config's Rₐ series."""
    te, r = ctx["ra"][ctx["stable_cfg"]]
    if len(te) < 8:
        return
    th = (te - te[0]) / 3600.0
    grid = [dict(win_h=w, disp_k=d, slope_k=s)
            for w in (18, 24, 36) for d in (3, 4, 5) for s in (4, 6)]
    sweep = fr.sensitivity_sweep(th, r, te, grid)
    default = {"win_h": 24, "disp_k": 4.0, "slope_k": 6.0,
               **{k: v for k, v in ctx["flat_params"].items()
                  if k in ("win_h", "disp_k", "slope_k")}}
    render.sensitivity_fig(sweep, ctx["record_t0"], default,
                           os.path.join(out_dir, f"{animal}_sensitivity.png"))


def _trace_groups(ctx):
    """(label, sample-mask, colour) tuples for the trace figures: pre-config,
    the other config(s), the stable window, and the rest of the stable config."""
    s = ctx["sample"]
    sci = ct.assign_epochs_to_configs(s["epoch"], ctx["intervals"])
    s_in = (s["epoch"] >= ctx["run"]["epoch_start"]) \
        & (s["epoch"] <= ctx["run"]["epoch_end"])
    sc = ctx["stable_cfg"]
    groups = []
    if (sci == -1).any():
        groups.append(("pre-config (record-only)", sci == -1, "#718096"))
    for o in range(len(ctx["intervals"])):
        if o != sc:
            groups.append((_cfg_name(ctx["intervals"], o), sci == o, "#3182ce"))
    groups.append(("stable window", s_in, "#38a169"))
    groups.append((f"Config {chr(65+sc)} outside\nthe stable window",
                   (sci == sc) & ~s_in, "#dd6b20"))
    return groups


def _trace_figs(out_dir, animal, ctx) -> None:
    """Waveform figures: ERP-image, example-trace gallery, mean-trace overlay."""
    s = ctx["sample"]
    if s["seg"].shape[0] < 2:
        return
    groups = _trace_groups(ctx)
    p = lambda n: os.path.join(out_dir, f"{animal}_{n}")
    render.erpimage_fig(s, ctx["intervals"], ctx["run"], p("erpimage.png"),
                        total_trials=int(ctx["metrics"]["epoch"].size))
    render.trace_gallery_fig(s, groups, ctx["template"], p("trace_gallery.png"))
    render.mean_trace_overlay_fig(s, groups, ctx["template"],
                                  p("mean_traces.png"))


def _distribution_figs(out_dir, animal, ctx) -> None:
    m = ctx["metrics"]
    sc = ctx["stable_cfg"]
    ci = ctx["cfg_idx"]
    ins = ctx["in_stable"]
    others = [k for k in range(len(ctx["intervals"])) if k != sc]
    for key in _METRIC_KEYS:
        groups = [m[key][ins], m[key][(ci == sc) & ~ins]]
        labels = ["stable window", f"Config {chr(65+sc)}\noutside stable window"]
        for o in others:
            groups.append(m[key][ci == o])
            labels.append(f"Config {chr(65+o)}\n({ctx['intervals'][o]['record_site']} rec)")
        render.distribution_panel(
            groups, labels, key,
            os.path.join(out_dir, f"{animal}_dist_{key}.png"))
    render.age_stratified_fig(
        m["epoch"][ci == sc], m["amplitude"][ci == sc],
        ctx["intervals"][sc]["start_epoch"], "amplitude",
        os.path.join(out_dir, f"{animal}_age_amplitude.png"))


def _write_markdown(out_dir, animal, ctx) -> None:
    from .report_text import build_markdown
    txt = build_markdown(animal, ctx)
    with open(os.path.join(out_dir, f"{animal}_stability_report.md"),
              "w", encoding="utf-8") as f:
        f.write(txt)
