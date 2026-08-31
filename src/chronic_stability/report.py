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
from . import compare, config_timeline as ct, metrics as M, ra_flatrun as fr
from . import render

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
        eye_lo: float, eye_hi: float, flat_params: dict | None = None,
        n_boot: int = 2000) -> dict:
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
    ctx = _assemble(store, animal, intervals, events, ra, stable_cfg, run_,
                    mets, seiz, sz_ep, eye_lo, eye_hi, tmpl, n_boot)
    _write_all(out_dir, animal, ctx)
    _log(f"done -> {out_dir}")
    return ctx


def _template(files, animal, run_):
    """Build the fixed artifact template from stable-window segments."""
    seg = M.collect_segments(files, animal, _ART_LO, _ART_HI,
                             run_["epoch_start"], run_["epoch_end"])
    assert seg["seg"].shape[0] >= 50, "too few stable-window trials for template"
    return M.build_template(seg["seg"])


def _assemble(store, animal, intervals, events, ra, stable_cfg, run_, mets,
              seiz, sz_ep, eye_lo, eye_hi, tmpl, n_boot) -> dict:
    """Assign trials to configs, count seizures, run comparisons + collinearity."""
    cfg_idx = ct.assign_epochs_to_configs(mets["epoch"], intervals)
    in_stable = (mets["epoch"] >= run_["epoch_start"]) \
        & (mets["epoch"] <= run_["epoch_end"])
    val = fr.validate_against_window(run_, ra[stable_cfg][0], eye_lo, eye_hi)
    per_cfg = _per_config_summary(intervals, cfg_idx, mets, sz_ep, ra)
    run_["seizures_in_window"] = int(((sz_ep >= run_["epoch_start"])
                                      & (sz_ep <= run_["epoch_end"])).sum())
    comps = _comparisons(intervals, stable_cfg, cfg_idx, in_stable, mets, n_boot)
    collin = _collinearity(intervals, stable_cfg, cfg_idx, in_stable, mets)
    return {"animal": animal, "intervals": intervals, "events": events,
            "stable_cfg": stable_cfg, "run": run_, "validation": val,
            "per_config": per_cfg, "comparisons": comps, "collinearity": collin,
            "metrics": mets, "cfg_idx": cfg_idx, "in_stable": in_stable,
            "ra": ra, "seizure_epochs": sz_ep, "eye": (eye_lo, eye_hi),
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


def _comparisons(intervals, stable_cfg, cfg_idx, in_stable, mets, n_boot) -> list:
    """Stable-vs-own-config, stable-config-vs-other, stable-window-vs-other."""
    sc = cfg_idx == stable_cfg
    others = [k for k in range(len(intervals)) if k != stable_cfg]
    out = []
    for key in _METRIC_KEYS:
        v = mets[key]
        out.append(compare.compare_two_groups(
            v[in_stable], v[sc & ~in_stable],
            name=f"{key}: stable-window vs rest-of-config", n_boot=n_boot))
        for o in others:
            oc = cfg_idx == o
            out.append(compare.compare_two_groups(
                v[sc], v[oc], name=f"{key}: stable-config vs config#{o}",
                n_boot=n_boot))
            out.append(compare.compare_two_groups(
                v[in_stable], v[oc],
                name=f"{key}: stable-window vs config#{o}", n_boot=n_boot))
    out.append(compare.compare_polarity(
        mets["polarity"][sc], mets["polarity"][cfg_idx == others[0]],
        name="polarity: stable-config vs other") if others else {})
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


def _write_figures(out_dir, animal, ctx) -> None:
    p = lambda n: os.path.join(out_dir, f"{animal}_{n}")
    sc = ctx["stable_cfg"]
    te, r = ctx["ra"][sc]
    render.ra_flatrun_fig(te, r, ctx["run"], *ctx["eye"], ctx["seizure_epochs"],
                          p("flatrun_validation.png"))
    m = ctx["metrics"]
    render.metric_timeseries_fig(
        m["epoch"], m, te, r, ctx["intervals"], ctx["run"],
        ctx["seizure_epochs"], p("metric_timeseries.png"))
    _distribution_figs(out_dir, animal, ctx)


def _distribution_figs(out_dir, animal, ctx) -> None:
    m = ctx["metrics"]
    sc = ctx["stable_cfg"]
    ci = ctx["cfg_idx"]
    ins = ctx["in_stable"]
    others = [k for k in range(len(ctx["intervals"])) if k != sc]
    for key in _METRIC_KEYS:
        groups = [m[key][ins], m[key][(ci == sc) & ~ins]]
        labels = ["stable window", "rest of stable-config"]
        for o in others:
            groups.append(m[key][ci == o])
            labels.append(f"config#{o}")
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
