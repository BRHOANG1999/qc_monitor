"""Markdown narrative for the chronic-stim stability map (mapping only)."""

from __future__ import annotations

import datetime as _dt

import numpy as np

_CONV = ("Convention: the NAMED electrode in `BCH111<AREA>` is the RECORD site; "
         "the unnamed site is the STIM target. ⚠️ The codebase's stim_map / "
         "impedance_refresh attribute the current (and thus the stored "
         "`access_r_kohm`) to that named electrode, i.e. the opposite wiring; "
         "the stored 'Rₐ' here is therefore the ΔV/ΔI on the RECORD channel "
         "while the other site is stimulated. It is used only as a stable, "
         "monotonic gating signal for interface stability.")


def _iso(e) -> str:
    return _dt.datetime.fromtimestamp(float(e)).strftime("%Y-%m-%d %H:%M")


def build_markdown(animal: str, ctx: dict) -> str:
    """Assemble the full mapping report."""
    L = [f"# {animal} — Stimulus-Stability History (mapping)", "",
         f"_Generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} "
         f"· read-only · mapping only, no experimental recommendations._", "",
         _CONV, ""]
    L += _sec_configs(ctx)
    L += _sec_swap(ctx)
    L += _sec_stable(ctx)
    L += _sec_traces(ctx)
    L += _sec_comparisons(ctx)
    L += _sec_collinearity(ctx)
    L += _sec_caveats(ctx)
    L += _sec_maintenance(ctx)
    return "\n".join(L)


def _sec_traces(ctx) -> list:
    """Point at the raw-waveform figures (configs named A/B as in §1)."""
    a = ctx["animal"]
    return [
        "## 4. Raw artifact waveforms (examples, not summaries)", "",
        f"Configs are lettered A/B as in §1. From a uniform time-ordered sample "
        f"of {ctx['sample']['seg'].shape[0]:,} trials "
        f"(of ~{ctx['metrics']['epoch'].size:,} total across the record):",
        f"- `{a}_erpimage.png` = that sample stacked by time, ONE real trial per "
        "row (colour = µV): the whole history at the waveform level "
        "(polarity flips, amplitude drift, the swap, the flat stable plateau). "
        "Note it is a sample, not every trial; at this image height even the "
        "sample already exceeds the pixel rows.",
        f"- `{a}_trace_gallery.png` — ~14 individual example traces per config "
        "+ the config mean + the stable template (dashed).",
        f"- `{a}_mean_traces.png` — mean artifact ± SD per config, overlaid.", ""]


def _sec_configs(ctx) -> list:
    L = ["## 1. Configuration timeline", "",
         "| config | stim | rec | start | end | dur (h) | trials | seizures |"
         " charge (nC) | Rₐ med (kΩ) | amp med | corr med | nRMSE med |"
         " pol switches |",
         "|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for i, c in enumerate(ctx["per_config"]):
        end = "ongoing" if not np.isfinite(c["end_epoch"]) else _iso(c["end_epoch"])
        L.append(
            f"| **{chr(65+i)}** {c['config_label']} | {c['stim_site']} | {c['record_site']} | "
            f"{_iso(c['start_epoch'])} | {end} | {c['duration_h']:.0f} | "
            f"{c['n_trials']} | {c['n_seizures']} | {c['modal_charge_nC']} | "
            f"{c['ra_median']:.3f} | {c['amp_median']:.2f} | "
            f"{c['corr_median']:.3f} | {c['nrmse_median']:.3f} | "
            f"{c['polarity_switch']} |")
    L += ["", f"Total seizures for {ctx['animal']} (record-wide): "
          f"{ctx['n_seizures_total']}.", ""]
    return L


def _sec_swap(ctx) -> list:
    L = ["## 2. Configuration swap(s) & within-config events", ""]
    swaps = [e for e in ctx["events"] if e["event_type"] == "site_swap"]
    if not swaps:
        L.append("No record-site swap detected — a single configuration.")
    for e in swaps:
        p = e["provenance"]
        L.append(f"- **SITE SWAP @ {_iso(e['at_epoch'])}**: {e['from_config']} "
                 f"→ {e['to_config']}. Provenance: `{p['table']}.{p['field']}`, "
                 f"session before `{_base(p['session_dir_before'])}` → after "
                 f"`{_base(p['session_dir_after'])}` (report ch "
                 f"{p['report_channel_after']}, fp {p['fp_status_after']}).")
    chg = [e for e in ctx["events"] if e["event_type"] == "charge_change"]
    if chg:
        L.append("")
        L.append(f"Within-config charge changes (NOT configuration swaps): "
                 + "; ".join(f"{_iso(e['at_epoch'])} {e['from_config']}→"
                             f"{e['to_config']}" for e in chg))
    L.append("")
    return L


def _sec_stable(ctx) -> list:
    r = ctx["run"]
    cfg = ctx["per_config"][ctx["stable_cfg"]]
    day0 = float(_np_min_epoch(ctx))
    L = ["## 3. Stable window (longest programmatic flat Rₐ run)", "",
         f"Detected on the **{cfg['record_site']}** record-electrode Rₐ trend "
         f"(Config {chr(65+ctx['stable_cfg'])}, {cfg['config_label']}). "
         f"See `{ctx['animal']}_ra_over_time.png` (both electrodes on one axis) "
         f"and `{ctx['animal']}_ra_aligned.png` (configs aligned, comparable).",
         "",
         f"- **Boundaries:** {_iso(r['epoch_start'])} → {_iso(r['epoch_end'])} "
         f"= **day {(r['epoch_start']-day0)/86400:.1f} → "
         f"{(r['epoch_end']-day0)/86400:.1f}** of the record "
         f"(trial-index {r['idx_start']}–{r['idx_end']} on the Rₐ series).",
         f"- **Duration:** {r['duration_h']:.1f} h "
         f"({r['duration_h']/24:.1f} days) · {r['n_samples']} Rₐ points "
         f"· median Rₐ {r['median_ra_kohm']:.3f} kΩ · robust CV "
         f"{r['robust_cv_pct']:.1f}%.",
         f"- **Artifacts:** {len(r['flagged_spikes_in_run'])} lone spike(s) "
         f"tolerated; {r['n_gaps_tolerated']} gap(s) tolerated, max gap "
         f"{r['max_gap_h']:.1f} h (a gap > tolerance would break the run).",
         f"- **Seizures inside the window:** **{r['seizures_in_window']}** "
         f"(expectation was ≥6; this is an OUTCOME, not a selection criterion).",
         f"- **Position vs swap:** the window starts "
         f"{(r['epoch_start']-ctx['intervals'][ctx['stable_cfg']]['start_epoch'])/86400.0:.1f} "
         f"days after the configuration it lives in began.", ""]
    return L


def _np_min_epoch(ctx) -> float:
    return float(ctx["metrics"]["epoch"].min())


def _sec_comparisons(ctx) -> list:
    L = ["## 5. Distributions (not point estimates)", "",
         "| comparison | n(a) | n(b) | med(a) | med(b) | KS p | MWU p | "
         "Cliff's δ | Δmedian [95% block-CI] |",
         "|---|--:|--:|--:|--:|--:|--:|--:|---|"]
    for c in ctx["comparisons"]:
        if "median_a" not in c:
            continue
        L.append(
            f"| {c['name']} | {c['n_a']} | {c['n_b']} | {c['median_a']:.3f} | "
            f"{c['median_b']:.3f} | {c['ks_p']:.1e} | {c['mwu_p']:.1e} | "
            f"{c['cliffs_delta']:+.3f} | {c['median_diff']:+.3f} "
            f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] |")
    pol = [c for c in ctx["comparisons"] if "frac_pos_a" in c]
    for c in pol:
        L += ["", f"Polarity ({c['name']}): frac(+) {c['frac_pos_a']:.3f} vs "
              f"{c['frac_pos_b']:.3f}, Fisher p {c['fisher_p']:.1e}."]
    L += ["", "_CIs use a dependence-aware moving-block bootstrap (consecutive "
          "trials are autocorrelated; an iid bootstrap would understate the "
          "CI)._", ""]
    return L


def _sec_collinearity(ctx) -> list:
    b = ctx["collinearity"]["stable_age_band"]
    L = ["## 6. Configuration age vs stability (collinearity — reported, not "
         "adjusted)", "",
         f"The stable window occupies configuration age "
         f"{b.get('stable_age_lo_h', float('nan')):.0f}–"
         f"{b.get('stable_age_hi_h', float('nan')):.0f} h, out of the config's "
         f"full {b.get('config_age_lo_h', float('nan')):.0f}–"
         f"{b.get('config_age_hi_h', float('nan')):.0f} h span "
         f"({100*b.get('stable_frac_of_config_age', float('nan')):.0f}% of its "
         f"age range). Stability and age are therefore confounded — a "
         f"stable-vs-full difference is read JOINTLY with age.", "",
         "Spearman ρ(config age, metric) per configuration:", ""]
    for lab, d in ctx["collinearity"]["per_config"].items():
        parts = ", ".join(f"{k} ρ={d[k]['rho']:+.2f} (p={d[k]['p']:.1e})"
                          for k in ("amplitude", "corr", "nrmse") if k in d)
        L.append(f"- {lab}: {parts}")
    L.append("")
    return L


def _sec_caveats(ctx) -> list:
    return [
        "## 7. Caveats & null/ambiguous results", "",
        "- **Cross-config shape:** the two configurations record from DIFFERENT "
        "electrodes, so corr/nRMSE of the other config vs a stable-config "
        "template conflate stimulus-delivery change with record-electrode "
        "identity. Amplitude / polarity / Rₐ are the cleaner cross-config "
        "comparators.",
        "- **Age–stability collinearity** is intrinsic (see §5) and is not "
        "adjusted away.",
        f"- **Seizure expectation:** the stable window holds "
        f"{ctx['run']['seizures_in_window']} seizures, below the anticipated ≥6 "
        "— reported as an outcome.",
        "- **Timestamps** are naive local wall-clock (evoked filenames and DB "
        "chunk_datetime share the same basis; no UTC layer).", ""]


def _base(p: str) -> str:
    import os
    return os.path.basename(p or "")


def _sec_maintenance(ctx) -> list:
    """Notes section: software stop/start times around battery / cage changes,
    plus the human-reported (approximate) change times."""
    mev = ctx.get("maintenance") or {}
    L = ["## 8. Maintenance log (recording stop/start around changes)", ""]
    if not mev.get("available"):
        return L + [f"Maintenance overlay unavailable ({mev.get('error')}). "
                    "The battery / cage change lines are absent from the "
                    "figures for this run.", ""]
    soft, bat, cage = mev["software"], mev["battery_human"], mev["cage_human"]
    L += [f"Ground truth = the recorder's software Notes tab ({len(soft)} "
          f"battery/cage stop/start events in the record window). The per-rig "
          f"Maintenance Tracker (Rig C) adds {len(bat)} battery and {len(cage)} "
          f"cage change times that are HUMAN-reported and approximate (not the "
          f"true instant of the change).", ""]
    L += _maint_cooccurrence(ctx)
    L += ["", "**Software stop/start events (ground truth):**", "",
          "| datetime | event | kind | note |", "|---|---|---|---|"]
    for e in soft:
        L.append(f"| {e['date_iso']} | {e['event']} | {e['kind']} | "
                 f"{e['text'][:80].replace('|', '/')} |")
    L += ["", "**Human-reported change times (approximate):**",
          "- Battery: " + ", ".join(e["date_iso"] for e in bat),
          "- Cage: " + ", ".join(e["date_iso"] for e in cage), ""]
    return L



def _maint_cooccurrence(ctx) -> list:
    """Neutral, factual co-occurrence notes (mapping only, no mechanism claim)."""
    lines = ctx.get("fig_lines", {})
    bat = np.asarray(lines.get("battery", []), float)
    r = ctx["run"]
    n_in = int(((bat >= r["epoch_start"]) & (bat <= r["epoch_end"])).sum())
    swap = next((e for e in ctx["events"] if e["event_type"] == "site_swap"),
                None)
    out = [f"- {n_in} battery change(s) fall INSIDE the stable window "
           f"(Rₐ robust CV {r['robust_cv_pct']:.1f}%); Rₐ stays flat across them."]
    if swap is not None and bat.size:
        near = float(np.min(np.abs(bat - swap["at_epoch"]))) / 3600.0
        out.append(f"- The configuration swap ({_iso(swap['at_epoch'])}) is "
                   f"{near:.1f} h from the nearest logged battery change "
                   "(the swap and a battery change were logged together).")
    return out
