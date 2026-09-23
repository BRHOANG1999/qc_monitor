"""Auto-pick + orchestrate both prongs of riding-event analysis for one animal.

Discovers the animal's recordings (``all_files_for_animal`` -> each file's evoked
``*_evoked.mat`` via ``evoked_output_path_for_file`` and its raw continuous
``file_path``), ranks them by riding-event prevalence, then renders Prong A
(template subtraction + spectra) and Prong B (continuous detection + matched
filter) on the strongest recording. Writes figures, a per-epoch CSV, a
per-detection CSV and a manifest to ``data/derivatives/riding_event/<animal>/``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess

import numpy as np

from src.riding_event import detect as _det
from src.riding_event import render as _r
from src.riding_event import render_interactive as _ri
from src.riding_event import residual as _res
from src.utils.animal import split_animal_electrode

logger = logging.getLogger("qc_monitor.riding_event.run")

_MAX_SCAN = 400          # NASA Rule 2: bound the auto-pick scan.
_DEFAULT_BAND = (20.0, 200.0)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(__file__),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                             # noqa: BLE001
        return ""


def discover(store, animal: str) -> list[dict]:
    """[{file_id, file_path, chunk_datetime}] for every recording that carries an
    *animal* evoked channel, newest first. Uses the ``evoked_waveforms`` join
    (video-independent, unlike ``all_files_for_animal``) so stim-only chronicStim
    recordings are included. The evoked ``.mat`` path is resolved lazily."""
    assert animal, "animal required"
    conn = store._connect()
    try:
        rows = conn.execute(
            """SELECT DISTINCT pf.id AS id, pf.file_path AS file_path,
                      pf.chunk_datetime AS chunk_datetime
               FROM evoked_waveforms ew JOIN processed_files pf
                 ON pf.id = ew.file_id
               WHERE ew.channel_name LIKE ? AND pf.chunk_datetime IS NOT NULL""",
            (f"{animal}%",)).fetchall()
    finally:
        conn.close()
    out = [{"file_id": int(r["id"]), "file_path": r["file_path"],
            "chunk_datetime": r["chunk_datetime"]} for r in rows]
    out.sort(key=lambda d: d.get("chunk_datetime") or "", reverse=True)
    return out


def _evoked_path(store, rec: dict) -> str | None:
    """Resolve + cache a recording's evoked ``.mat`` path (None when absent)."""
    if "evoked_path" not in rec:
        fid = rec.get("file_id")
        ep = store.evoked_output_path_for_file(int(fid)) if fid else None
        rec["evoked_path"] = ep if (ep and os.path.exists(ep)) else None
    return rec["evoked_path"]


def autopick(store, recs: list, animal: str, *, scan_files: int = 20,
             progress=None) -> list[dict]:
    """Rank *recs* by event prevalence (cheap sub-sampled probe per recording),
    most events first. Scans at most ``min(scan_files, _MAX_SCAN)`` newest files
    whose evoked ``.mat`` is on disk."""
    prog = progress or (lambda *_a: None)
    cap = min(int(scan_files), _MAX_SCAN)
    ranked, scanned = [], 0
    for rec in recs:
        if scanned >= cap:
            break
        ep = _evoked_path(store, rec)
        if not ep:
            continue
        scanned += 1
        prog(f"scan {scanned}/{cap}: {os.path.basename(ep)}")
        pv = _res.event_prevalence(ep, animal)
        ranked.append({**rec, "evoked_path": ep, **pv})
    ranked.sort(key=lambda d: (d.get("event_rate") if np.isfinite(
        d.get("event_rate", float("nan"))) else -1.0), reverse=True)
    return ranked


def build(store, animal: str, *, out_dir: str, mode: str = "both",
          recording: str | None = None, band=None, scan_files: int = 40,
          progress=None) -> dict:
    """Render the requested prong(s) for *animal*. *recording* forces a specific
    evoked path (skips auto-pick). Returns a summary dict."""
    assert animal and out_dir, "animal and out_dir required"
    prog = progress or (lambda *_a: None)
    os.makedirs(out_dir, exist_ok=True)
    target = _resolve_target(store, animal, recording, scan_files, prog)
    if target is None:
        return {"animal": animal, "reason": "no usable recording", "figures": []}
    prog(f"target: {os.path.basename(target['evoked_path'])}")
    summary = {"animal": animal, "target": target, "figures": [],
               "git_sha": _git_sha(), "generated_at": None}
    res_a = None
    if mode in ("both", "residual"):
        res_a = _run_prong_a(target, animal, out_dir, summary, prog)
    if mode in ("both", "detect"):
        _run_prong_b(store, target, animal, out_dir, summary, res_a, band, prog)
    _write_manifest(out_dir, animal, summary)
    return summary


def _resolve_target(store, animal, recording, scan_files, prog) -> dict | None:
    recs = discover(store, animal)
    if not recs:
        return None
    if recording:
        for r in recs:
            ep = _evoked_path(store, r)
            if recording in (ep, os.path.basename(ep) if ep else None,
                             r.get("file_path")):
                return {**r, "evoked_path": ep}
        # a raw/evoked path not in the discovered set: use it as-is (evoked only)
        if os.path.exists(recording):
            return {"file_id": None, "file_path": None, "evoked_path": recording,
                    "chunk_datetime": None}
        return {**recs[0], "evoked_path": _evoked_path(store, recs[0])}
    prog("auto-picking the strongest recording...")
    ranked = autopick(store, recs, animal, scan_files=scan_files, progress=prog)
    top = ranked[0] if ranked else None
    for r in ranked[1:4]:
        prog(f"  runner-up: {os.path.basename(r['evoked_path'])} "
             f"event_rate={r.get('event_rate', float('nan')):.3f}")
    return top


def _run_prong_a(target, animal, out_dir, summary, prog) -> dict | None:
    prog("Prong A: template subtraction + spectra...")
    res = _res.analyze_recording(target["evoked_path"], animal)
    if res is None:
        prog("  no usable channel; skipping Prong A")
        return None
    stem = os.path.join(out_dir, f"{_safe(animal)}_{_safe(res['channel'])}_A")
    figs = [_r.fig_raw_overlay(res, stem + "_ridgeline.png"),
            _r.fig_residual_erpimage(res, stem + "_erpimage.png"),
            _r.fig_density(res, stem + "_density.png"),
            _r.fig_spectral(res, stem + "_spectrum.png")]
    summary["figures"].extend(figs)
    inspector = _ri.write_event_inspector(res, stem + "_inspector.html")
    summary["figures"].append(inspector)
    summary["interactive"] = inspector
    prog(f"  interactive per-event inspector -> {os.path.basename(inspector)}")
    _write_epoch_csv(stem + "_epochs.csv", res)
    fund = res.get("fundamental", {})
    summary["prong_a"] = {
        "channel": res["channel"], "fs": res["fs"], "n": res["n"],
        "n_event": res["n_event"], "event_rate": res["event_rate"],
        "fundamental_hz": fund.get("fundamental_hz"),
        "n_harmonics": fund.get("n_harmonics"), "excess_band": res.get("excess_band")}
    prog(f"  event rate {res['event_rate']*100:.1f}%  fundamental "
         f"{fund.get('fundamental_hz', float('nan')):.1f} Hz")
    return res


def _run_prong_b(store, target, animal, out_dir, summary, res_a, band, prog):
    prog("Prong B: continuous detection + matched filter...")
    elec = split_animal_electrode(res_a["channel"])[1] if res_a else None
    use_band = _detect_band(band, res_a)
    stem = os.path.join(out_dir, f"{_safe(animal)}_B")
    tmpl_a = _det.harvest_from_prongA(res_a) if res_a is not None else {}
    if tmpl_a:                                     # artifact-free stim-locked template
        summary["figures"].append(_r.fig_alignment(
            tmpl_a, stem + "_stimlocked_template.png", fs=res_a["fs"],
            animal=animal, channel=res_a["channel"],
            title_extra="(stim-locked, artifact-removed)"))
    if not target.get("file_path"):
        prog("  no raw continuous path for this recording; skipping the LFP sweep")
        summary["prong_b"] = {"reason": "no raw file_path", "band": use_band}
        return
    loaded = _det.load_channel(target["file_path"], animal, prefer=elec)
    if loaded is None:
        prog("  animal channel not found in the raw recording; skipping")
        summary["prong_b"] = {"reason": "no animal channel", "band": use_band}
        return
    signal, fs, ch = loaded
    override = tmpl_a.get("template") if tmpl_a else None
    stim_times = res_a.get("times") if res_a is not None else None
    blanked = "yes" if stim_times is not None and len(stim_times) else "no"
    prog(f"  detecting on {ch} ({len(signal)} samp @ {fs:.0f} Hz, "
         f"{use_band[0]:.0f}-{use_band[1]:.0f} Hz; "
         f"template={'stim-locked' if override is not None else 'continuous'}; "
         f"stim-blanked={blanked})...")
    det = _det.run_detector(signal, fs, band=use_band, template_override=override,
                            exclude_times_sec=stim_times)
    prox = _validation(store, animal, target, det, fs)
    figs = [_r.fig_candidates(det, stem + "_candidates.png", animal=animal,
                              channel=ch),
            _r.fig_alignment(det["template"], stem + "_template.png", fs=fs,
                             animal=animal, channel=ch, title_extra="(continuous)"),
            _r.fig_matched_filter(det, stem + "_matched.png", animal=animal,
                                  channel=ch, prox=prox)]
    summary["figures"].extend(figs)
    _write_detection_csv(stem + "_detections.csv", det, fs)
    summary["prong_b"] = {
        "channel": ch, "band": use_band, "n_candidates": int(det["cand_locs"].size),
        "n_raw_detections": int(det.get("n_raw_detections", 0)),
        "n_detections": int(np.asarray(det["det_locs"]).size),
        "template_source": det.get("template_source"),
        "template_n": det["template"].get("n") if det["template"] else 0,
        "proximity": prox}
    prog(f"  {int(det['cand_locs'].size)} spontaneous candidates -> "
         f"{int(det.get('n_raw_detections', 0))} shape matches -> "
         f"{int(np.asarray(det['det_locs']).size)} after amplitude gate")


def _detect_band(band, res_a) -> tuple:
    """Detection band: explicit --band, else centred on Prong A's fundamental
    (event energy concentrates around f0 + harmonics), else the excess band, else
    the default 20-200 Hz."""
    if band is not None:
        return (float(band[0]), float(band[1]))
    if res_a is not None:
        nyq = 0.49 * res_a["fs"]
        f0 = res_a.get("fundamental", {}).get("fundamental_hz", float("nan"))
        if np.isfinite(f0) and f0 > 0:
            return (max(1.0, f0 * 0.5), min(f0 * 4.0, nyq))
        lo, hi = res_a.get("excess_band", (float("nan"), float("nan")))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return (max(1.0, lo * 0.8), min(hi * 1.25, nyq))
    return _DEFAULT_BAND


def _validation(store, animal, target, det, fs) -> dict | None:
    """Proximity of detections to scored seizure onsets, when this recording
    overlaps any (best-effort; the demo recording is short and may overlap none)."""
    try:
        from src.preictal.isi import scored_seizures
        from src.riding_event.detect import scores_by_proximity
        from src.utils.evoked_output import parse_recording_dt
        onsets = [s.onset_epoch for s in scored_seizures(store, animal)]
    except Exception:                              # noqa: BLE001
        return None
    if not onsets or not target.get("chunk_datetime"):
        return None
    dt = parse_recording_dt(target["evoked_path"]) or None
    start = dt.timestamp() if dt else None
    if start is None:
        return None
    det_sec = start + np.asarray(det.get("det_locs", []), float) / float(fs)
    return scores_by_proximity(det_sec, onsets, window_sec=300.0)


def _write_epoch_csv(path: str, res: dict) -> None:
    times = np.asarray(res.get("times", []))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch_index", "stim_time_sec", "event_energy", "is_event"])
        for j in range(res["n"]):
            st = float(times[j]) if j < times.size else ""
            w.writerow([j, st, f"{res['energy'][j]:.6g}",
                        int(bool(res["event_mask"][j]))])


def _write_detection_csv(path: str, det: dict, fs: float) -> None:
    locs = np.asarray(det.get("det_locs", []))
    scores = np.asarray(det.get("det_scores", []))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "time_sec", "score"])
        for i in range(locs.size):
            w.writerow([int(locs[i]), f"{locs[i]/fs:.6f}", f"{scores[i]:.4f}"])


def _write_manifest(out_dir: str, animal: str, summary: dict) -> None:
    m = dict(summary)
    m["figures"] = [os.path.basename(p) for p in summary.get("figures", [])]
    tgt = m.get("target") or {}
    m["target"] = {k: tgt.get(k) for k in ("file_id", "evoked_path", "file_path",
                                           "chunk_datetime")}
    with open(os.path.join(out_dir, f"{_safe(animal)}_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(m, f, indent=2, default=str)
