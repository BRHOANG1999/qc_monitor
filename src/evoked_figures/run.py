"""Build one animal's evoked-metric figure set: pick the primary channel,
render scatter + percentile PNGs for every declared metric, write the colocated
CSV + provenance manifest, and prune stale metric dirs (zombie cleanup)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

from src.evoked_figures import config as _cfg
from src.evoked_figures import data as _data
from src.evoked_figures import render as _render

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, cwd=_REPO_ROOT)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:                                  # noqa: BLE001
        return ""


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name))


def inputs_signature(inputs: list) -> str:
    """Stable hash of an animal's (sidecar, mtime) inputs — lets the worker skip
    an animal whose evoked data is unchanged since its last success."""
    h = hashlib.sha256()
    for path, mtime in sorted(inputs):
        h.update(os.path.basename(path).encode("utf-8", "replace"))
        h.update(f"|{mtime:.3f}|".encode("ascii"))
    return h.hexdigest()


def _prune_stale(animal_dir: str, keep: set) -> int:
    """Remove metric subdirs not in *keep* (metrics dropped from the declared
    set or no longer carrying data). Files (CSV, manifest) are left alone."""
    n = 0
    try:
        entries = os.listdir(animal_dir)
    except OSError:
        return 0
    for name in entries:
        p = os.path.join(animal_dir, name)
        if not os.path.isdir(p) or name in keep:
            continue
        for f in os.listdir(p):
            try:
                os.remove(os.path.join(p, f))
            except OSError:
                pass
        try:
            os.rmdir(p)
            n += 1
        except OSError:
            pass
    return n


def _write_manifest(animal_dir, animal, channel, metrics, rendered, missing,
                    inputs, config, now_iso) -> None:
    manifest = {
        "animal": animal,
        "primary_channel": channel,
        "git_sha": _git_sha(),
        "generated_at": now_iso or datetime.now(timezone.utc).isoformat(),
        "roll_window": _cfg.roll_window(config),
        "declared_metrics": list(metrics),
        "rendered_metrics": list(rendered),
        "missing_metrics": list(missing),
        "inputs_signature": inputs_signature(inputs),
        "n_inputs": len(inputs),
        "inputs": [{"sidecar": os.path.basename(p), "mtime": mt}
                   for p, mt in sorted(inputs)],
    }
    tmp = os.path.join(animal_dir, "manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, os.path.join(animal_dir, "manifest.json"))


def build_animal(animal: str, config: dict, *, apply: bool = True,
                 now_iso: str | None = None) -> dict:
    """Render *animal*'s scatter + percentile PNGs for every declared metric on
    its primary channel; write the colocated CSV + manifest; prune stale metric
    dirs. ``apply=False`` is a dry-run (counts, no writes). Returns a summary."""
    assert animal, "animal required"
    evoked_dir = _cfg.evoked_dir(config)
    root = _cfg.out_root(config)
    metrics = _cfg.declared_metrics(config)
    win = _cfg.roll_window(config)
    override = _cfg.primary_overrides(config).get(animal)

    # ONE pass -> float arrays per (channel, metric). Never holds the row dicts,
    # so a 2.3M-epoch animal (BCH040) costs ~400 MB, not GBs.
    series, inputs = _data.animal_series(animal, evoked_dir, metrics)
    if not series:
        return {"animal": animal, "channel": None, "rendered": 0,
                "missing": list(metrics), "pruned": 0, "n_rows": 0,
                "inputs_sig": inputs_signature(inputs), "note": "no fresh rows"}
    channel = _data.primary_channel(animal, series, metrics, override)
    animal_dir = os.path.join(root, _safe(animal))

    n_rows = 0
    if apply:
        os.makedirs(animal_dir, exist_ok=True)
        # streamed: the CSV for a big animal is >1 GB raw
        n_rows = _data.stream_channel_csv_gz(
            animal, evoked_dir, channel,
            os.path.join(animal_dir, f"{_safe(animal)}_evoked_timeseries.csv.gz"))

    rendered: list = []
    missing: list = []
    for m in metrics:
        dts, secs, vals = _data.finite_series(series, channel, m)
        if vals.size == 0:
            missing.append(m)
            continue
        rendered.append(m)
        if not apply:
            continue
        mdir = os.path.join(animal_dir, _safe(m))
        os.makedirs(mdir, exist_ok=True)
        title = f"{animal} · {channel} · {m}"
        _render.render_scatter(
            dts, vals, f"{title} — scatter", m,
            os.path.join(mdir, f"{_safe(animal)}_{_safe(m)}_scatter.png"))
        pct_png = os.path.join(mdir, f"{_safe(animal)}_{_safe(m)}_percentile.png")
        bands = _data.rolling_percentiles(secs, vals, win)
        if bands is not None:
            bsec, p10, p25, p50, p75, p90 = bands
            bdts = [datetime.fromtimestamp(s) for s in bsec]
            _render.render_percentile(
                bdts, p10, p25, p50, p75, p90,
                f"{title} — percentiles (win {win})", m, pct_png,
                pts_dts=dts, pts_vals=vals,
                gap_sec=_cfg.gap_break_sec(config))
        else:                                          # too few for a band
            _render.render_scatter(
                dts, vals, f"{title} — percentiles (too few points)", m, pct_png)

    pruned = 0
    if apply:
        pruned = _prune_stale(animal_dir, {_safe(m) for m in rendered})
        _write_manifest(animal_dir, animal, channel, metrics, rendered, missing,
                        inputs, config, now_iso)
    else:                                   # dry-run: report the epoch count
        d = series.get(channel) or {}
        n_rows = int((d.get("secs") is not None) and d["secs"].size or 0)
    return {"animal": animal, "channel": channel, "rendered": len(rendered),
            "missing": missing, "pruned": pruned, "n_rows": n_rows,
            "inputs_sig": inputs_signature(inputs)}
