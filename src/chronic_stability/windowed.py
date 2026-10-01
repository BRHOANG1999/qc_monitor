"""Zoomed, seizure-labelled views of the chronic-stability access-resistance
timeline.

Renders 24 h / 2 day / 1 week "states" of the Rₐ-over-time figure, each centred
on the densest seizure cluster and labelling every seizure that falls inside the
window ("label when they occur"). Reuses the *lightweight* half of the stability
pipeline -- impedance series, config intervals, stable-run detection, seizure
onsets, maintenance lines -- so it does NOT stream per-trial metrics and is cheap
to re-run.

CLI:  python -m src.chronic_stability.windowed [--animal BCH111] [--out DIR]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

import numpy as np

from src.dashboard.data_helpers import load_config
from src.db.store import Store
from src.preictal.isi import scored_seizures
from . import config_timeline as ct, maintenance, render
from .report import _detect_stable

# (tag, duration seconds, human label) for each "state". "and so on" -> extend.
_DURATIONS = [("24h", 24 * 3600.0, "24 hour"),
              ("2day", 2 * 86400.0, "2 day"),
              ("1week", 7 * 86400.0, "1 week")]


def _log(msg: str) -> None:
    print(f"[chronic_stability.windowed] {msg}", flush=True)


def _fmt(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(float(epoch)).strftime("%Y-%m-%d %H:%M")


def _densest_center(seizures, span_sec: float) -> tuple[float, int]:
    """Centroid epoch of the *span_sec* window holding the most seizures (ties ->
    earliest). Bounded loop over the sorted onsets."""
    e = np.sort(np.asarray(seizures, float))
    assert e.size, "no seizures to anchor the windows on"
    best_c, best_n = float(e[0]), 0
    for i in range(e.size):
        inw = e[(e >= e[i]) & (e <= e[i] + span_sec)]
        if inw.size > best_n:
            best_n, best_c = int(inw.size), float(inw.mean())
    return best_c, best_n


def _clamp(lo: float, hi: float, data_lo: float, data_hi: float):
    """Shift [lo, hi] to stay within the data span, preserving its width."""
    assert hi > lo and data_hi > data_lo, "bad window / data span"
    width = hi - lo
    if width >= data_hi - data_lo:
        return data_lo, data_hi
    if lo < data_lo:
        lo, hi = data_lo, data_lo + width
    if hi > data_hi:
        lo, hi = data_hi - width, data_hi
    return lo, hi


def load_context(store, animal: str) -> dict:
    """Lightweight context for the Rₐ timeline (no per-trial metric streaming)."""
    assert animal, "animal required"
    sessions = ct.bch111_sessions(store, animal)
    intervals = ct.config_intervals(sessions)
    assert intervals, "no configurations found"
    best, ra = _detect_stable(store, animal, intervals, {})
    assert best is not None, "no stable Rₐ run detected in any configuration"
    _, run_ = best
    sz = np.asarray([s.onset_epoch for s in scored_seizures(store, animal)],
                    float)
    assert sz.size, "no scored seizures found for this animal"
    spans = [(te[0], te[-1]) for te, _r in ra.values() if len(te)]
    assert spans, "no Rₐ data points found"
    data_lo = min(s for s, _ in spans)
    data_hi = max(e for _, e in spans)
    ev = maintenance.load_events(load_config(), data_lo, data_hi)
    return {"intervals": intervals, "ra": ra, "run": run_, "seizures": sz,
            "data_lo": float(data_lo), "data_hi": float(data_hi),
            "lines": maintenance.figure_lines(ev)}


def build(animal: str = "BCH111", out_dir: str | None = None) -> list[str]:
    """Render the 24 h / 2 day / 1 week windowed Rₐ figures; return their paths."""
    cfg = load_config()
    store = Store(cfg["database"]["path"])
    _log(f"loading Rₐ series / intervals / seizures for {animal} ...")
    c = load_context(store, animal)
    _log(f"{c['seizures'].size} seizures; record spans "
         f"{_fmt(c['data_lo'])} .. {_fmt(c['data_hi'])}")
    center, n24 = _densest_center(c["seizures"], 24 * 3600.0)
    _log(f"densest 24 h cluster: {n24} seizures, centred {_fmt(center)}")
    out_dir = out_dir or os.path.join("data", "derivatives",
                                      "chronic_stability", animal, "windows")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, (tag, dur, label) in enumerate(_DURATIONS, 1):
        lo, hi = _clamp(center - dur / 2, center + dur / 2,
                        c["data_lo"], c["data_hi"])
        n = int(((c["seizures"] >= lo) & (c["seizures"] <= hi)).sum())
        _log(f"[{i}/{len(_DURATIONS)}] rendering {label} window "
             f"({_fmt(lo)} .. {_fmt(hi)}) -- {n} seizures ...")
        out = os.path.join(out_dir, f"{animal}_ra_over_time_{tag}.png")
        render.ra_over_time_window_fig(
            c["ra"], c["intervals"], c["run"], c["seizures"], out,
            win_lo=lo, win_hi=hi, label=label, animal=animal,
            events=c["lines"])
        _log(f"        wrote {out}")
        paths.append(out)
    _log(f"done -> {out_dir}")
    return paths


def _args(argv):
    p = argparse.ArgumentParser(description="Windowed Rₐ-over-time figures")
    p.add_argument("--animal", default="BCH111")
    p.add_argument("--out", default=None,
                   help="output dir (default: data/derivatives/"
                        "chronic_stability/<animal>/windows)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):           # Rₐ/µ glyphs in console
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    a = _args(argv)
    paths = build(a.animal, a.out)
    assert paths, "no figures were produced"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
