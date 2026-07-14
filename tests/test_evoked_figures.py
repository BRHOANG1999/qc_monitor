"""Evoked-metric figure export: data prep, the SHIPPED-CSV epoch round-trip,
matplotlib rendering, and build + zombie-prune.

Run with: pytest tests/test_evoked_figures.py -q
"""

from __future__ import annotations

import csv
import io
import os
import sys
from datetime import datetime, timedelta

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import evoked_features as ef                    # noqa: E402
from src.utils.evoked_output import write_feature_sidecar       # noqa: E402
from src.evoked_figures import data as _data                    # noqa: E402
from src.evoked_figures import render as _render                # noqa: E402
from src.evoked_figures import run as _run                      # noqa: E402

_BASE = datetime(2025, 3, 1, 13, 45, 30)


def _row(channel, i, **vals):
    dt = _BASE + timedelta(seconds=60 * i)
    r = {"channel": channel, "electrode": channel, "rec_dt": _BASE.isoformat(),
         "session": "sess", "stim_time_sec": 60 * i, "abs_dt": dt.isoformat(),
         "peak": 1.0, "trough": -1.0}
    for m in ef.ALL_COLUMNS:
        r[m] = None
    r.update(vals)
    return r


def _seed(evoked_dir, animal, rows):
    fname = "sess__stimCopy_BCH062SR_BCH062SLM_2025_03_01__00_00_00_evoked.mat"
    mat = os.path.join(evoked_dir, fname)
    with open(mat, "wb") as f:
        f.write(b"\x00")                    # dummy .mat so mtime stamping works
    write_feature_sidecar(mat, animal, rows)
    return mat


# ------------------------------------------------------------------ #
#  data prep: primary channel, series, percentile bands
# ------------------------------------------------------------------ #

def test_primary_channel_and_series(tmp_path):
    ev = tmp_path / "evoked"; ev.mkdir()
    rows = ([_row("BCH062SR", i, line_length=float(i)) for i in range(20)]
            + [_row("BCH062SLM", i, line_length=float(i)) for i in range(5)])
    _seed(str(ev), "BCH062", rows)

    series, inputs = _data.animal_series("BCH062", str(ev), ["line_length"])
    assert set(series) == {"BCH062SR", "BCH062SLM"} and len(inputs) == 1
    assert series["BCH062SR"]["secs"].size == 20
    # SR has more finite points -> primary
    assert _data.primary_channel(
        "BCH062", series, ["line_length"]) == "BCH062SR"

    dts, secs, vals = _data.finite_series(series, "BCH062SR", "line_length")
    assert vals.size == 20 and list(vals) == [float(i) for i in range(20)]
    assert secs[0] < secs[-1]                          # sorted ascending
    assert len(dts) == 20

    bands = _data.rolling_percentiles(secs, vals, win=11)
    assert bands is not None
    bsec, p10, p25, p50, p75, p90 = bands
    assert bsec.size == p50.size and (p10 <= p90).all() and (p25 <= p75).all()


def test_series_is_nan_filled_and_finite_filtered(tmp_path):
    """A metric with no data for the animal comes back as an all-NaN column
    (stable declared set), and finite_series drops it to an empty series."""
    ev = tmp_path / "evoked"; ev.mkdir()
    _seed(str(ev), "BCH062",
          [_row("BCH062SR", i, line_length=float(i)) for i in range(12)])
    series, _ = _data.animal_series("BCH062", str(ev),
                                    ["line_length", "rms_amplitude"])
    import numpy as np
    assert np.isnan(series["BCH062SR"]["metrics"]["rms_amplitude"]).all()
    _d, _s, v = _data.finite_series(series, "BCH062SR", "rms_amplitude")
    assert v.size == 0                                  # -> reported "missing"


# ------------------------------------------------------------------ #
#  epoch round-trip THROUGH THE SHIPPED CSV (ISO string is the format)
# ------------------------------------------------------------------ #

def test_epoch_roundtrip_through_csv():
    known = datetime(2025, 3, 1, 13, 45, 30)
    row = _row("BCH062SR", 0, line_length=1.0)
    row["abs_dt"] = known.isoformat()
    text = _data.rows_to_csv([row])
    reader = list(csv.DictReader(io.StringIO(text)))
    cell = reader[0]["abs_dt"]                          # the stored ISO string
    assert cell == known.isoformat()
    parsed = _data.parse_iso(cell)                      # what the plotter parses
    assert parsed == known                              # no silent axis shift
    # matplotlib's date encoding must round-trip the parsed value too
    import matplotlib.dates as mdates
    assert mdates.num2date(mdates.date2num(parsed)).replace(
        tzinfo=None) == known


# ------------------------------------------------------------------ #
#  rendering writes real PNGs
# ------------------------------------------------------------------ #

def test_gap_break_inserts_nan_across_silence():
    """The rolling percentile uses a COUNT window, so consecutive band samples
    can be weeks apart; without a break, matplotlib draws a confident line
    through a period with no recordings. Assert a >24h gap becomes a NaN break
    (and that a continuous run is left untouched)."""
    import numpy as np
    base = datetime(2026, 1, 1)
    # 3 points, then a 10-day hole, then 3 more
    dts = [base + timedelta(hours=i) for i in range(3)]
    dts += [base + timedelta(days=10, hours=i) for i in range(3)]
    vals = np.arange(6, dtype=float)
    nd, (nv,) = _render._break_gaps(dts, (vals,), gap_sec=24 * 3600)
    assert len(nd) == 7 and nv.size == 7          # one NaN sample inserted
    assert np.isnan(nv).sum() == 1
    assert np.isnan(nv[3])                        # break sits in the hole
    # a continuous run is untouched
    nd2, (nv2,) = _render._break_gaps(dts[:3], (vals[:3],), gap_sec=24 * 3600)
    assert len(nd2) == 3 and not np.isnan(nv2).any()


def test_render_writes_pngs(tmp_path):
    dts = [_BASE + timedelta(seconds=60 * i) for i in range(30)]
    vals = [float(i) for i in range(30)]
    sp = _render.render_scatter(dts, vals, "t", "line_length",
                                str(tmp_path / "s.png"))
    assert os.path.getsize(sp) > 0
    pp = _render.render_percentile(dts, vals, vals, vals, vals, vals, "t",
                                   "line_length", str(tmp_path / "p.png"),
                                   pts_dts=dts, pts_vals=vals)
    assert os.path.getsize(pp) > 0


# ------------------------------------------------------------------ #
#  build_animal end-to-end + missing-metric accounting
# ------------------------------------------------------------------ #

def _config(ev, out, metrics):
    return {"chronic_evoked": {"evoked_output_dir": str(ev)},
            "evoked_figures": {"out_root": str(out), "metrics": metrics,
                               "roll_window": 11}}


def test_build_animal_writes_tree_and_flags_missing(tmp_path):
    ev = tmp_path / "evoked"; ev.mkdir()
    out = tmp_path / "out"
    # line_length has data; rms_amplitude is all-None -> "missing"
    rows = [_row("BCH062SR", i, line_length=float(i)) for i in range(20)]
    _seed(str(ev), "BCH062", rows)

    s = _run.build_animal("BCH062",
                          _config(ev, out, ["line_length", "rms_amplitude"]),
                          apply=True, now_iso="2026-07-13T00:00:00+00:00")
    assert s["channel"] == "BCH062SR"
    assert s["rendered"] == 1 and "rms_amplitude" in s["missing"]
    adir = out / "BCH062"
    gz = adir / "BCH062_evoked_timeseries.csv.gz"
    assert gz.is_file()
    # the gzipped CSV is the shipped data format: it must read back + parse
    import gzip
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        recs = list(csv.DictReader(f))
    assert len(recs) == 20 and _data.parse_iso(recs[0]["abs_dt"]) is not None
    assert (adir / "line_length" / "BCH062_line_length_scatter.png").is_file()
    assert (adir / "line_length" / "BCH062_line_length_percentile.png").is_file()
    assert not (adir / "rms_amplitude").exists()       # no data -> no dir
    import json
    man = json.loads((adir / "manifest.json").read_text())
    assert man["primary_channel"] == "BCH062SR"
    assert "rms_amplitude" in man["missing_metrics"]


def test_zombie_prune_removes_dropped_metric(tmp_path):
    ev = tmp_path / "evoked"; ev.mkdir()
    out = tmp_path / "out"
    rows = [_row("BCH062SR", i, line_length=float(i), rms_amplitude=float(i))
            for i in range(20)]
    _seed(str(ev), "BCH062", rows)

    _run.build_animal("BCH062",
                      _config(ev, out, ["line_length", "rms_amplitude"]),
                      apply=True)
    assert (out / "BCH062" / "rms_amplitude").is_dir()
    # re-run with rms dropped from the declared set -> its dir is pruned
    s = _run.build_animal("BCH062", _config(ev, out, ["line_length"]),
                          apply=True)
    assert s["pruned"] >= 1
    assert not (out / "BCH062" / "rms_amplitude").exists()
    assert (out / "BCH062" / "line_length").is_dir()
