"""Pre-ictal figures + self-contained HTML report.

Tests the ASSEMBLY logic (survivor detection, table, headline, section
presence) without depending on kaleido, plus a best-effort PNG render that
degrades gracefully when the image backend is unavailable.

Run with: pytest tests/test_preictal_report.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.preictal import figures, report  # noqa: E402


def _rows(surviving=True):
    return [
        {"feature": "line_length", "pseudo_freq_hz": 1.0 / 40.0,
         "collapse_stat": 0.30 if surviving else -0.01,
         "null_mean": 0.0, "null_std": 0.05,
         "null_p": 0.004 if surviving else 0.7,
         "collapse_loso_mean": 0.28, "forecast_roc_auc": 0.71,
         "forecast_pr_auc": 0.6, "n_seizures": 12},
        {"feature": "line_length", "pseudo_freq_hz": 1.0 / 4.0,
         "collapse_stat": -0.02, "null_mean": 0.0, "null_std": 0.05,
         "null_p": 0.5, "forecast_roc_auc": 0.48, "forecast_pr_auc": 0.4,
         "n_seizures": 12},
    ]


# ------------------------------------------------------------------ #
#  assembly logic -- no image backend needed
# ------------------------------------------------------------------ #

def test_survivors_positive_and_significant():
    surv = report._survivors(_rows(surviving=True))
    assert len(surv) == 1 and surv[0]["feature"] == "line_length"
    assert report._survivors(_rows(surviving=False)) == []


def test_scale_table_highlights_survivor():
    html = report._scale_table(_rows(surviving=True))
    assert "<table" in html and "line_length" in html
    assert "background:#fff1f0" in html            # survivor row highlighted


def test_report_assembles_without_backend(tmp_path, monkeypatch):
    # isolate the HTML assembly from PNG rendering + the live trace
    monkeypatch.setattr(figures, "render_run_figures", lambda *a, **k: [])
    import src.preictal.engine as engine
    monkeypatch.setattr(engine, "trace_one_seizure", lambda *a, **k: None)

    run = {"id": 7, "scope": "daily", "period_start": "2025-03-26",
           "period_end": "2025-03-26", "n_seizures": 41, "n_scales": 2,
           "status": "done"}
    path = report.render_run_report(None, run, _rows(surviving=True),
                                    {"preictal": {}}, str(tmp_path))
    assert os.path.isfile(path)
    doc = open(path, encoding="utf-8").read()
    assert "Pre-ictal sweep report — run 7" in doc
    assert "survive the surrogate null" in doc      # headline (survivor branch)
    assert "<table" in doc and "line_length" in doc  # scale table present
    assert "All scales" in doc


def test_report_null_headline(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "render_run_figures", lambda *a, **k: [])
    import src.preictal.engine as engine
    monkeypatch.setattr(engine, "trace_one_seizure", lambda *a, **k: None)
    run = {"id": 8, "scope": "daily", "period_start": "2025-03-26",
           "status": "done", "n_scales": 2}
    path = report.render_run_report(None, run, _rows(surviving=False),
                                    {"preictal": {}}, str(tmp_path))
    doc = open(path, encoding="utf-8").read()
    assert "No pre-ictal time-scale survived" in doc


# ------------------------------------------------------------------ #
#  PNG render -- best-effort, must not crash if kaleido is absent
# ------------------------------------------------------------------ #

def test_render_run_figures_degrades_gracefully(tmp_path):
    paths = figures.render_run_figures({"id": 1, "period_start": "2025-03-26"},
                                       _rows(surviving=True), str(tmp_path))
    assert isinstance(paths, list)
    for p in paths:                                 # any produced file is real
        assert os.path.isfile(p) and os.path.getsize(p) > 0
