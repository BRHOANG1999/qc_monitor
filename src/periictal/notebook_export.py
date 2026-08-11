"""Export a peri-ictal lens's built analysis as a self-contained Jupyter notebook.

The peri-ictal analysis is pure + Dash-free (``matrix.build_matrix`` ->
``sliding_auc`` / ``forecast`` / ``trendtest`` / ``embed``), so a lens's result
can be reproduced outside the dashboard. This module FREEZES the built matrix
(the ``full`` DataFrame) to a gzip pickle and generates a ``.ipynb`` whose cells
re-run the SAME pure functions and re-plot with matplotlib. Notebook + data are
bundled into one ZIP.

Reproduces anywhere the ``qc_monitor`` repo is importable -- no database, no
share, no re-scoring drift (the data is frozen in time). Only the analysis code
is shared (imported from ``src.periictal.*``), so a notebook never drifts from
what the tool actually computes.
"""

from __future__ import annotations

import io
import json
import zipfile

DATA_FILE = "matrix.pkl.gz"


# --------------------------------------------------------------------- #
#  Minimal nbformat builders (no nbformat dependency)
# --------------------------------------------------------------------- #

def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


def _notebook(cells: list) -> dict:
    for i, c in enumerate(cells):          # nbformat 4.5 requires a cell id
        c.setdefault("id", f"cell{i}")
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


# --------------------------------------------------------------------- #
#  Shared preamble
# --------------------------------------------------------------------- #

def _preamble_cells(meta: dict) -> list:
    prov = "\n".join(f"- **{k}**: {v}" for k, v in meta.items())
    return [
        _md(f"# Peri-ictal — {meta.get('lens', 'analysis')}\n\n"
            "Self-contained reproducible export. This notebook plus the frozen "
            f"`{DATA_FILE}` reproduce the analysis with **no database or share "
            "access** — it re-runs the SAME pure functions the dashboard uses, so "
            "it never drifts from the tool.\n\n"
            "**Requirements:** the `qc_monitor` repo importable on `sys.path` (for "
            "`src.periictal.*`), plus numpy / pandas / scipy / scikit-learn / "
            "matplotlib.\n\n"
            f"## Provenance\n{prov}"),
        _code(
            "import os, sys\n"
            "import numpy as np, pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "# The qc_monitor repo must be importable for src.periictal.*.\n"
            "# Set QC_MONITOR_REPO, or run this notebook from the repo root.\n"
            "REPO = os.environ.get('QC_MONITOR_REPO', os.getcwd())\n"
            "if REPO not in sys.path:\n"
            "    sys.path.insert(0, REPO)\n"
            f"full = pd.read_pickle('{DATA_FILE}', compression='gzip')\n"
            "print('matrix:', full.shape, '|', int(full['seizure_idx'].nunique()), "
            "'seizures')"),
    ]


# --------------------------------------------------------------------- #
#  Per-lens cells
# --------------------------------------------------------------------- #

def _cells_slidingauc(p: dict) -> list:
    return [
        _md("## Sliding-window ROC-AUC\n\nThe fixed 30-min preictal window vs N "
            "sampled interictal windows across a lookback band; scored per "
            "feature per seizure, then aggregated across seizures."),
        _code(
            "from src.periictal.sliding_auc import sliding_window_auc\n"
            "from src.periictal import config as cfg\n"
            f"features = cfg.metrics_for_variant('{p['variant']}')\n"
            f"res = sliding_window_auc(full, features, n_windows={p['n_windows']},\n"
            f"                         band_lo={p['band_lo']}, band_hi={p['band_hi']})\n"
            "print('scored', res['n_seizures_used'], 'seizures')\n"
            "print('group top-5:', res['group_top5'])"),
        _md("### Group top features (mean AUC across seizures)"),
        _code(
            "ranked = [f for f in res['group_ranked'] "
            "if np.isfinite(res['group'][f])][:12][::-1]\n"
            "top = set(res['group_top5'])\n"
            "fig, ax = plt.subplots(figsize=(7, 5))\n"
            "ax.barh(range(len(ranked)), [res['group'][f] for f in ranked],\n"
            "        color=['#5e7ce2' if f in top else '#888' for f in ranked])\n"
            "ax.set_yticks(range(len(ranked))); ax.set_yticklabels(ranked, fontsize=8)\n"
            "ax.axvline(0.5, ls='--', c='#aaa'); ax.set_xlim(0.5, 1.0)\n"
            "ax.set_xlabel('mean AUC (>=.5)'); ax.set_title('Group top features')\n"
            "plt.tight_layout(); plt.show()"),
        _md("### AUC by feature × window (group mean)"),
        _code(
            "feats = res['features']; ranked = res['group_ranked']\n"
            "# group AUC matrix = mean across seizures of each per-seizure "
            "feature×window matrix\n"
            "mats = [np.asarray(res['per_seizure'][s]['auc']) "
            "for s in sorted(res['per_seizure'])]\n"
            "group_auc = np.nanmean(np.stack(mats), axis=0)\n"
            "Z = group_auc[[feats.index(f) for f in ranked]]\n"
            "fig, ax = plt.subplots(figsize=(8, 7))\n"
            "im = ax.imshow(Z, aspect='auto', vmin=0.5, vmax=1.0, cmap='viridis')\n"
            "ax.set_yticks(range(len(ranked))); ax.set_yticklabels(ranked, fontsize=7)\n"
            "ax.set_xticks(range(len(res['win_labels'])))\n"
            "ax.set_xticklabels(res['win_labels'], rotation=45, fontsize=7)\n"
            "ax.set_xlabel('interictal window (before onset)')\n"
            "fig.colorbar(im, ax=ax, label='AUC'); plt.tight_layout(); plt.show()"),
        _md("### Window-AUC trajectory (group #1 feature)"),
        _code(
            "feat = res['group_top5'][0]; fi = res['features'].index(feat)\n"
            "x = np.asarray(res['offsets']) / 3600.0\n"
            "fig, ax = plt.subplots(figsize=(8, 5)); rows = []\n"
            "for s in sorted(res['per_seizure']):\n"
            "    row = np.asarray(res['per_seizure'][s]['auc'])[fi]; rows.append(row)\n"
            "    ax.plot(x, row, marker='o', ms=3, lw=1, alpha=0.45)\n"
            "med = np.nanmedian(np.vstack(rows), axis=0)\n"
            "ax.plot(x, med, 'k-', lw=3, label='group median')\n"
            "ax.axhline(0.5, ls='--', c='#aaa'); ax.invert_xaxis()\n"
            "ax.set_xlabel('time before onset (h)'); ax.set_ylabel('AUC (auc_norm)')\n"
            "ax.set_title('Window AUC vs lead time · ' + feat); ax.legend()\n"
            "plt.tight_layout(); plt.show()"),
    ]


def _cells_pdfcdf(p: dict) -> list:
    return [
        _md("## Preictal vs interictal (Chang et al. 2026)\n\nPreictal (0–30 min "
            "before onset) vs interictal (60–90 min); per-feature rank AUC and "
            "distributions."),
        _code(
            "from src.periictal import forecast as fc\n"
            "from src.periictal import config as cfg\n"
            f"features = cfg.metrics_for_variant('{p['variant']}')\n"
            "lab = fc.label_classes(full)\n"
            "scan = fc.scan_features(lab, features, n_perm=200)\n"
            "pd.DataFrame(scan)[['feature', 'auc_norm', 'direction', 'p', 'q',\n"
            "                    'n_pre', 'n_inter']].head(12)"),
        _md("### Per-feature discrimination (rank AUC)"),
        _code(
            "top = scan[:20][::-1]\n"
            "fig, ax = plt.subplots(figsize=(7, 6))\n"
            "ax.barh(range(len(top)), [r['auc_norm'] for r in top],\n"
            "        color=['#e8433f' if (np.isfinite(r['q']) and r['q'] < 0.05) "
            "else '#888' for r in top])\n"
            "ax.set_yticks(range(len(top)))\n"
            "ax.set_yticklabels([r['feature'] for r in top], fontsize=7)\n"
            "ax.axvline(0.5, ls='--', c='#aaa'); ax.set_xlim(0.5, 1.0)\n"
            "ax.set_xlabel('AUC (norm)'); ax.set_title('Per-feature AUC (red = q<0.05)')\n"
            "plt.tight_layout(); plt.show()"),
        _md("### PDF / CDF of the top feature"),
        _code(
            "feat = scan[0]['feature']; pc = fc.pdf_cdf(lab, feat)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))\n"
            "if pc['pre_kde'] is not None:\n"
            "    a1.plot(pc['grid'], pc['pre_kde'], color='#e8433f', "
            "label='preictal (n=%d)' % pc['n_pre'])\n"
            "    a1.fill_between(pc['grid'], pc['pre_kde'], color='#e8433f', alpha=0.2)\n"
            "if pc['inter_kde'] is not None:\n"
            "    a1.plot(pc['grid'], pc['inter_kde'], color='#3b7fc4', "
            "label='interictal (n=%d)' % pc['n_inter'])\n"
            "    a1.fill_between(pc['grid'], pc['inter_kde'], color='#3b7fc4', alpha=0.2)\n"
            "a1.set_title('PDF · ' + feat); a1.legend()\n"
            "a2.plot(pc['pre_cdf_x'], pc['pre_cdf_y'], color='#e8433f')\n"
            "a2.plot(pc['inter_cdf_x'], pc['inter_cdf_y'], color='#3b7fc4')\n"
            "a2.set_title('CDF · %s (AUC %.3f)' % (feat, pc['auc_norm']))\n"
            "plt.tight_layout(); plt.show()"),
    ]


def _cells_trend(p: dict) -> list:
    return [
        _md("## Per-seizure trend test\n\nSpearman rho of a feature vs "
            "time-to-onset, one per seizure (the honest per-seizure unit)."),
        _code(
            "from src.periictal import trendtest as tt\n"
            f"feat = '{p['feature']}'\n"
            "per_sz = tt.per_seizure_trend(full, feat)\n"
            "tdf = pd.DataFrame(per_sz); tdf"),
        _md("### Per-seizure rho"),
        _code(
            "rho = tdf['rho'].to_numpy() if 'rho' in tdf else np.full(len(tdf), np.nan)\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "ax.bar(range(len(rho)), rho, color='#5e7ce2'); ax.axhline(0, c='#aaa')\n"
            "ax.set_xlabel('seizure'); ax.set_ylabel('Spearman rho')\n"
            "ax.set_title(feat + ': per-seizure trend vs time-to-onset')\n"
            "plt.tight_layout(); plt.show()"),
    ]


def _cells_embedding(p: dict) -> list:
    return [
        _md("## Embedding\n\nPCA/UMAP of the pre-onset stimuli, coloured by time "
            "to onset. (UMAP invents clusters from noise — a figure, not a test.)"),
        _code(
            "from src.periictal.embed import embed\n"
            "from src.periictal import config as cfg\n"
            "pre = full[full['phase'] == 'pre'].reset_index(drop=True)\n"
            f"res = embed(pre, method='{p['method']}', cap=cfg.INTERACTIVE_POINT_CAP)\n"
            "sub = pre.iloc[res['rows']].reset_index(drop=True)\n"
            "emb = np.asarray(res['emb']); print('embedded', emb.shape)"),
        _md("### Coloured by time to onset"),
        _code(
            "fig, ax = plt.subplots(figsize=(7, 6))\n"
            "sc = ax.scatter(emb[:, 0], emb[:, 1], c=sub['time_to_onset_sec'],\n"
            "                s=6, cmap='viridis')\n"
            "fig.colorbar(sc, ax=ax, label='time to onset (s)')\n"
            f"ax.set_title('{p['method'].upper()} embedding')\n"
            "ax.set_xlabel('dim 1'); ax.set_ylabel('dim 2')\n"
            "plt.tight_layout(); plt.show()"),
    ]


_LENSES = {
    "periictal_slidingauc": ("Sliding-window ROC-AUC", _cells_slidingauc),
    "periictal_pdfcdf": ("Preictal vs interictal", _cells_pdfcdf),
    "periictal_trend": ("Per-seizure trend test", _cells_trend),
    "periictal_embedding": ("Embedding", _cells_embedding),
}


def supported(lens_id: str) -> bool:
    return lens_id in _LENSES


def _readme(meta: dict) -> str:
    return ("# Reproducible peri-ictal export\n\n"
            "1. `pip install numpy pandas scipy scikit-learn matplotlib`\n"
            "2. Make the `qc_monitor` repo importable: run the notebook from the "
            "repo root, or set `QC_MONITOR_REPO=/path/to/qc_monitor`.\n"
            "3. Open `analysis.ipynb` and Run All.\n\n"
            f"Data frozen in `{DATA_FILE}` (gzip pickle of the built matrix).\n")


def build_export(lens_id: str, full_df, meta: dict, params: dict):
    """Return (zip_bytes, filename) bundling the notebook + frozen matrix for
    *lens_id*. Raises ValueError for an unsupported lens."""
    assert lens_id and full_df is not None, "lens_id and matrix required"
    entry = _LENSES.get(lens_id)
    if entry is None:
        raise ValueError(f"no notebook exporter for lens {lens_id!r}")
    label, cell_fn = entry
    m = dict(meta or {})
    m["lens"] = label
    nb = _notebook(_preamble_cells(m) + cell_fn(params or {}))

    pbuf = io.BytesIO()
    full_df.to_pickle(pbuf, compression="gzip")
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("analysis.ipynb", json.dumps(nb, indent=1))
        z.writestr(DATA_FILE, pbuf.getvalue())
        z.writestr("README.md", _readme(m))
    animal = (meta or {}).get("animal", "") or "export"
    return zbuf.getvalue(), f"{lens_id}_{animal}.zip"
