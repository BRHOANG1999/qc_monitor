# ML/AI Proof-of-Concept Notebooks

Four PoC notebooks exploring ML directions for the QC Monitor pipeline.
All notebooks read from production data (monitor.db, .mat files, .mp4 videos)
in **read-only** mode. No production code is modified.

## Setup

```bash
# From repo root — install ML deps (separate from service deps)
pip install -r notebooks/requirements-ml.txt

# Launch Jupyter
cd notebooks
jupyter notebook
```

## Notebooks

| # | Notebook | Question | Key deps |
|---|----------|----------|----------|
| 1 | `01_seizure_classifier.ipynb` | Can a CNN beat the threshold-based seizure detector? | torch, scikit-learn |
| 2 | `02_evoked_anomaly.ipynb` | Can we auto-flag drifted evoked responses? | scikit-learn, umap-learn |
| 3 | `03_video_behavior.ipynb` | Can we extract useful behavior from _v1.mp4? | opencv-python |
| 4 | `04_llm_summary_chat.ipynb` | Are Claude-powered summaries + NL queries useful? | anthropic |

## Shared code

`_common.py` provides shared loaders that delegate to production modules:
- `load_db()` — read-only SQLite connection
- `load_evoked_feature_matrix()` — 26-feature DataFrame from evoked_features table
- `load_lfp_window()` — time-windowed LFP from .mat files
- `load_video_frames()` — frame extraction from .mp4 via OpenCV
- `db_schema_text()` — SQL schema string for LLM prompts

## Success criteria

Each notebook ends with a go/no-go verdict cell:
- **Ship** — integrate into the pipeline
- **Iterate** — promising but needs more work
- **Drop** — not worth pursuing
