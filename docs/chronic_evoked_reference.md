# Chronic Evoked Analyzer — reference for a new dashboard tab

> Captured from the external MATLAB toolkit (NOT in this repo, so the cloud
> planner can't read it):
> `D:\code\Stimulation-Telemetry-Modulation-NeuroEngineering-Toolkit\daqSignalGenerator\src\ui\chronic\`
> (`ChronicAnalyzer.m`, `ChronicPlotter.m`, `ChronicDataLoader.m`,
> `ChronicTabState.m`, `ChronicTabController.m` ~15k lines), launched by
> `runDAQApp('evoked')` → `EvokedResponseAnalyzer` → its "Chronic" tab.
>
> Goal: add an equivalent **Chronic evoked** tab to the qc_monitor Dash
> dashboard, reading the **already-stored** `evoked_features` (no MATLAB, no
> recompute).

## What the MATLAB analyzer does

**Concept.** Concatenate **every stim-evoked epoch across an animal's whole
implant period** (many recordings) into one long, datetime-stamped series,
then study how the evoked response and its features drift over days/weeks.

**Data model (MATLAB).** Loads many `*_evoked.mat` files; each has
`evokedData [samples × epochs]` + per-epoch datetimes. Concatenated into
`AllTraces [samples × totalEpochs]`, `AllDatetimes [1 × totalEpochs]`,
`TimeAxis [1 × samples]` (ms). Analysis window default **−100…500 ms** around
each stim; optional **baseline correction** and a **pre-stimulus (passive)**
feature mode (same feature computed on the pre-stim portion).

**Views / plots** (`ChronicPlotter.m`):
- **Feature timeline (centerpiece)** — pick a feature → scatter of that
  feature **vs absolute datetime**, points **rainbow-colored by time**, with
  **trend line + moving average + regression** and trend stats.
  (`plotFeatureTimeline`, `plotWithMovingAverage`, `addRegressionLine`,
  `addTemporalColorbar`.) Clicking a point selects that epoch.
- **Evoked overlay** — all epoch traces overlaid, colored by time
  progression (`plotEvokedOverlay`); **per-session mean traces**
  (`plotMeanResponses`); single-trace + **context view** (N epochs
  before/after a selected one).
- **Stim ↔ evoked correlation** scatter w/ regression (`plotCorrelation`).
- **Per-epoch inspector** — EEG + spectrogram + PSD for a clicked epoch
  (`plotEEGWithSpectrogram`).

**Analysis algorithms** (`ChronicAnalyzer.m`):
- **Temporal trends** — linear + quadratic fit, slope, r, p, mean, std,
  **CV**, range (`analyzeTemporalTrends`).
- **Per-day / per-session segment means ± std** (`analyzeTimeSegments`,
  default 24 h; `computeSessionStatistics`, `computeSessionMeans`).
- **Stim vs evoked correlations** — peak/trough/peak-to-peak
  (`computeCorrelations`, `computeMultipleCorrelations`).
- **Response stability over time** — rolling CV + template correlation to a
  running median (`analyzeResponseStability`).
- **Outlier detection** — z-score / IQR / MAD (`detectOutliers`).

**State/controls** (`ChronicTabState.m`): feature dropdown, display mode
(overlay/individual), scatter display mode, analysis window (start/end ms),
baseline-correction toggle, context-view toggle + N, artifact/ictal overlays,
spectrogram channel.

## How it maps onto qc_monitor (reuse, don't reinvent)

The dashboard already **stores** exactly this data, so the tab reads the DB
instead of recomputing:

- **`evoked_features` table** (`src/db/schema.py`): one row per stim epoch
  per file — `file_id, epoch_index, epoch_time_sec, <feature columns>,
  is_artifact, is_ictal, version_id`. The file's `chunk_datetime`
  (`processed_files`) + `epoch_time_sec` give each epoch an **absolute
  time** → the chronic timeline.
- **`store.get_evoked_feature_timeseries(feature_name, session_dir, version_id,
  hours)`** (`store.py:667`) — returns `chunk_datetime, epoch_index,
  epoch_time_sec, value, is_artifact, is_ictal` ordered by time. **But it's
  scoped to ONE `session_dir`.** Chronic needs the whole animal across all
  its sessions → **add a per-animal variant** (e.g.
  `get_evoked_feature_timeseries_for_animal(animal_id, feature_name, …)`)
  joining `session_config.channel_names LIKE '%"<animal>%'` the way
  `mass_analyze.pending_files_for_animal` (`src/utils/mass_analyze.py:301`)
  already does, ordered by `chunk_datetime, epoch_index`.
- **Feature list** — `EVOKED_FEATURE_COLS` + `EVOKED_FEATURE_LABELS`
  (`src/dashboard/data_helpers.py:83`) drive feature dropdowns app-wide;
  the Chronic feature picker reads from them (so it stays in sync). The
  validated whitelist is `store._EVOKED_FEATURE_COLS` (`store.py:626`).
- **Animal / electrode / session pickers** — `data_helpers`
  (`session_dropdown_options`), `animal` helpers, and
  `mass_analyze.animal_channel_index` for the per-animal channel.
- **Patterns to mirror** — `src/dashboard/tabs/session_compare.py` (already
  compares evoked features across sessions; closest sibling) and
  `tabs/evoked.py` (feature checklist from `EVOKED_FEATURE_COLS`). Tab wiring
  = `NAV_GROUPS` + `render_tab` dispatch + `register_callbacks` in
  `src/dashboard/app.py` (see how `training`/`annotations` are wired).
- **Components** — `components.card/button/empty_state/DARK_TABLE_STYLE`,
  `design.py` tokens, Plotly (`go.Scattergl`, color-by-time via a continuous
  colorscale on `chunk_datetime`).
- **Slow-gamma features** — the live band-power features added in
  `src/utils/filters.py` (`band_power`, `SLOW_GAMMA_BAND`, `epoch_band_power`)
  are natural additional chronic metrics, but those are computed live, not in
  `evoked_features`; the chronic tab's stored-feature timeline won't include
  them unless they're persisted (out of scope for v1).

## Suggested dashboard scope (for the planner to refine)

**Pure core (testable):** `src/utils/chronic.py` — trend fit (linear +
quadratic, slope/CV/r), per-day & per-session aggregation (mean ± std, n),
moving average, outlier mask (z/IQR/MAD), stim↔evoked correlation. Mirror
`ChronicAnalyzer.m` exactly; unit-test like `apply_event_types` /
`grade_attempt`.

**Tab `src/dashboard/tabs/chronic_evoked.py`:**
- Pickers: animal → electrode/channel → feature (from `EVOKED_FEATURE_COLS`),
  date range, artifact/ictal include toggles, "per-session mean" toggle.
- **Feature timeline** plot: feature vs absolute datetime, points colored by
  time (continuous colorbar), with trend line + moving average + regression
  stats in the status line; per-session mean ± std overlay/band.
- **Per-session / per-day table**: n epochs, mean, std, CV, trend slope.
- Optional: stim↔evoked correlation scatter; click a point → deep-link to
  Video Review / Evoked waveform for that file (reuse the
  `lfp-to-video-bridge` deep-link pattern in `video.py` / `event_verification`).
- Place under the **Analysis** nav group (next to Evoked features / Video
  review).

**Out of scope v1:** the MATLAB per-epoch spectrogram/PSD inspector and the
raw-trace overlay (the dashboard stores features, not the full evoked traces;
revisit if raw evoked traces get persisted).

## Open questions for the planner
- Timeline x-axis: true absolute datetime (`chunk_datetime + epoch_time_sec`)
  vs "hours since first recording" (MATLAB uses both).
- Aggregation default: per-epoch scatter vs per-session/day mean (perf — an
  animal can have very many epochs; may need server-side decimation like the
  LFP `envelope()` path).
- Whether to add a per-animal store query (recommended) vs aggregating
  per-session calls client-side.
