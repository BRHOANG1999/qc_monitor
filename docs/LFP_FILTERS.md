# LFP filter + PSD controls

Each LFP-bearing tab in the dashboard has a filter strip the user can
adjust on the fly. The math lives in `src/utils/filters.py`; tab
modules just thread the settings through.

## Where the controls live

| Tab | Where | Controls |
|---|---|---|
| **Analysis → LFP Browser** | Strip above the trace | Preset · HP · LP · Notch · Smooth · Show PSD · Apply |
| **Analysis → Video Review** | Strip above the LFP trace | HP · LP · Notch · Smooth · Apply |
| **Analysis → Evoked Waveforms** | Inline with the file dropdown | Smooth |
| **Sessions → Session Compare** | Inline with the session dropdowns | Smooth |

The two raw-LFP tabs share the full HP / LP / Notch cascade. The
mean-evoked tabs only expose smoothing because the underlying data is
already averaged across epochs.

## The strip

* **Preset** (LFP Browser only): writes the matching HP + LP into
  the number inputs. Choices:
  * `Raw (no filter)` — HP / LP both 0
  * `Delta` (1–4 Hz), `Theta` (4–8), `Alpha` (8–13), `Beta` (13–30),
    `Gamma` (30–100), `Spike band` (300–3000)
  * `Custom` — leaves inputs alone
* **HP (Hz)** — Butterworth order 4 highpass, zero-phase. `0` = off.
* **LP (Hz)** — Butterworth order 4 lowpass, zero-phase. `0` = off.
* **Notch** — `Off / 50 Hz / 60 Hz`. Q = 30.
* **Smooth (ms)** — Gaussian smoothing. `0` = off.
* **Show PSD** (LFP Browser only) — toggle a Welch PSD panel
  beneath the trace. Log Y axis; X clipped to `min(1000 Hz, fs/2)`.
  Dotted vertical guides at 50 / 60 / 120 / 180 Hz.
* **Apply** — single submission point. Edits stay local until you
  click Apply so typing into a number input doesn't trigger
  half-second filter passes per keystroke.

## Behavior

The cascade order is **HP → LP → Notch → Smoothing**. Stages set to
`0` / `None` / `Off` are no-ops.

Zero-phase filtering: HP / LP / Notch all use `scipy.signal.sosfiltfilt`
(forward + backward). Phase response is flat, but the first/last
`~3 × order / cutoff` seconds carry transient ringing — visible if you
HP at 0.1 Hz on a 60 s clip. Crop the edges in your analysis if it
matters.

NaN-safe: stim-blank windows in Video Review are NaN runs > 50 ms.
`apply_filter` linearly interpolates those before filtering and
restores them to NaN afterward, so the filter never sees a NaN and
the trace still renders as a break.

Caching: filter results are LRU-cached per `(file_path, settings)` so
zooming and panning don't re-run the filter. Cap is 4 entries. Drop
the cache for a specific file with `evict_filtered(file_path)`; drop
everything with `evict_filtered()`.

## Tests

```
python -m tests.test_filters     # or: pytest tests/test_filters.py -q
```

11 tests cover the cascade math (synthetic tones), PSD on noise + a
known tone, NaN gap handling (short interpolated, long preserved),
and cache reuse.

## Where to extend

* Adding a new preset: edit `_LFP_PRESETS` in
  `src/dashboard/app.py` (search for the dict).
* Adding a new filter stage: add the math in
  `src/utils/filters.apply_filter`, update
  `filter_settings_key` so the cache key includes it, and thread the
  new control through the two callbacks (`load_lfp` +
  `_update_lfp` in `video.py`).
* Adding the strip to another tab: import `apply_filter` from
  `src/utils/filters`, wire a small Apply button + a few inputs into
  the existing render callback.
