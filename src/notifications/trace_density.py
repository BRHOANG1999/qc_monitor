"""Memory-safe FULL-DENSITY trace overlay.

Draws *every* evoked trace as an individual translucent line without ever holding
all traces in RAM. Each trace is rasterized into a fixed-size pixel buffer as it
streams by (min/max row per column, so vertical excursions render as connected
strokes); the buffer counts how many traces pass through each pixel. Rendering
then maps that count to alpha exactly like plotting N lines at ``alpha=a0``
(``1-(1-a0)**count``) in one bright colour over the dark panel -- so a dense band
saturates into a smear while a lone outlier trace still shows at ``a0`` (the whole
point: the few traces that emerge above the smear stay visible).

Resolution/pixel-aware by construction: the signal is never filtered or
decimated, only mapped to pixels (an oscilloscope-style min/max-per-column
envelope preserves the true extrema in each column). Memory is O(W*H) per band,
independent of the trace count, so it is daemon-safe at 10^5+ traces.
"""

from __future__ import annotations

import numpy as np

_DEFAULT_W = 520
_DEFAULT_H = 320


class LineDensity:
    """Fixed-size additive line-rasterization buffer for ONE panel (one decile).

    ``xlim``/``ylim`` are the data ranges the panel spans (ms, amplitude); traces
    outside are clipped. Call ``add`` per trace, then ``counts`` / ``rgba``."""

    def __init__(self, xlim, ylim, width: int = _DEFAULT_W, height: int = _DEFAULT_H):
        self.x0, self.x1 = float(xlim[0]), float(xlim[1])
        self.y0, self.y1 = float(ylim[0]), float(ylim[1])
        assert self.x1 > self.x0, "xlim must be increasing"
        assert self.y1 > self.y0, "ylim must be increasing"
        self.W, self.H = int(width), int(height)
        # +1 row so a top/bottom-clamped max can close its column span via cumsum.
        self._delta = np.zeros((self.H + 1, self.W), dtype=np.float64)
        self.n = 0

    def add(self, x_ms, y) -> None:
        """Rasterize one trace. ``x_ms`` must be finite+ascending (the evoked time
        axis is); NaNs in ``y`` (e.g. the blanked stim-artifact window) break the
        stroke there rather than bridging across the gap."""
        x = np.asarray(x_ms, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape or x.size == 0:
            return
        m = np.isfinite(x) & np.isfinite(y) & (x >= self.x0) & (x <= self.x1)
        if not m.any():
            return
        xw, yw = x[m], y[m]
        col = np.clip(((xw - self.x0) / (self.x1 - self.x0) * (self.W - 1)),
                      0, self.W - 1).astype(np.intp)
        # row 0 = top (y1); y increases downward on screen.
        row = np.clip(((self.y1 - yw) / (self.y1 - self.y0) * (self.H - 1)),
                      0, self.H - 1).astype(np.intp)
        # col is non-decreasing (x ascending), so unique+reduceat gives, per pixel
        # column, the min and max row this trace spans -> a filled vertical segment.
        uniq, idx = np.unique(col, return_index=True)
        rmin = np.minimum.reduceat(row, idx)
        rmax = np.maximum.reduceat(row, idx)
        # Additive vertical segments via a difference array + later cumsum down rows.
        np.add.at(self._delta, (rmin, uniq), 1.0)
        np.add.at(self._delta, (rmax + 1, uniq), -1.0)
        self.n += 1

    def counts(self) -> np.ndarray:
        """(H, W) int-ish array: how many traces pass through each pixel."""
        return np.cumsum(self._delta, axis=0)[:self.H]

    def rgba(self, color_rgb, a0: float = 0.30) -> np.ndarray:
        """(H, W, 4) float image: the density rendered as ``color_rgb`` lines at
        per-trace opacity ``a0``. alpha = 1-(1-a0)**count, so 1 trace -> a0 (still
        visible), many -> opaque. Meant for ``imshow(..., origin='upper')``."""
        c = self.counts()
        alpha = 1.0 - np.power(1.0 - float(a0), c)
        img = np.zeros((self.H, self.W, 4), dtype=np.float64)
        img[..., 0], img[..., 1], img[..., 2] = color_rgb[0], color_rgb[1], color_rgb[2]
        img[..., 3] = alpha
        return img

    def extent(self):
        """(x0, x1, y0, y1) for ``imshow(extent=...)`` with ``origin='upper'``."""
        return (self.x0, self.x1, self.y0, self.y1)

    def column_quantiles(self, qs):
        """Per-column amplitude quantiles of the rasterized cloud -> decile envelopes
        laid over the density. *qs* is a list of cumulative fractions in (0,1);
        returns (len(qs), W) of y-values (NaN for empty columns) and the column
        x-centres (W,). Derived from the occupancy buffer (memory-safe, no traces
        held): counts per (row,col) approximate the trace-value histogram of each
        time column, so these are density deciles -- exact where traces are locally
        flat, slightly middle-biased where a trace is steep within one column."""
        qs = np.atleast_1d(np.asarray(qs, dtype=np.float64))
        assert qs.ndim == 1 and qs.size >= 1, "qs must be a non-empty 1-D list"
        assert np.all((qs > 0.0) & (qs < 1.0)), "quantiles must be in (0, 1)"
        c = self.counts()                                  # (H, W); row 0 = top (y1)
        tot = c.sum(axis=0)                                # (W,)
        out = np.full((qs.size, self.W), np.nan, dtype=np.float64)
        xc = self.x0 + (np.arange(self.W) + 0.5) / self.W * (self.x1 - self.x0)
        if not np.any(tot > 0):
            return out, xc
        # CDF from the BOTTOM up (low y -> high y) so quantile q = value with fraction
        # q of the mass below it: q=0.1 -> low amplitude, q=0.9 -> high amplitude.
        frac = np.cumsum(c[::-1, :], axis=0) / np.where(tot > 0, tot, 1.0)
        rows = np.arange(self.H, dtype=np.float64)
        y_of_row = self.y1 - rows / max(1, self.H - 1) * (self.y1 - self.y0)
        for qi in range(qs.size):
            ge = frac >= qs[qi]                            # first crossing from bottom
            k = np.argmax(ge, axis=0)                      # rows counted up from bottom
            ok = (tot > 0) & ge.any(axis=0)
            yv = y_of_row[self.H - 1 - k]
            yv[~ok] = np.nan
            out[qi] = yv
        return out, xc


def robust_ylim(lo_hi_samples, pad: float = 0.08):
    """Symmetric-ish y-range from streamed per-trace (min, max) robust extrema.

    Pass an (N, 2) array of each trace's [p-low, p-high] windowed amplitude; returns
    (y0, y1) at the 1st/99th percentile of those extrema, padded. Keeps one wild
    trace from blowing the axis while still showing genuine outliers."""
    a = np.asarray(lo_hi_samples, dtype=np.float64)
    a = a[np.isfinite(a).all(axis=1)]
    if a.size == 0:
        return (-1.0, 1.0)
    lo = np.percentile(a[:, 0], 1.0)
    hi = np.percentile(a[:, 1], 99.0)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return (lo - pad * span, hi + pad * span)
