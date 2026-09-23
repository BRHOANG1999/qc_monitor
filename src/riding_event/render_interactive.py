"""Interactive per-event inspector for the riding-event residuals.

Writes a self-contained HTML page (Plotly from CDN + a small click handler): a
clickable residual ERP-image on top (one row per event-flagged epoch, time order)
and, below, the individual residual + raw trace of whichever event you click.
Opens in a browser, no server needed -- the complement to the static PNGs when you
want to inspect each flagged event one at a time.
"""

from __future__ import annotations

import json
import logging

import numpy as np

logger = logging.getLogger("qc_monitor.riding_event.render_interactive")

_MAX_EVENTS = 500          # NASA Rule 3: bound the embedded payload.


def _event_payload(res: dict, *, win_ms=(-10.0, 200.0), n_time: int = 600) -> dict:
    """Compact JSON-able payload: per event-flagged epoch, its decimated residual
    and raw trace over *win_ms*, plus the ERP-image matrix and metadata."""
    ev = np.flatnonzero(res["event_mask"])
    assert res["traces"].ndim == 2, "traces must be 2-D"
    if ev.size > _MAX_EVENTS:                          # keep the strongest events
        order = np.argsort(res["energy"][ev])[::-1][:_MAX_EVENTS]
        ev = np.sort(ev[order])
    t = np.asarray(res["time_ms"], dtype=np.float64)
    m = (t >= win_ms[0]) & (t <= win_ms[1])
    tt = t[m]
    idx = np.linspace(0, tt.size - 1, min(int(n_time), tt.size)).astype(int) \
        if tt.size else np.empty(0, dtype=int)
    resid = res["resid"][np.ix_(ev, np.flatnonzero(m))][:, idx] if ev.size else \
        np.empty((0, 0))
    raw = res["traces"][np.ix_(ev, np.flatnonzero(m))][:, idx] if ev.size else \
        np.empty((0, 0))
    _times = res.get("times")
    times = np.asarray(_times, dtype=np.float64) if _times is not None \
        else np.empty(0)
    return {
        "animal": res["animal"], "channel": res["channel"],
        "time_ms": [round(float(x), 3) for x in tt[idx]],
        "resid": [[round(float(v), 4) for v in row] for row in resid],
        "raw": [[round(float(v), 4) for v in row] for row in raw],
        "epoch": [int(j) for j in ev],
        "stim_time": [round(float(times[j]), 3) if j < times.size else None
                      for j in ev],
        "energy": [round(float(e), 5) for e in res["energy"][ev]],
    }


def write_event_inspector(res: dict, out_html: str) -> str:
    """Write the interactive inspector HTML for *res* (a Prong-A result). Returns
    the path. Degrades to a short 'no events' page when nothing was flagged."""
    assert res and out_html, "res and out_html required"
    payload = _event_payload(res)
    html = _TEMPLATE.replace("/*__DATA__*/null", json.dumps(payload))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Riding-event inspector</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root { color-scheme: dark; }
  body { background:#1e1e2f; color:#f0f0f5; font:14px/1.4 system-ui,sans-serif;
         margin:0; padding:16px; }
  h1 { font-size:18px; margin:0 0 4px; }
  p.hint { color:#9a9ab0; margin:0 0 12px; }
  #erp { width:100%; height:380px; }
  #detail { width:100%; height:340px; }
  .meta { color:#9a9ab0; font-size:13px; margin:6px 2px; }
</style></head>
<body>
<h1 id="title">Riding-event inspector</h1>
<p class="hint">Click any row in the residual image to inspect that event's
individual residual (orange) and raw trace (blue) below.</p>
<div id="erp"></div>
<div class="meta" id="meta"></div>
<div id="detail"></div>
<script>
const DATA = /*__DATA__*/null;
const BG="#1e1e2f", PANEL="#26263a", TX="#f0f0f5", MU="#9a9ab0";
const EVT="#ff7043", RAW="#5e7ce2";
document.getElementById("title").textContent =
  DATA.animal+" "+DATA.channel+" — "+DATA.epoch.length+" event residuals";
function baseLayout(extra){
  return Object.assign({paper_bgcolor:BG, plot_bgcolor:PANEL,
    font:{color:TX}, margin:{l:60,r:20,t:34,b:44}}, extra||{});
}
if(!DATA.epoch.length){
  document.getElementById("erp").innerHTML =
    "<p style='color:#9a9ab0'>No event-flagged epochs in this recording.</p>";
} else {
  Plotly.newPlot("erp", [{
    z: DATA.resid, x: DATA.time_ms,
    y: DATA.epoch.map((e,i)=>i), type:"heatmap",
    colorscale:"RdBu", reversescale:true, zmid:0,
    colorbar:{title:{text:"residual (µV)",font:{color:TX}},
              tickfont:{color:MU}},
    hovertemplate:"event row %{y}<br>%{x} ms<br>%{z:.3f} µV<extra></extra>"
  }], baseLayout({title:{text:"Event residuals (click a row)",font:{color:TX}},
    xaxis:{title:"time from stim (ms)",color:MU,gridcolor:"#3a3a52"},
    yaxis:{title:"event (time order)",color:MU,gridcolor:"#3a3a52"}}),
    {responsive:true, displmodeBar:false});
  const erp = document.getElementById("erp");
  erp.on("plotly_click", ev => { if(ev.points&&ev.points.length)
      show(ev.points[0].pointIndex[0]); });
  show(0);
}
function show(i){
  const st = DATA.stim_time[i]==null ? "n/a" : DATA.stim_time[i]+" s";
  document.getElementById("meta").textContent =
    "Event row "+i+"  ·  epoch #"+DATA.epoch[i]+"  ·  stim @ "+st+
    "  ·  residual energy "+DATA.energy[i];
  Plotly.react("detail", [
    {x:DATA.time_ms, y:DATA.raw[i], name:"raw", mode:"lines",
     line:{color:RAW,width:1}},
    {x:DATA.time_ms, y:DATA.resid[i], name:"residual (event)", mode:"lines",
     line:{color:EVT,width:1.6}}
  ], baseLayout({title:{text:"Event row "+i+" — individual trace",
      font:{color:TX}},
    xaxis:{title:"time from stim (ms)",color:MU,gridcolor:"#3a3a52"},
    yaxis:{title:"LFP (µV)",color:MU,gridcolor:"#3a3a52"},
    legend:{font:{color:TX}}}), {responsive:true, displayModeBar:true});
}
</script></body></html>
"""
