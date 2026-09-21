"""Per-animal peri-ictal stim-artifact email, sent when enough of an animal's
seizures move into the needs-scoring queue (the count-based scheduler trigger in
:mod:`src.notifications.scheduler`).

Wraps :func:`src.periictal.peri_stim_artifact.run` (render per-seizure + pooled
overlay PNGs) and emails the headline figures inline, matching the ad-hoc send
this replaces. Heavy (reads raw evoked ``.mat`` epochs), so the scheduler calls
it on a background thread.
"""

from __future__ import annotations

import logging
import os
import tempfile

logger = logging.getLogger("qc_monitor.notifications.peri_stim_needs_scoring")

_MAX_INLINE = 24                       # cap inlined figures so the email stays sane


def _headline_pngs(res: dict) -> list:
    """The figures to inline, in reading order: each channel's pooled +
    pooled-overlay, then every per-seizure by-window overlay panel. Only paths
    that exist on disk, de-duplicated, capped at ``_MAX_INLINE``."""
    ordered = []
    for _ch, d in (res.get("channels") or {}).items():
        ordered += [d.get("pooled"), d.get("pooled_overlay")]
    ordered += [p for p in (res.get("panels") or [])
                if str(p).endswith("_overlay.png")]
    seen, out = set(), []
    for p in ordered:
        if p and p not in seen and os.path.exists(p):
            seen.add(p)
            out.append(p)
    return out[:_MAX_INLINE]


def _build_html(animal: str, res: dict, imgs: list, reason: str) -> str:
    """Inline-figure HTML body (cid:<basename> per image)."""
    def block(p):
        cid = os.path.basename(p)
        return (f'<p style="margin:16px 0 4px"><b>{cid}</b></p>'
                f'<img src="cid:{cid}" style="max-width:820px;width:100%;'
                f'border:1px solid #ddd">')
    n_sz = sum(d.get("n_seizures", 0) for d in (res.get("channels") or {}).values())
    figs = "".join(block(p) for p in imgs)
    return (
        '<div style="font-family:Segoe UI,Arial,sans-serif;color:#222;'
        'max-width:860px">'
        f'<h2>Peri-ictal stim-artifact — {animal}</h2>'
        f'<p>{reason} Is a seizure destabilising the stim artifact? For each '
        'scored seizure, the LFP stim artifact (−1..2&nbsp;ms) is captured in '
        'three 30-min windows — <b>pre</b> (−30..0), <b>at</b> (0..+30), '
        '<b>post</b> (+30..+60&nbsp;min) — truncated so a window never overlaps a '
        'neighbouring seizure. Overlays show every constituent stimulus artifact '
        '(nothing downsampled) under each window mean.</p>'
        f'<p>{n_sz} seizure-panels across '
        f'{len(res.get("channels") or {})} channel(s).</p>'
        f'{figs}'
        '</div>')


def send_peri_stim_artifact(animal: str, config: dict, store, emailer, *,
                            reason: str = "", out_dir: str | None = None) -> dict:
    """Render *animal*'s peri-ictal stim-artifact figures and email the headline
    set. Returns ``{sent, animal, n_channels, n_inlined, reason}``. Never raises
    for the 'nothing to show' case (no scored seizures / no stim coverage) — it
    returns ``sent=False`` with a reason so the caller can still advance its
    high-water mark."""
    assert animal, "animal required"
    assert emailer is not None and store is not None, "store + emailer required"
    from src.periictal import peri_stim_artifact as P
    cfg = ((config.get("notifications", {}) or {}).get("peri_stim", {}) or {})
    recipients = (cfg.get("recipients")
                  or (config.get("alerting", {}) or {}).get("smtp", {})
                  .get("recipients", []))
    tmp = out_dir or os.path.join(tempfile.gettempdir(),
                                  f"peri_stim_{_safe(animal)}")
    res = P.run(store, animal, out_dir=tmp, channel=cfg.get("channel"))
    imgs = _headline_pngs(res)
    if not imgs:
        logger.info("peri-stim email skipped for %s: %s", animal,
                    res.get("reason") or "no figures")
        return {"sent": False, "animal": animal, "n_channels": 0,
                "n_inlined": 0, "reason": res.get("reason") or "no figures"}
    subject = f"Peri-ictal stim-artifact — {animal}"
    html = _build_html(animal, res, imgs, reason)
    ok = emailer.send(subject, f"Peri-ictal stim-artifact figures for {animal}.",
                      severity="info", recipients=recipients or None,
                      body_html=html, subject_prefix=False, attachments=imgs)
    logger.info("peri-stim email for %s: sent=%s (%d figures -> %s)",
                animal, ok, len(imgs), recipients)
    return {"sent": bool(ok), "animal": animal,
            "n_channels": len(res.get("channels") or {}),
            "n_inlined": len(imgs), "reason": reason}


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))
