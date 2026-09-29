"""Auto-generate report.md for the evoked-shape analysis: the discrete-vs-continuum
verdict with its evidence, the k range and why, the figures, and every assumption that
had to be made. Plain markdown, no external deps."""

from __future__ import annotations

import os
from datetime import datetime, timezone


def _fmt_coverage(cov: dict) -> str:
    if not cov:
        return "  (no session tokens resolved)"
    order = sorted(cov.items())
    return "\n".join(f"  - `{tok}`: {status}" for tok, status in order)


def build_report(*, animal: str, channel: str, ds: dict, pre: dict, struct: dict,
                 sel: dict, figures: dict, n_boot: int, config_notes: dict) -> str:
    """Assemble the report markdown string."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_lab = int(config_notes.get("n_preictal", 0))
    lines = [
        f"# Evoked response shape analysis: {animal}",
        "",
        f"Generated {now} UTC. Channel `{channel}`. "
        f"Trials gathered {ds.get('n_gathered', 0)}, "
        f"kept after QC {pre.get('n_kept', 0)} "
        f"(dropped {pre.get('n_dropped', 0)} with no evoked response). "
        f"fs = {ds.get('fs', float('nan')):.0f} Hz.",
        "",
        "## Stage 2: discrete or continuous?",
        "",
        "```",
        struct.get("text", ""),
        "```",
        "",
        f"![structure]({os.path.basename(figures.get('structure', ''))})",
        "",
        "## Stage 3: shape templates",
        "",
        f"![templates]({os.path.basename(figures.get('templates', ''))})",
        "",
        "## Stage 4: how many shapes (k)?",
        "",
        sel.get("text", ""),
        "",
        f"![select_k]({os.path.basename(figures.get('select_k', ''))})",
        "",
        "## Seizure / stim coverage",
        "",
        f"Trials falling inside a pre-ictal window (cluster-leader seizures, "
        f"ISI-ceilinged): {n_lab}.",
        "",
        "Per session-token stim fingerprint status:",
        _fmt_coverage(ds.get("stim_coverage", {})),
        "",
        "## Assumptions and caveats (flagged)",
        "",
        f"- Analysis window {config_notes.get('window_ms')} ms post-stim; the stim "
        f"artifact is excluded by cropping (1 ms guard), not blanked.",
        f"- Trials amplitude-normalized ('{config_notes.get('norm')}'); the norm "
        f"divisor is kept as the per-trial gain (there is no per-trial gain in the "
        f".mat). Shape structure is therefore invariant to amplitude scaling.",
        "- `session` is the recording DAY, so held-out and split-half validation are "
        "by day. `stim_status` is resolved at the animal level, not per trial "
        "(the subsampled trace reader does not carry each trial's session_dir).",
        f"- Bootstrap stability used {n_boot} resamples with a dependence-aware "
        "moving-block length and a PC-shuffled null; the reported stability is "
        "observed minus null.",
        "- Risk analysis (shape vs time-to-seizure) and the conv autoencoder are "
        "deferred: for this cohort too few seizures fall in stim-covered sessions for "
        "a defensible risk result yet.",
        "",
    ]
    return "\n".join(lines)


def write_report(path: str, text: str) -> str:
    """Write the report atomically. Returns the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return path
