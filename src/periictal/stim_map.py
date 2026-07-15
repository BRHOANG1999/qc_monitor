"""Map an (animal, session) to the stim-hardware fingerprint it received.

The lab's protocol name lives only in the session FOLDER token (e.g.
``chronicStim-5nC-2nC``); two differently-named protocols can be physically
identical, so grouping recordings by that string is wrong. The ground truth is
the sibling ``*_STIM_REPORT.txt`` (parsed by ``src.utils.stim_parser``), which
records per-output-channel charge / pulse-width / neg-ratio / frequency / gain.

The animal->output-channel link is POSITIONAL and was verified empirically
against the live share: the k-th ``stimCopy`` in the session's channel list
records STIM_REPORT **Channel k+1**, and the animal channel IMMEDIATELY
following that stimCopy is the one it stimulates. An animal not preceded by a
stimCopy is record-only (valid for the passive embedding, excluded from the
stim-evoked one). When the report is missing or the counts disagree we return
``unknown`` and surface it -- we never guess a charge.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

from src.utils.animal import (is_animal_channel, split_animal_electrode,
                              stim_copy_indices)
from src.utils.stim_parser import StimReport, parse_stim_report_file

_MAX_CH = 4096          # NASA Rule 2: explicit bound on channel-list scans.


@dataclass(frozen=True)
class StimFingerprint:
    """The stim an animal received in one session. ``status`` is
    ``mapped`` (params resolved), ``record_only`` (animal not stimulated), or
    ``unknown`` (no report / count mismatch -- surfaced, never guessed)."""
    status: str
    report_channel: int | None = None      # 1-based STIM_REPORT channel number
    charge_nC: float | None = None
    pulse_width_us: float | None = None
    neg_pulse_ratio: float | None = None
    frequency_hz: float | None = None
    gain: float | None = None
    active: bool | None = None

    def key(self) -> str:
        """A stable grouping key: identical hardware settings -> identical key,
        regardless of the protocol folder name."""
        if self.status != "mapped":
            return self.status
        tail = "" if self.active else " (inactive)"
        return (f"{_g(self.charge_nC)}nC/{_g(self.pulse_width_us)}us/"
                f"x{_g(self.neg_pulse_ratio)}/{_g(self.frequency_hz)}Hz/"
                f"g{_g(self.gain)}{tail}")


def _g(v) -> str:
    return "?" if v is None else f"{float(v):g}"


def find_stim_report(session_dir: str) -> str | None:
    """Path to a ``*_STIM_REPORT.txt`` in *session_dir* (any one -- the stim
    settings are constant within a protocol session), or None when the folder
    is unreachable / has no report."""
    if not session_dir:
        return None
    norm = session_dir.replace("\\", os.sep).replace("/", os.sep)
    try:
        if not os.path.isdir(norm):
            return None
        hits = sorted(glob.glob(os.path.join(norm, "*_STIM_REPORT.txt")))
    except OSError:
        return None
    return hits[0] if hits else None


def _stim_ordinal(channel_names: list, animal_channel_index: int) -> int | None:
    """0-based ordinal k of the stimCopy that immediately precedes
    *animal_channel_index* among all stimCopies -- i.e. this animal's stim is
    STIM_REPORT Channel k+1. None when the animal isn't preceded by a stimCopy
    (record-only)."""
    assert animal_channel_index >= 0, "channel index must be >= 0"
    sc = sorted(stim_copy_indices(channel_names))
    prev = animal_channel_index - 1
    if prev in sc:
        return sc.index(prev)
    return None


def _channel_names(store, session_dir: str) -> list:
    cfg = store.get_session_config(session_dir) or {}
    raw = cfg.get("channel_names")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    return list(raw or [])


def _stimulated_electrode_index(names: list, animal: str) -> int | None:
    """Index of *animal*'s electrode that is preceded by a stimCopy (the
    stimulated site), or None if the animal has no stimulated electrode."""
    sc = set(stim_copy_indices(names))
    for i, n in enumerate(names):
        assert i < _MAX_CH, "channel-name scan runaway"
        if not (isinstance(n, str) and is_animal_channel(n)):
            continue
        if split_animal_electrode(n)[0] == animal and (i - 1) in sc:
            return i
    return None


def resolve_fingerprint(store, session_dir: str, animal: str) -> StimFingerprint:
    """The stim *animal* received in *session_dir*. See module docstring for the
    positional rule and the fail-loud ``unknown`` policy."""
    assert animal, "animal required"
    names = _channel_names(store, session_dir)
    if not names:
        return StimFingerprint("unknown")
    j = _stimulated_electrode_index(names, animal)
    if j is None:
        return StimFingerprint("record_only")
    k = _stim_ordinal(names, j)
    if k is None:
        return StimFingerprint("record_only")
    report_path = find_stim_report(session_dir)
    if not report_path:
        return StimFingerprint("unknown", report_channel=k + 1)
    try:
        report: StimReport = parse_stim_report_file(report_path)
    except (OSError, ValueError):
        return StimFingerprint("unknown", report_channel=k + 1)
    if k >= len(report.channels):
        return StimFingerprint("unknown", report_channel=k + 1)
    ch = report.channels[k]
    return StimFingerprint(
        "mapped", report_channel=k + 1, charge_nC=ch.charge_nC,
        pulse_width_us=ch.pulse_width_us, neg_pulse_ratio=ch.neg_pulse_ratio,
        frequency_hz=ch.frequency_hz, gain=ch.gain, active=ch.active)


def session_dir_by_token(store, animal: str) -> dict[str, str]:
    """Map each protocol-folder token (``chronicStim-5nC-2nC``) to a representative
    session_dir for *animal*, so an evoked row's ``session`` field can be resolved
    to a session (and thence a fingerprint). Newest session per token wins."""
    assert animal, "animal required"
    out: dict[str, str] = {}
    for i, cfg in enumerate(store.get_all_session_configs()):
        assert i < 1_000_000, "session-config scan runaway"
        names = cfg.get("channel_names") or ""
        if animal not in (names if isinstance(names, str) else json.dumps(names)):
            continue
        sd = cfg.get("session_dir") or ""
        token = os.path.basename(sd).split("__", 1)[0]
        if token and token not in out:
            out[token] = sd
    return out
