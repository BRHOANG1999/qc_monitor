"""Amplifier-gain lookup from the KMrecorder ``File_Records`` sheet.

The recording-amplifier gain (e.g. 300x, 150x) is what converts a raw
recorded deflection into an absolute voltage, and it is NOT in any .mat
file -- it lives in the lab's ``File_Records`` tab of the KMrecorder Data
Log spreadsheet (the same sheet the Sessions / Notes / Reviewer
Assignments tabs already read). Columns:

  * ``EEG File``            -- recording base name = .mat basename w/o ext
  * ``Animal ID & Channel`` -- pipe-separated channel labels, in order
  * ``Amplifier Gain``      -- pipe-separated gains, positionally aligned
  * ``Date``                -- recording date (for latest-wins fallback)

We build three lookups so a recording's per-channel gain resolves even
when the exact chunk isn't logged:

  1. exact per-file (EEG File == chunk basename),
  2. per-session (EEG File minus the trailing timestamp == session_name),
  3. latest-known gain for that channel / (animal, electrode).

Loaded through the shared TTL-cached Sheets client so a dashboard tick
doesn't pay an API call per refresh.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.utils.sheets import (
    _load_sheet_via_api, _resolve_sa_path, _find_column, _normalize_text,
)
from src.utils.animal import split_animal_electrode

logger = logging.getLogger("qc_monitor.utils.amplifier_records")

# Trailing recorder timestamp: "..__2026_07_02__15_47_09" (2-3 leading _).
_TS_RX = re.compile(r"_{2,3}\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$")


# --------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------- #

def _norm_chan(name) -> str:
    """Channel key: lowercase, alphanumerics only (so 'BCH110 SLM',
    'BCH110SLM', 'bch110_slm' all collapse to the same key)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _base_no_ext(name: str) -> str:
    """Strip directory + a trailing .mat from a recording name."""
    s = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    if s.lower().endswith(".mat"):
        s = s[:-4]
    return s


def _strip_ts(base: str) -> str:
    """Drop the trailing recorder timestamp, yielding the session name."""
    return _TS_RX.sub("", str(base))


def _to_float(x) -> float | None:
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


# --------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------- #

@dataclass
class AmplifierGains:
    """Resolved gain lookups. All channel keys are ``_norm_chan``-ed."""
    by_file: dict = field(default_factory=dict)          # base -> {ch: gain}
    by_session: dict = field(default_factory=dict)       # sess -> {ch: gain}
    latest_by_channel: dict = field(default_factory=dict)   # ch -> (date, gain)
    latest_by_ae: dict = field(default_factory=dict)     # (animal,elec)->(date,gain)

    def resolve(self, session_name: str, file_basename: str,
                channel_name: str, animal_id: str = "",
                electrode: str = "") -> float | None:
        """Best gain for one (recording, channel): exact file -> session
        -> latest-by-channel -> latest-by-(animal,electrode) -> None."""
        ch = _norm_chan(channel_name)
        if not ch:
            return None
        fb = _norm_chan_free(_base_no_ext(file_basename))
        if fb in self.by_file and ch in self.by_file[fb]:
            return self.by_file[fb][ch]
        sk = _norm_chan_free(_strip_ts(session_name))
        if sk in self.by_session and ch in self.by_session[sk]:
            return self.by_session[sk][ch]
        if ch in self.latest_by_channel:
            return self.latest_by_channel[ch][1]
        ae = (str(animal_id).lower(), str(electrode).lower())
        if ae in self.latest_by_ae:
            return self.latest_by_ae[ae][1]
        return None


def _norm_chan_free(name) -> str:
    """File/session key: lowercase, keep alphanumerics only. Same idea as
    channel keys but applied to the whole recording base name."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# --------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------- #

def _cfg(config: dict) -> dict:
    return (config or {}).get("amplifier_records", {}) or {}


def load_amplifier_gains(config: dict, ttl_sec: float | None = None
                         ) -> AmplifierGains | None:
    """Fetch + parse the File_Records tab into an ``AmplifierGains``.

    Returns an empty ``AmplifierGains`` when the feature is disabled or the
    sheet is unreachable/empty, and ``None`` only on a hard config error so
    callers can distinguish "no gains" from "not configured".
    """
    cfg = _cfg(config)
    if not cfg or not cfg.get("sheet_id"):
        return AmplifierGains()
    sheet_id = cfg.get("sheet_id")
    tab_name = cfg.get("tab_name") or "File_Records"
    sa_raw = cfg.get("service_account_file") or (
        (config.get("data_log_xref", {}) or {}).get("service_account_file", ""))
    sa_path = _resolve_sa_path(config or {}, sa_raw)
    if not sa_path:
        return AmplifierGains()
    if ttl_sec is None:
        ttl_sec = float(cfg.get("refresh_minutes", 60)) * 60.0
    df = _load_sheet_via_api(sheet_id, tab_name, sa_path, ttl_sec)
    if df is None or df.empty:
        return AmplifierGains()
    return _parse_frame(df)


def _parse_frame(df) -> AmplifierGains:
    """Turn the File_Records DataFrame into an AmplifierGains."""
    file_col = _find_column(df, ["EEG File", "EEG_File", "Filename",
                                  "File", "Recording File"])
    chan_col = _find_column(df, ["Animal ID & Channel", "Animal ID and Channel",
                                  "Animal_ID_Channel", "Channels", "Channel"])
    gain_col = _find_column(df, ["Amplifier Gain", "Amplifier_Gain", "Gain"])
    date_col = _find_column(df, ["Date", "Recording Date"])
    out = AmplifierGains()
    if chan_col is None or gain_col is None:
        logger.warning("File_Records missing Channel/Gain columns "
                        "(have: %s)", list(df.columns))
        return out
    max_iter = len(df) + 1
    for i, (_idx, row) in enumerate(df.iterrows()):
        assert i < max_iter, "File_Records scan runaway"
        chans = _split_pipes(row.get(chan_col))
        gains = _split_pipes(row.get(gain_col))
        if not chans or not gains:
            continue
        ch_gain = _zip_channel_gains(chans, gains)
        if not ch_gain:
            continue
        date_str = _normalize_text(row.get(date_col)) if date_col else ""
        base = _base_no_ext(_normalize_text(row.get(file_col))) if file_col else ""
        _index_row(out, base, ch_gain, date_str, chans, gains)
    logger.info("File_Records: %d files, %d channels indexed",
                len(out.by_file), len(out.latest_by_channel))
    return out


def _split_pipes(value) -> list[str]:
    text = _normalize_text(value)
    if not text:
        return []
    return [p.strip() for p in text.split("|")]


def _zip_channel_gains(chans: list[str], gains: list[str]) -> dict:
    """Align pipe lists positionally -> {norm_channel: gain_float}."""
    n = min(len(chans), len(gains))
    out = {}
    for i in range(n):
        g = _to_float(gains[i])
        ck = _norm_chan(chans[i])
        if g is not None and ck:
            out[ck] = g
    return out


def _index_row(out: AmplifierGains, base: str, ch_gain: dict,
               date_str: str, chans: list[str], gains: list[str]) -> None:
    """Populate all four lookups from one File_Records row."""
    if base:
        out.by_file[_norm_chan_free(base)] = ch_gain
        out.by_session.setdefault(_norm_chan_free(_strip_ts(base)), {}).update(
            ch_gain)
    for i in range(min(len(chans), len(gains))):
        g = _to_float(gains[i])
        if g is None:
            continue
        ck = _norm_chan(chans[i])
        if ck:
            _keep_latest(out.latest_by_channel, ck, date_str, g)
        animal, elec = split_animal_electrode(chans[i].strip())
        if animal:
            _keep_latest(out.latest_by_ae,
                         (animal.lower(), (elec or "").lower()), date_str, g)


def _keep_latest(m: dict, key, date_str: str, gain: float) -> None:
    """Latest-wins by date string (ISO dates sort lexically). Blank dates
    lose to any dated entry but still seed an empty slot."""
    prev = m.get(key)
    if prev is None or date_str >= prev[0]:
        m[key] = (date_str, gain)


def resolve_gain(gains: AmplifierGains | None, session_name: str,
                 file_basename: str, channel_name: str,
                 animal_id: str = "", electrode: str = "") -> float | None:
    """Module-level convenience delegating to ``AmplifierGains.resolve``."""
    if gains is None:
        return None
    return gains.resolve(session_name, file_basename, channel_name,
                         animal_id, electrode)
