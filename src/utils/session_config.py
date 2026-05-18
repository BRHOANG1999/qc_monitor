"""Auto-discovery of session configuration from recorder_settings.mat and session .mat files."""

import json
import logging
import os
from dataclasses import dataclass, field, asdict

import numpy as np

logger = logging.getLogger("qc_monitor.session_config")


@dataclass
class SessionConfig:
    # Channel mapping
    channel_names: list[str] = field(default_factory=list)
    eeg_channels: list[int] = field(default_factory=list)
    stim_copy_channels: list[int] = field(default_factory=list)
    reference_channels: list[int] = field(default_factory=list)

    # Stimulus parameters
    stim_frequency_hz: float = 0.5
    stim_charge_nC: float = 14.0
    stim_pulse_width_us: float = 150.0
    stim_neg_ratio: float = 3.0
    stim_active_output_channels: list[int] = field(default_factory=list)

    # Recording params
    fs: int = 20000
    num_channels: int = 4
    framesize_ms: int = 1000

    # Analysis thresholds (from saved spike settings)
    spike_threshold: float = 10.0
    spike_min_width: int = 5
    spike_max_width: int = 50
    power_threshold: float = 20.0
    coastline_threshold: float = 800.0

    # Source
    source_file: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def primary_eeg_channel(self) -> int:
        """Return the first EEG channel index (0-based). Falls back to 0."""
        return self.eeg_channels[0] if self.eeg_channels else 0

    def primary_stim_channel(self) -> int:
        """Return the first stim copy channel index (0-based). Falls back to 1."""
        return self.stim_copy_channels[0] if self.stim_copy_channels else min(1, self.num_channels - 1)


def discover_session_config(mat_path: str) -> SessionConfig:
    """Auto-discover session configuration from a .mat file.

    Works with both recorder_settings.mat and individual session .mat files.
    Both contain fnstr, stim_params, spike settings, etc.
    """
    import scipy.io

    config = SessionConfig(source_file=mat_path)

    try:
        data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        logger.warning("Failed to load %s for auto-discovery: %s", mat_path, e)
        return config

    # --- Channel names from fnstr ---
    if "fnstr" in data:
        fnstr = data["fnstr"]
        if hasattr(fnstr, "__len__") and not isinstance(fnstr, str):
            names = []
            for item in fnstr:
                if isinstance(item, np.ndarray):
                    names.append(str(item.flat[0]) if item.size > 0 else "")
                else:
                    names.append(str(item))
            config.channel_names = names
        elif isinstance(fnstr, str):
            config.channel_names = [fnstr]

    # --- Number of channels ---
    if "number_of_channels" in data:
        config.num_channels = int(data["number_of_channels"])

    # Trim channel names to active count
    if config.channel_names and config.num_channels:
        config.channel_names = config.channel_names[:config.num_channels]

    # --- Auto-detect channel roles from names ---
    # Rule: stimCopy channels are stimulus artifact recordings.
    # ALL other channels are LFP (including saline, BCH*, generic names).
    for i, name in enumerate(config.channel_names):
        name_lower = name.lower().strip()
        if "stim" in name_lower:
            config.stim_copy_channels.append(i)
        else:
            config.eeg_channels.append(i)  # all non-stim = LFP

    # --- Sampling rate ---
    if "fs" in data:
        config.fs = int(data["fs"])

    # --- Frame size ---
    if "framesize" in data:
        config.framesize_ms = int(data["framesize"])

    # --- Stimulus parameters from stim_params ---
    if "stim_params" in data:
        sp = data["stim_params"]
        try:
            if hasattr(sp, "channels"):
                channels = sp.channels
                if not hasattr(channels, "__len__"):
                    channels = [channels]
                for i, ch in enumerate(channels):
                    is_active = getattr(ch, "isActive", 0)
                    if is_active:
                        config.stim_active_output_channels.append(i)
                        config.stim_frequency_hz = float(getattr(ch, "frequency", 0.5))
                        config.stim_charge_nC = float(getattr(ch, "chargePerPhase", 14.0))
                        config.stim_pulse_width_us = float(getattr(ch, "pulseWidth", 150.0))
                        config.stim_neg_ratio = float(getattr(ch, "negPulseWidthRatio", 3.0))
        except Exception as e:
            logger.debug("Could not parse stim_params.channels: %s", e)

    # --- Spike detection thresholds ---
    if "spike1" in data:
        s1 = data["spike1"]
        try:
            if hasattr(s1, "thr"):
                thr = s1.thr
                if hasattr(thr, "__len__"):
                    config.spike_threshold = float(thr[0])
                else:
                    config.spike_threshold = float(thr)
            if hasattr(s1, "minwidth"):
                mw = s1.minwidth
                config.spike_min_width = int(mw[0] if hasattr(mw, "__len__") else mw)
            if hasattr(s1, "maxwidth"):
                mw = s1.maxwidth
                config.spike_max_width = int(mw[0] if hasattr(mw, "__len__") else mw)
        except Exception:
            pass

    # --- Power threshold ---
    if "power" in data:
        pwr = data["power"]
        try:
            if hasattr(pwr, "pwrthr"):
                pt = pwr.pwrthr
                config.power_threshold = float(pt[0] if hasattr(pt, "__len__") else pt)
        except Exception:
            pass

    # --- Coastline threshold ---
    if "coast" in data:
        coast = data["coast"]
        try:
            if hasattr(coast, "thr"):
                ct = coast.thr
                config.coastline_threshold = float(ct[0] if hasattr(ct, "__len__") else ct)
        except Exception:
            pass

    logger.info("Auto-discovered config from %s:", os.path.basename(mat_path))
    logger.info("  Channels: %s", config.channel_names)
    logger.info("  EEG: %s, StimCopy: %s, Reference: %s",
                [config.channel_names[i] for i in config.eeg_channels] if config.eeg_channels else "none",
                [config.channel_names[i] for i in config.stim_copy_channels] if config.stim_copy_channels else "none",
                [config.channel_names[i] for i in config.reference_channels] if config.reference_channels else "none")
    logger.info("  Stim: %.1f Hz, %.1f nC, %.0f µs", config.stim_frequency_hz, config.stim_charge_nC, config.stim_pulse_width_us)
    logger.info("  Fs: %d Hz, Channels: %d", config.fs, config.num_channels)

    return config


def discover_from_session_dir(session_dir: str) -> SessionConfig:
    """Try to discover config from a session directory.

    Prefers the first actual data .mat in the session dir, because its fnstr
    reflects what was *recorded* in this session. A shared recorder_settings.mat
    upstream may carry a stale channel layout from a previous rig configuration
    and would mislabel channels (e.g. naming a stimCopy electrode as an LFP).
    Falls back to recorder_settings.mat only when no data file is readable.
    """
    # Prefer the first data .mat in the session dir — it has fnstr, stim_params,
    # spike1, power, coast, plus the channel layout actually used for this recording.
    try:
        for entry in sorted(os.scandir(session_dir), key=lambda e: e.name):
            if entry.name.endswith(".mat") and not entry.name.startswith("metadata"):
                cfg = discover_session_config(entry.path)
                if cfg.channel_names:
                    return cfg
                break  # data file existed but had no fnstr; try settings fallback
    except OSError:
        pass

    # Fall back to recorder_settings.mat in parent / grandparent
    parent = os.path.dirname(session_dir)
    for candidate in (
        os.path.join(parent, "recorder_settings.mat"),
        os.path.join(os.path.dirname(parent), "recorder_settings.mat"),
    ):
        if os.path.isfile(candidate):
            return discover_session_config(candidate)

    logger.warning("No settings found for session %s, using defaults", session_dir)
    return SessionConfig()
