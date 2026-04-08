"""Parse KMrecorder STIM_REPORT.txt files."""

import re
from dataclasses import dataclass


@dataclass
class StimChannel:
    channel_num: int
    active: bool
    charge_nC: float
    pulse_width_us: float
    neg_pulse_ratio: float
    frequency_hz: float
    gain: float
    total_pulses: int
    total_charge_nC: float


@dataclass
class StimReport:
    recording_file: str
    report_generated: str
    stim_start: str
    stim_stop: str
    duration_sec: float
    daq_device: str
    output_channels: list[int]
    sampling_rate: int
    num_channels: int
    channels: list[StimChannel]
    events: list[tuple[str, str, str]]  # (timestamp, event_type, channel)


def parse_stim_report(text: str) -> StimReport:
    """Parse a STIM_REPORT.txt file content into a StimReport."""

    recording_file = _extract(r"Recording File:\s*(.+)", text, "")
    report_generated = _extract(r"Report Generated:\s*(.+)", text, "")
    stim_start = _extract(r"Stimulation Start:\s*(.+)", text, "")
    stim_stop = _extract(r"Stimulation Stop:\s*(.+)", text, "")

    duration_match = re.search(r"Duration:\s*([\d.]+)\s*seconds", text)
    duration_sec = float(duration_match.group(1)) if duration_match else 0.0

    daq_device = _extract(r"DAQ Device:\s*(\S+)", text, "")

    oc_match = re.search(r"Output Channels:\s*(.+)", text)
    output_channels = []
    if oc_match:
        output_channels = [int(x) for x in re.findall(r"\d+", oc_match.group(1))]

    sr_match = re.search(r"Sampling Rate:\s*(\d+)", text)
    sampling_rate = int(sr_match.group(1)) if sr_match else 0

    nc_match = re.search(r"Number of Channels:\s*(\d+)", text)
    num_channels = int(nc_match.group(1)) if nc_match else 0

    # Parse channel blocks
    channels = []
    channel_blocks = re.finditer(
        r"Channel\s+(\d+):\s*(ACTIVE|INACTIVE)(.*?)(?=Channel\s+\d+:|---\s*CONVERSION|---\s*SAFETY|---\s*EVENT LOG|\Z)",
        text, re.DOTALL
    )
    for m in channel_blocks:
        ch_num = int(m.group(1))
        active = m.group(2) == "ACTIVE"
        block = m.group(3)

        charge = _extract_float(r"Charge per phase:\s*([\d.]+)", block)
        pw = _extract_float(r"Pulse width:\s*([\d.]+)", block)
        npr = _extract_float(r"Neg pulse ratio:\s*([\d.]+)", block)
        freq = _extract_float(r"Frequency:\s*([\d.]+)", block)
        gain = _extract_float(r"Gain:\s*([\d.]+)", block)
        pulses = int(_extract_float(r"Total pulses:\s*(\d+)", block))
        charge_total = _extract_float(r"Total charge:\s*([\d.]+)", block)

        channels.append(StimChannel(
            channel_num=ch_num, active=active, charge_nC=charge,
            pulse_width_us=pw, neg_pulse_ratio=npr, frequency_hz=freq,
            gain=gain, total_pulses=pulses, total_charge_nC=charge_total,
        ))

    # Parse event log
    events = []
    event_section = re.search(r"---\s*EVENT LOG\s*---(.+?)(?:={5,}|\Z)", text, re.DOTALL)
    if event_section:
        for em in re.finditer(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\|\s*(\S+)\s*\|\s*(.+)",
            event_section.group(1)
        ):
            events.append((em.group(1).strip(), em.group(2).strip(), em.group(3).strip()))

    return StimReport(
        recording_file=recording_file.strip(),
        report_generated=report_generated.strip(),
        stim_start=stim_start.strip(),
        stim_stop=stim_stop.strip(),
        duration_sec=duration_sec,
        daq_device=daq_device.strip(),
        output_channels=output_channels,
        sampling_rate=sampling_rate,
        num_channels=num_channels,
        channels=channels,
        events=events,
    )


def parse_stim_report_file(path: str) -> StimReport:
    """Parse a STIM_REPORT.txt file from path."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_stim_report(f.read())


def _extract(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else default


def _extract_float(pattern: str, text: str, default: float = 0.0) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default
