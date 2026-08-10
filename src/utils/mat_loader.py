"""Load KMrecorder .mat files (sbuf/fs/trdata format) into Python."""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChunkData:
    signal: np.ndarray          # [samples x channels], float64
    fs: float                   # sampling rate Hz
    num_channels: int
    num_samples: int
    duration_sec: float
    source_path: str
    timestamps: np.ndarray | None = None
    trdata: dict = field(default_factory=dict)
    channel_names: list = field(default_factory=list)


def load_mat(path: str) -> ChunkData:
    """Load a KMrecorder .mat file. Handles both v5 and v7.3 formats."""
    from src.utils.mirror import local_first
    # Read from the fast local rolling mirror when this file is mirrored; falls
    # back to the SMB share for anything outside the window. No-op when the
    # mirror is disabled. Works in spawned child readers (env-driven).
    path = local_first(str(path))

    try:
        import scipy.io
        data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        # v7.3 HDF5 format
        import h5py
        data = _load_hdf5(path)

    # KMrecorder format: sbuf field
    if "sbuf" in data:
        sbuf = np.asarray(data["sbuf"], dtype=np.float64)
        fs = float(data.get("fs", 20000))
        trdata_raw = data.get("trdata", None)
        trdata = _parse_trdata(trdata_raw) if trdata_raw is not None else {}
    # STiM-NET chunk format: data + chunkInfo
    elif "data" in data:
        sbuf = np.asarray(data["data"], dtype=np.float64)
        chunk_info = data.get("chunkInfo", None)
        # v7 (scipy) gives a struct object (.samplingRate); v7.3 (_load_hdf5)
        # gives a dict. getattr on a dict silently missed samplingRate and
        # defaulted fs to 20000 -- corrupting duration + every time->sample
        # conversion for HDF5 chunks. Handle both shapes.
        if isinstance(chunk_info, dict):
            fs = float(chunk_info.get("samplingRate", 20000))
        elif chunk_info is not None:
            fs = float(getattr(chunk_info, "samplingRate", 20000))
        else:
            fs = 20000.0
        trdata = {}
    else:
        raise ValueError(f"Unknown .mat format. Keys: {list(data.keys())}")

    if sbuf.ndim == 1:
        sbuf = sbuf.reshape(-1, 1)

    # Orient to [samples x channels]. MATLAB v7.3 (HDF5) stores arrays
    # transposed vs MATLAB, so _load_hdf5 returns sbuf as [channels x
    # samples]; v7 (scipy) returns [samples x channels]. Samples always far
    # outnumber channels (>=1 s @ kHz vs <=64 ch), so the longer axis is
    # samples -- transpose when it isn't axis 0. Without this a v7.3 file
    # reads as e.g. 5 "samples" x 72M "channels" (a 250 us degenerate trace).
    if sbuf.ndim == 2 and sbuf.shape[0] < sbuf.shape[1]:
        sbuf = sbuf.T

    num_samples, num_channels = sbuf.shape
    duration_sec = num_samples / fs

    channel_names = _parse_fnstr(data.get("fnstr"), num_channels)

    return ChunkData(
        signal=sbuf,
        fs=fs,
        num_channels=num_channels,
        num_samples=num_samples,
        duration_sec=duration_sec,
        source_path=path,
        trdata=trdata,
        channel_names=channel_names,
    )


def _orient_dims(shape) -> tuple[int, int]:
    """(num_samples, num_channels) from a raw array shape, mirroring load_mat's
    orientation rule: samples always far outnumber channels, so the LONGER axis
    is samples. 1-D -> single channel."""
    if len(shape) == 1:
        return int(shape[0]), 1
    a, b = int(shape[0]), int(shape[1])
    return (a, b) if a >= b else (b, a)


def read_mat_header(path: str) -> tuple[int, int, float]:
    """Cheaply read ``(num_samples, num_channels, fs)`` from a KMrecorder /
    STiM-NET .mat WITHOUT loading the full signal array -- used by the duration
    backfill so it doesn't pull megabytes per file across the network. Mirrors
    load_mat's format detection + [samples x channels] orientation. Raises on
    unknown format or unreadable file (caller decides how to count it)."""
    path = str(path)
    try:
        import scipy.io
        info = scipy.io.whosmat(path)          # [(name, shape, dtype)]; no data
    except NotImplementedError:
        return _read_hdf5_header(path)          # v7.3 (HDF5)
    shapes = {name: shape for (name, shape, _dt) in info}
    if "sbuf" in shapes:
        shape = shapes["sbuf"]
        small = scipy.io.loadmat(path, variable_names=["fs"], squeeze_me=True)
        raw_fs = small.get("fs")
        fs = float(raw_fs) if raw_fs is not None else 20000.0
    elif "data" in shapes:
        shape = shapes["data"]
        small = scipy.io.loadmat(path, variable_names=["chunkInfo"],
                                 squeeze_me=True, struct_as_record=False)
        ci = small.get("chunkInfo", None)
        if isinstance(ci, dict):
            fs = float(ci.get("samplingRate", 20000))
        elif ci is not None:
            fs = float(getattr(ci, "samplingRate", 20000))
        else:
            fs = 20000.0
    else:
        raise ValueError(f"Unknown .mat format. Vars: {list(shapes)}")
    num_samples, num_channels = _orient_dims(shape)
    return num_samples, num_channels, (fs if fs and fs > 0 else 20000.0)


def _read_hdf5_header(path: str) -> tuple[int, int, float]:
    """Header-only ``(num_samples, num_channels, fs)`` for a v7.3 (HDF5) .mat:
    h5py exposes dataset .shape as metadata, so no signal array is read."""
    import h5py
    with h5py.File(path, "r") as f:
        key = "sbuf" if "sbuf" in f else ("data" if "data" in f else None)
        if key is None:
            raise ValueError(f"Unknown v7.3 .mat format. Keys: {list(f.keys())}")
        shape = f[key].shape                    # metadata only -- no data read
        fs = None
        if "fs" in f:
            fs = float(np.squeeze(f["fs"][()]))
        elif ("chunkInfo" in f and isinstance(f["chunkInfo"], h5py.Group)
              and "samplingRate" in f["chunkInfo"]):
            fs = float(np.squeeze(f["chunkInfo"]["samplingRate"][()]))
    num_samples, num_channels = _orient_dims(shape)
    return num_samples, num_channels, (fs if fs and fs > 0 else 20000.0)


def _parse_fnstr(fnstr, num_channels: int) -> list:
    """Extract per-channel names from a MATLAB fnstr field. Trim to num_channels."""
    if fnstr is None:
        return []
    names: list = []
    try:
        if isinstance(fnstr, str):
            names = [fnstr]
        elif hasattr(fnstr, "__len__"):
            for item in fnstr:
                if isinstance(item, np.ndarray):
                    names.append(str(item.flat[0]) if item.size > 0 else "")
                else:
                    names.append(str(item))
    except (AttributeError, TypeError):
        return []
    return [n.strip() for n in names[:num_channels]]


def _load_hdf5(path: str) -> dict:
    """Fallback loader for MATLAB v7.3 (HDF5) files."""
    import h5py
    result = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key.startswith("#"):
                continue
            ds = f[key]
            if isinstance(ds, h5py.Dataset):
                val = ds[()]
                if val.dtype.kind == "O":
                    continue  # skip complex object refs
                result[key] = np.squeeze(val)
            elif isinstance(ds, h5py.Group):
                result[key] = _load_hdf5_group(ds)
    return result


def _load_hdf5_group(group) -> dict:
    """Recursively load HDF5 group into dict."""
    import h5py
    result = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = np.squeeze(item[()])
        elif isinstance(item, h5py.Group):
            result[key] = _load_hdf5_group(item)
    return result


def _parse_trdata(trdata_raw) -> dict:
    """Parse MATLAB trdata struct array into dict of channel timestamps."""
    result = {}
    try:
        if hasattr(trdata_raw, "__len__"):
            for i, ch in enumerate(trdata_raw):
                if hasattr(ch, "timestamp"):
                    result[i] = np.asarray(ch.timestamp, dtype=np.float64)
        elif hasattr(trdata_raw, "timestamp"):
            result[0] = np.asarray(trdata_raw.timestamp, dtype=np.float64)
    except (AttributeError, TypeError):
        pass
    return result
