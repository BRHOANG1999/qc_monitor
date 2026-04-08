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


def load_mat(path: str) -> ChunkData:
    """Load a KMrecorder .mat file. Handles both v5 and v7.3 formats."""
    path = str(path)

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
        if chunk_info is not None:
            fs = float(getattr(chunk_info, "samplingRate", 20000))
        else:
            fs = 20000.0
        trdata = {}
    else:
        raise ValueError(f"Unknown .mat format. Keys: {list(data.keys())}")

    if sbuf.ndim == 1:
        sbuf = sbuf.reshape(-1, 1)

    num_samples, num_channels = sbuf.shape
    duration_sec = num_samples / fs

    return ChunkData(
        signal=sbuf,
        fs=fs,
        num_channels=num_channels,
        num_samples=num_samples,
        duration_sec=duration_sec,
        source_path=path,
        trdata=trdata,
    )


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
