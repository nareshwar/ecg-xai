"""
ecgxai.preprocessing

Preprocessing helpers that convert raw PhysioNet/CinC ECG records into the
exact tensor format expected by the trained model.

Core contract:
- Input: a PhysioNet-style record (.mat + .hea) path (either extension is OK).
- Output: x of shape (T, F) where:
    T = maxlen (time samples after pad/truncate)
    F = number of leads (typically 12)

Notes:
- load_physionet_data() returns data shaped (F, T_raw)
- We pad/truncate each lead independently to maxlen, then transpose to (T, F).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from keras.utils import pad_sequences

from .config import MAXLEN
from .utils import load_physionet_data

PathLike = Union[str, Path]


def preprocess_for_model(record_path: PathLike, maxlen: int = MAXLEN) -> np.ndarray:
    """Load and preprocess one ECG record into model-ready (T, F).

    Args:
        record_path: Path to a PhysioNet record. Can be:
            - "A0001" (base name)
            - "A0001.mat"
            - "A0001.hea"
        maxlen: Target time length in samples. Signal is truncated/padded at the end
            ("post") to reach this length.

    Returns:
        x: Float32 array of shape (maxlen, F). For 12-lead ECGs, F=12.

    Raises:
        ValueError: If maxlen is not positive.
        FileNotFoundError / KeyError: Propagated from load_physionet_data() when files are missing.
    """
    if maxlen <= 0:
        raise ValueError(f"maxlen must be > 0, got {maxlen}")

    data, _header = load_physionet_data(str(record_path))  # (F, T_raw)

    # Pad/truncate each lead (sequence) to maxlen -> (F, maxlen)
    data_p = pad_sequences(
        data, maxlen=maxlen, truncating="post", padding="post"
    )

    # Transpose to (T, F)
    x = np.asarray(data_p).T
    return x.astype(np.float32, copy=False)


def load_mat_TF(record_path: PathLike) -> np.ndarray:
    """Alias used by the rest of the codebase.

    Returns:
        Preprocessed ECG tensor of shape (MAXLEN, F), float32.
    """
    return preprocess_for_model(record_path)


def ensure_paths(filename: PathLike) -> Tuple[str, str]:
    """Given a record path, return (hea_path, mat_path) as strings.

    Accepts either '.mat', '.hea', or a base name without extension.

    Args:
        filename: Path-like record identifier.

    Returns:
        hea_path: Header path (string).
        mat_path: MAT path (string).
    """
    filename = str(filename)
    base, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == ".mat":
        mat_path = filename
        hea_path = base + ".hea"
    elif ext == ".hea":
        hea_path = filename
        mat_path = base + ".mat"
    else:
        mat_path = base + ".mat"
        hea_path = base + ".hea"

    return hea_path, mat_path


def _read_header_lines(hea_path: PathLike) -> List[str]:
    """Read .hea file lines with robust encoding."""
    hea_path = str(hea_path)
    try:
        with open(hea_path, "r", encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    except UnicodeDecodeError:
        with open(hea_path, "r", encoding="latin-1") as f:
            return [ln.rstrip("\n") for ln in f]


def infer_fs_from_header(hea_path: PathLike, default: float = 500.0) -> float:
    """Infer sampling frequency (Hz) from the first line of a PhysioNet .hea file.

    The first line is typically:
        <record> <n_sig> <fs> <n_samples> ...

    We first try to parse the canonical third token as fs. If that fails, we fall
    back to scanning tokens for a plausible fs.

    Args:
        hea_path: Path to .hea file.
        default: Used when file is missing/unparseable.

    Returns:
        Sampling frequency in Hz.
    """
    hea_path = str(hea_path)
    if not os.path.exists(hea_path):
        return float(default)

    lines = _read_header_lines(hea_path)
    if not lines:
        return float(default)

    parts = lines[0].split()

    # canonical: parts[2] is fs
    if len(parts) >= 3:
        try:
            fs = float(parts[2])
            if 50 <= fs <= 2000:
                return float(fs)
        except Exception:
            pass

    # fallback: scan for plausible numeric token
    for tok in parts:
        try:
            x = float(tok)
        except Exception:
            continue
        if 50 <= x <= 2000:
            return float(x)

    return float(default)


def parse_fs_and_leads(hea_path: PathLike, default_fs: float = 500.0) -> Tuple[float, Optional[List[str]]]:
    """Return (fs, lead_names) from a .hea header.

    Args:
        hea_path: Path to .hea file.
        default_fs: Used when header is missing/unparseable.

    Returns:
        fs: Sampling frequency in Hz.
        lead_names: List of lead names if parseable, else None.

    Notes:
        This assumes the header has n_leads on the first line, and that the next
        n_leads lines each end with the lead name token.
    """
    hea_path = str(hea_path)
    if not os.path.exists(hea_path):
        return float(default_fs), None

    lines = _read_header_lines(hea_path)
    if not lines:
        return float(default_fs), None

    fs = infer_fs_from_header(hea_path, default_fs)

    parts = lines[0].split()
    lead_names: Optional[List[str]] = None

    # first line: parts[1] is often n_leads
    if len(parts) >= 2 and parts[1].isdigit():
        n_leads = int(parts[1])
        if len(lines) >= 1 + n_leads:
            lead_names = [lines[i].split()[-1] for i in range(1, 1 + n_leads)]

    return float(fs), lead_names
