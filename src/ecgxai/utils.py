from __future__ import annotations

"""
ecgxai.utils

Shared utilities for:
- PhysioNet/CinC 2020 style header parsing (.hea)
- lightweight weighted ridge regression (numpy-only)
- saving/loading experiment runs (parquet + joblib + json)

This file is intentionally dependency-light and reusable across pipelines.
"""

import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

PathLike = Union[str, Path]

__all__ = [
    # physionet i/o
    "import_key_data",
    "load_physionet_data",
    "parse_header",
    "normalize_sex",
    "coerce_age",
    # maths helpers
    "cosine_distance_to_ones",
    "weighted_ridge_fit",
    "class_index",
    # run persistence
    "save_run",
    "load_run",
]


# ---------------------------------------------------------------------
# PhysioNet-style record indexing / loading
# ---------------------------------------------------------------------
def import_key_data(
    root: PathLike,
    *,
    show_progress: bool = True,
    desc: str = "Indexing ECG headers",
) -> Tuple[List[Optional[str]], List[Optional[int]], List[Optional[str]], List[str]]:
    """Scan a PhysioNet-style directory and extract basic metadata from headers.

    This function walks the directory for `.hea` files, reads their metadata, and
    only keeps records that have a matching `.mat` file.

    Args:
        root: Base directory containing PhysioNet-style ECG records.
        show_progress: If True, show a tqdm progress bar.
        desc: Label for the progress bar.

    Returns:
        gender_list: List of normalized sex values ('M'/'F'/None).
        age_list: List of ages (int or None).
        labels: List of raw DX strings (e.g., "164889003,426783006") or None.
        ecg_files: List of `.mat` paths (strings), aligned with the lists above.
    """
    root = Path(root)

    gender_list: List[Optional[str]] = []
    age_list: List[Optional[int]] = []
    labels: List[Optional[str]] = []
    ecg_files: List[str] = []

    hea_files = sorted(root.rglob("*.hea"))
    iterator = tqdm(hea_files, desc=desc) if show_progress else hea_files

    for hea_path in iterator:
        mat_path = hea_path.with_suffix(".mat")
        if not mat_path.exists():
            continue

        sex, age, label, _signal_len, _fs = parse_header(hea_path)

        gender_list.append(sex)
        age_list.append(age)
        labels.append(label)
        ecg_files.append(str(mat_path))

    return gender_list, age_list, labels, ecg_files


def load_physionet_data(path: PathLike) -> Tuple[np.ndarray, List[str]]:
    """Load a PhysioNet-style ECG record and its header.

    Args:
        path: Path to a record base name or file.
            Examples:
              - "A0001" (will load "A0001.mat" + "A0001.hea")
              - "A0001.mat" (will load "A0001.mat" + "A0001.hea")

    Returns:
        data: ECG array of shape (n_leads, n_samples), dtype float64.
        header_lines: Raw lines from the corresponding .hea file.

    Raises:
        FileNotFoundError: If the .mat or .hea file is missing.
        KeyError: If the MAT file doesn't contain the expected 'val' key.
    """
    path = Path(path)
    mat_path = path.with_suffix(".mat")
    hea_path = path.with_suffix(".hea")

    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    if not hea_path.exists():
        raise FileNotFoundError(f"Header file not found: {hea_path}")

    x = loadmat(mat_path)
    if "val" not in x:
        raise KeyError(f"'val' key not found in {mat_path}")
    data = np.asarray(x["val"], dtype=np.float64)

    # Robust decoding
    try:
        header_lines = hea_path.read_text(encoding="utf-8").splitlines(True)
    except UnicodeDecodeError:
        header_lines = hea_path.read_text(encoding="latin-1").splitlines(True)

    return data, header_lines


# ---------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------
def cosine_distance_to_ones(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Cosine distance between each row of M and an all-ones vector.

    Args:
        M: Array of shape (N, D).
        eps: Small constant for numerical stability.

    Returns:
        distances: Array of shape (N,), where 0 means identical to all-ones.
    """
    M = np.asarray(M, dtype=np.float64)
    ones = np.ones((M.shape[1],), dtype=np.float64)

    dot = M @ ones
    norm_M = np.linalg.norm(M, axis=1)
    norm_1 = np.linalg.norm(ones)

    cos_sim = dot / (norm_M * norm_1 + eps)
    return 1.0 - cos_sim


def weighted_ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    alpha: float = 1e-3,
    fit_intercept: bool = True,
) -> Tuple[np.ndarray, float]:
    """Fit weighted ridge regression using closed-form normal equations (numpy-only).

    This is used by both your LIME and TimeSHAP-style explainers to fit a local
    linear surrogate model on binary masks.

    Args:
        X: Design matrix (N, P).
        y: Target vector (N,) or (N,1).
        sample_weight: Optional weights (N,). If None, all ones.
        alpha: L2 regularisation strength. (Intercept is not regularised.)
        fit_intercept: If True, include an intercept term.

    Returns:
        coef: Float32 array of shape (P,).
        intercept: Intercept term (float).

    Notes:
        - Uses sqrt-weighting to convert to an unweighted least squares system.
        - Falls back to lstsq if `solve` fails (e.g., near-singular matrix).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,P). Got shape {X.shape}")
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"y length must match X rows. Got {y.shape[0]} vs {n}")

    if sample_weight is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).ravel()
        if w.shape[0] != n:
            raise ValueError(f"sample_weight length must match X rows. Got {w.shape[0]} vs {n}")
        w = np.clip(w, 1e-12, None)

    if fit_intercept:
        X_aug = np.column_stack([np.ones(n, dtype=np.float64), X])
        p_aug = p + 1
    else:
        X_aug = X
        p_aug = p

    sw = np.sqrt(w)[:, None]
    Xw = X_aug * sw
    yw = y * np.sqrt(w)

    A = Xw.T @ Xw
    if alpha and alpha > 0:
        reg = np.eye(p_aug, dtype=np.float64) * float(alpha)
        if fit_intercept:
            reg[0, 0] = 0.0  # do not regularise intercept
        A = A + reg

    b = Xw.T @ yw

    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinalgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]

    if fit_intercept:
        intercept = float(beta[0])
        coef = beta[1:].astype(np.float32)
    else:
        intercept = 0.0
        coef = beta.astype(np.float32)

    return coef, intercept


def class_index(class_names: Sequence[Any], key: Any) -> int:
    """Return the index of `key` in `class_names` (string-matched).

    Args:
        class_names: Sequence of class identifiers (strings/ints/etc.).
        key: Class identifier to find.

    Returns:
        Index of the first match.

    Raises:
        ValueError: If key is not present.
    """
    arr = np.asarray(class_names).astype(str)
    idx = np.where(arr == str(key))[0]
    if len(idx) == 0:
        raise ValueError(f"{key!r} not found in class_names.")
    return int(idx[0])


# ---------------------------------------------------------------------
# Header parsing helpers (PhysioNet Challenge ECG .hea files)
# ---------------------------------------------------------------------
SEX_RE = re.compile(r"(?i)#\s*sex\s*:\s*([A-Za-z?]+)")
AGE_RE = re.compile(r"(?i)#\s*age\s*:\s*([0-9]+|NaN|\?)")
DX_RE = re.compile(r"(?i)#\s*dx\s*:\s*(.+)$")


def coerce_age(val: Optional[str]) -> Optional[int]:
    """Convert age string from header to int, or return None if missing/invalid."""
    if val is None:
        return None
    v = str(val).strip()
    if v.lower() in {"nan", "?", ""}:
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return int(float(v))
        except Exception:
            return None


def normalize_sex(val: Optional[str]) -> Optional[str]:
    """Normalize sex/gender from header to 'M' / 'F' / None."""
    if not val:
        return None
    v = str(val).strip().upper()
    if v in {"M", "MALE"}:
        return "M"
    if v in {"F", "FEMALE"}:
        return "F"
    return None


def parse_header(
    hea_path: PathLike,
) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[int], Optional[int]]:
    """Parse a PhysioNet-style .hea header file.

    Returns:
        sex: 'M' / 'F' / None
        age: int or None
        label: raw dx string (e.g. "164889003, 426783006") or None
        signal_len: number of samples (nsamp) or None
        fs: sampling frequency in Hz or None
    """
    hea_path = Path(hea_path)

    sex: Optional[str] = None
    age: Optional[int] = None
    label: Optional[str] = None
    signal_len: Optional[int] = None
    fs: Optional[int] = None

    if not hea_path.exists():
        return sex, age, label, signal_len, fs

    # robust decoding
    try:
        lines = hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except UnicodeDecodeError:
        lines = hea_path.read_text(encoding="latin-1", errors="ignore").splitlines()

    if not lines:
        return sex, age, label, signal_len, fs

    # First line often: "<recname> <n_sig> <fs> <nsamp> ..."
    parts = lines[0].strip().split()
    if len(parts) >= 4:
        try:
            fs = int(float(parts[2]))
        except Exception:
            fs = None
        try:
            signal_len = int(parts[3])
        except Exception:
            signal_len = None

    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("#"):
            continue

        m = SEX_RE.search(ln)
        if m and sex is None:
            sex = normalize_sex(m.group(1))

        m = AGE_RE.search(ln)
        if m and age is None:
            age = coerce_age(m.group(1))

        m = DX_RE.search(ln)
        if m and label is None:
            label = m.group(1).strip()

    return sex, age, label, signal_len, fs


# ---------------------------------------------------------------------
# Run persistence (parquet + joblib + json)
# ---------------------------------------------------------------------
def save_run(
    run_dir: PathLike,
    all_fused_payloads: Any,
    df_lime_all: pd.DataFrame,
    df_ts_all: pd.DataFrame,
    sel_df: pd.DataFrame,
    *,
    meta: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save a full experiment run to disk.

    Files created:
        - df_lime_all.parquet
        - df_ts_all.parquet
        - sel_df.parquet
        - all_fused_payloads.joblib
        - meta.json

    Args:
        run_dir: Output directory.
        all_fused_payloads: Nested payload structure (joblib-serialised).
        df_lime_all: LIME output table.
        df_ts_all: TimeSHAP output table.
        sel_df: Selection table used to choose ECGs.
        meta: Optional JSON-serialisable metadata dict.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    df_lime_all.to_parquet(run_dir / "df_lime_all.parquet", index=False)
    df_ts_all.to_parquet(run_dir / "df_ts_all.parquet", index=False)
    sel_df.to_parquet(run_dir / "sel_df.parquet", index=False)

    joblib.dump(all_fused_payloads, run_dir / "all_fused_payloads.joblib", compress=3)

    meta_obj = dict(meta) if meta is not None else {}
    (run_dir / "meta.json").write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")


def load_run(
    run_dir: PathLike,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a saved run from disk.

    Args:
        run_dir: Directory created by `save_run()`.

    Returns:
        all_fused_payloads, df_lime_all, df_ts_all, sel_df
    """
    run_dir = Path(run_dir)

    df_lime_all = pd.read_parquet(run_dir / "df_lime_all.parquet")
    df_ts_all = pd.read_parquet(run_dir / "df_ts_all.parquet")
    sel_df = pd.read_parquet(run_dir / "sel_df.parquet")
    all_fused_payloads = joblib.load(run_dir / "all_fused_payloads.joblib")

    return all_fused_payloads, df_lime_all, df_ts_all, sel_df
