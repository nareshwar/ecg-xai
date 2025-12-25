from __future__ import annotations

import numpy as np
import re
import json
import pandas as pd
import joblib
from scipy.io import loadmat
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Optional

def import_key_data(
    root: str | Path,
    *,
    show_progress: bool = True,
    desc: str = "Indexing ECG headers"
) -> Tuple[List[str], List[int], List[str], List[str]]:
    """
    Fast version: scan all .hea files, parse sex/age/diagnosis, and
    return lists aligned with .mat filenames.

    Args
    ----
    root : base directory containing PhysioNet-style ECG records
    show_progress : if True, show a tqdm progress bar with ETA
    desc : label for the progress bar

    Returns
    -------
    gender_list : list of 'M'/'F'/None (as strings; may include None)
    age_list    : list of age values (int or None)
    labels      : raw DX string (the 'dx:' line)
    ecg_files   : list of .mat paths (str)
    """
    root = Path(root)

    gender_list: List[str] = []
    age_list:    List[int] = []
    labels:      List[str] = []
    ecg_files:   List[str] = []

    # Collect all header files first so tqdm can show total + ETA
    hea_files = sorted(root.rglob("*.hea"))

    iterator = hea_files
    if show_progress:
        iterator = tqdm(hea_files, desc=desc)

    for hea_path in iterator:
        mat_path = hea_path.with_suffix(".mat")
        if not mat_path.exists():
            # No matching .mat -> skip
            continue

        # parse_header should be defined in utils as we discussed
        sex, age, label, signal_len, fs = parse_header(hea_path)

        gender_list.append(sex)
        age_list.append(age)
        labels.append(label)
        ecg_files.append(str(mat_path))

    return gender_list, age_list, labels, ecg_files


def load_physionet_data(path: str | Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load a PhysioNet-style ECG record.

    Returns
    -------
    data : np.ndarray
        ECG array of shape (n_leads, n_samples), dtype float64.
    header_lines : list[str]
        Lines from the corresponding .hea file.
    """
    path = Path(path)
    mat_path = path.with_suffix(".mat")
    hea_path = path.with_suffix(".hea")

    # --- load signal ---
    x = loadmat(mat_path)
    if "val" not in x:
        raise KeyError(f"'val' key not found in {mat_path}")
    data = np.asarray(x["val"], dtype=np.float64)

    # --- load header (robust encodings) ---
    try:
        with hea_path.open("r", encoding="utf-8") as f:
            header_lines = f.readlines()
    except UnicodeDecodeError:
        with hea_path.open("r", encoding="latin-1") as f:
            header_lines = f.readlines()

    return data, header_lines

def cosine_distance_to_ones(M, eps=1e-9):
    """
    Cosine distance between each row of M and an all-ones vector.
    """
    M = np.asarray(M, dtype=np.float64)
    ones = np.ones((M.shape[1],), dtype=np.float64)
    dot = M @ ones
    norm_M = np.linalg.norm(M, axis=1)
    norm_1 = np.linalg.norm(ones)
    cos_sim = dot / (norm_M * norm_1 + eps)
    return 1.0 - cos_sim

def weighted_ridge_fit(X, y, sample_weight=None, alpha=1e-3, fit_intercept=True):
    """
    Simple weighted ridge regression in pure numpy.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape

    if sample_weight is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).ravel()
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
        import numpy as _np
        reg = _np.eye(p_aug, dtype=np.float64) * float(alpha)
        if fit_intercept:
            reg[0, 0] = 0.0
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


def class_index(class_names, key):
    arr = np.asarray(class_names).astype(str)
    idx = np.where(arr == str(key))[0]
    if len(idx) == 0:
        raise ValueError(f"{key!r} not found in class_names.")
    return int(idx[0])

# ---------------------------------------------------------------------
# Header parsing helpers (for PhysioNet Challenge ECG .hea files)
# ---------------------------------------------------------------------

SEX_RE = re.compile(r'(?i)#\s*sex\s*:\s*([A-Za-z?]+)')
AGE_RE = re.compile(r'(?i)#\s*age\s*:\s*([0-9]+|NaN|\?)')
DX_RE  = re.compile(r'(?i)#\s*dx\s*:\s*(.+)$')


def coerce_age(val: Optional[str]) -> Optional[int]:
    """Convert age string from header to int (or None if missing/invalid)."""
    if val is None:
        return None
    v = str(val).strip()
    if v.lower() in {"nan", "?", ""}:
        return None
    try:
        return int(v)
    except ValueError:
        # sometimes age can look like a float; try that
        try:
            return int(float(v))
        except Exception:
            return None


def normalize_sex(val: Optional[str]) -> Optional[str]:
    """
    Normalize sex/gender from header to 'M' / 'F' / None.
    Returns None for unknown / other encodings.
    """
    if not val:
        return None
    v = str(val).strip().upper()
    if v in {"M", "MALE"}:
        return "M"
    if v in {"F", "FEMALE"}:
        return "F"
    return None  # Unknown/other


def parse_header(
    hea_path: str | Path,
) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[int], Optional[int]]:
    """
    Parse a PhysioNet-style .hea header file.

    Returns:
        sex        : 'M' / 'F' / None
        age        : int or None
        label      : raw dx string (e.g. "164889003, 426783006")
        signal_len : number of samples (nsamp) or None
        fs         : sampling frequency (Hz) or None
    """
    hea_path = Path(hea_path)

    sex: Optional[str] = None
    age: Optional[int] = None
    label: Optional[str] = None
    signal_len: Optional[int] = None
    fs: Optional[int] = None

    try:
        with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return sex, age, label, signal_len, fs

    if not lines:
        return sex, age, label, signal_len, fs

    # First line usually: "<recname> <n_sig> <fs> <nsamp> <date> <time>"
    first = lines[0].strip()
    parts = first.split()
    if len(parts) >= 4:
        # parts[2] = fs, parts[3] = number of samples
        try:
            fs = int(float(parts[2]))
        except Exception:
            fs = None
        try:
            signal_len = int(parts[3])
        except Exception:
            signal_len = None

    # Now scan all comment lines for sex / age / dx
    for raw in lines:
        ln = raw.strip()
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
            # full DX string as in the header
            label = m.group(1).strip()

    return sex, age, label, signal_len, fs

def save_run(run_dir, all_fused_payloads, df_lime_all, df_ts_all, sel_df, meta=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # DataFrames
    df_lime_all.to_parquet(run_dir / "df_lime_all.parquet", index=False)
    df_ts_all.to_parquet(run_dir / "df_ts_all.parquet", index=False)
    sel_df.to_parquet(run_dir / "sel_df.parquet", index=False)

    # Nested payloads
    joblib.dump(all_fused_payloads, run_dir / "all_fused_payloads.joblib", compress=3)

    # Metadata
    if meta is None:
        meta = {}
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_run(run_dir):
    run_dir = Path(run_dir)

    df_lime_all = pd.read_parquet(run_dir / "df_lime_all.parquet")
    df_ts_all = pd.read_parquet(run_dir / "df_ts_all.parquet")
    sel_df = pd.read_parquet(run_dir / "sel_df.parquet")
    all_fused_payloads = joblib.load(run_dir / "all_fused_payloads.joblib")

    return all_fused_payloads, df_lime_all, df_ts_all, sel_df

