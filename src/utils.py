import numpy as np

from pathlib import Path
from typing import Tuple, List

import numpy as np
from scipy.io import loadmat


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
