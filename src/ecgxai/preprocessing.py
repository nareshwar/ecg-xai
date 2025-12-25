import os
import numpy as np
from keras.utils import pad_sequences

from config import MAXLEN
from utils import load_physionet_data

def preprocess_for_model(mat_path, maxlen=MAXLEN):
    """
    Load ECG and return (T, F) = (maxlen, 12) in the same format
    your model was trained on.
    """
    data, header = load_physionet_data(str(mat_path))  # (12, T)
    data_p = pad_sequences(data, maxlen=maxlen, truncating="post", padding="post")  # (12, maxlen)
    x = np.asarray(data_p).T
    return x.astype(np.float32, copy=False)


def load_mat_TF(mat_path):
    """Alias used by the rest of the code."""
    return preprocess_for_model(mat_path)


def ensure_paths(filename):
    """
    Given any of .mat or .hea, return (hea_path, mat_path).
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


def _read_header_lines(hea_path):
    try:
        with open(hea_path, "r", encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    except UnicodeDecodeError:
        with open(hea_path, "r", encoding="latin-1") as f:
            return [ln.rstrip("\n") for ln in f]


def infer_fs_from_header(hea_path, default=500.0):
    import os
    if not os.path.exists(hea_path):
        return float(default)

    lines = _read_header_lines(hea_path)
    if not lines:
        return float(default)

    toks = lines[0].split()
    for tok in toks:
        try:
            x = float(tok)
        except Exception:
            continue
        if 50 <= x <= 2000:
            return float(x)
    return float(default)


def parse_fs_and_leads(hea_path, default_fs=500.0):
    """
    Return (fs, lead_names) from a .hea file.
    lead_names may be None if not parseable.
    """
    import os
    if not os.path.exists(hea_path):
        return float(default_fs), None

    lines = _read_header_lines(hea_path)
    if not lines:
        return float(default_fs), None

    fs = infer_fs_from_header(hea_path, default_fs)

    parts = lines[0].split()
    lead_names = None
    if len(parts) >= 2 and parts[1].isdigit():
        n_leads = int(parts[1])
        if len(lines) >= 1 + n_leads:
            lead_names = [lines[i].split()[-1] for i in range(1, 1 + n_leads)]

    return float(fs), lead_names
