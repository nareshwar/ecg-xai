"""
ecgxai.selection

Utilities to:
1) Build a selection DataFrame (sel_df) containing example ECG records per target class.
2) Build a multi-hot y_true matrix from PhysioNet-style DX strings.

Selection contract
------------------
build_selection_df_with_aliases(...) returns a DataFrame with at least:
- group_class   : canonical target key from target_meta (e.g. "164889003")
- filename      : record path (usually to .mat, but can be base path)
- sel_idx       : integer index into ecg_filenames/probs arrays
- duration_sec  : duration estimated from .hea (optional, if duration filtering enabled)
- prob_meta     : meta-probability used for ranking/selection (max over aliases)

Notes
-----
- "target_meta" maps a canonical meta_key -> {"name": str, "aliases": [codes...]}
- "aliases" are SNOMED codes that appear in class_names (model output columns).
- If y_true is given, selection prefers true positives (positive for ANY alias).
- Duration filtering reads nsamp and fs from the first line of .hea.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Segment = Tuple[int, int]


# ---------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------
def estimate_duration_sec_from_header(mat_path: str, default_fs: float = 500.0) -> Optional[float]:
    """Estimate ECG duration (seconds) using the matching PhysioNet .hea header.

    Reads the first header line, which is typically:
        <record> <n_sig> <fs> <n_samples> ...

    Args:
        mat_path: Path to the record .mat file (or anything with .mat suffix).
        default_fs: Used if fs parsing fails but nsamp is parseable.

    Returns:
        duration_sec = nsamp / fs, or None if header not found / unparseable.
    """
    hea_path = Path(mat_path).with_suffix(".hea")
    if not hea_path.exists():
        return None

    try:
        first = hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
    except Exception:
        return None

    parts = first.split()
    if len(parts) < 4:
        return None

    try:
        fs = float(parts[2])
        nsamp = int(parts[3])
    except Exception:
        # fallback: still try nsamp, use default fs
        try:
            nsamp = int(parts[3])
            fs = float(default_fs)
        except Exception:
            return None

    if fs <= 0:
        return None
    return float(nsamp) / float(fs)


def _load_or_compute_durations(
    ecg_filenames: np.ndarray,
    *,
    max_duration_sec: float,
    duration_cache_path: Optional[str],
    default_fs: float = 500.0,
) -> np.ndarray:
    """Compute (or load cached) durations array aligned to ecg_filenames."""
    N = len(ecg_filenames)

    durations: Optional[np.ndarray] = None
    if duration_cache_path and os.path.exists(duration_cache_path):
        try:
            durations = np.load(duration_cache_path)
            if durations.shape[0] != N:
                print("[WARN] duration cache has wrong length; recomputing.")
                durations = None
        except Exception:
            durations = None

    if durations is None:
        durations = np.full(N, np.nan, dtype=float)
        for i, fpath in enumerate(ecg_filenames):
            durations[i] = estimate_duration_sec_from_header(str(fpath), default_fs=default_fs)

        if duration_cache_path:
            try:
                np.save(duration_cache_path, durations)
                print(f"[INFO] Saved duration cache to {duration_cache_path}")
            except Exception as e:
                print(f"[WARN] Failed to save duration cache to {duration_cache_path}: {e}")

    print(f"[INFO] Duration filter enabled: keeping ECGs <= {max_duration_sec:.1f} s")
    return durations


# ---------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------
def build_selection_df_with_aliases(
    ecg_filenames: Sequence[str],
    probs: np.ndarray,
    class_names: Sequence[str],
    target_meta: Mapping[str, Mapping[str, Any]],
    y_true: Optional[np.ndarray] = None,
    *,
    k_per_class: int = 5,
    min_prob: float = 0.85,
    random_seed: int = 42,
    max_duration_sec: Optional[float] = 20.0,
    duration_cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """Select up to k_per_class ECGs per canonical meta-class in target_meta.

    Selection logic:
    - meta-probability is max(prob[:, alias_idx]) across aliases for the meta-key.
    - initial candidate set:
        (prob_meta >= min_prob) AND duration_ok
        and if y_true is provided, ALSO requires meta-label positive.
    - if candidates < k_per_class:
        relax by taking top prob_meta among duration_ok (ignoring y_true constraint)
    - final selection picks `take` examples uniformly at random from candidates.

    Args:
        ecg_filenames: List/array of record paths aligned with probs rows.
        probs: (N, C) float array of model probabilities.
        class_names: length C list of model output SNOMED codes (strings).
        target_meta: dict meta_key -> {"name": str, "aliases": list[str]}.
        y_true: Optional (N, C) int array. If given, prefer true positives.
        k_per_class: Max examples per meta_key.
        min_prob: High-confidence threshold for initial candidate filter.
        random_seed: RNG seed for reproducible random selection.
        max_duration_sec: If not None, filter by header-estimated duration.
        duration_cache_path: Optional .npy file path to cache durations.

    Returns:
        DataFrame with one row per selected ECG.
    """
    rng = np.random.default_rng(int(random_seed))

    ecg_filenames = np.asarray(ecg_filenames, dtype=object)
    probs = np.asarray(probs)
    class_names = np.asarray(class_names, dtype=str)

    N = len(ecg_filenames)
    if probs.shape[0] != N:
        raise ValueError(f"probs.shape[0] ({probs.shape[0]}) != len(ecg_filenames) ({N})")
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N,C), got shape {probs.shape}")

    if y_true is not None:
        y_true = np.asarray(y_true).astype(int)
        if y_true.shape != probs.shape:
            raise ValueError(f"y_true shape {y_true.shape} must match probs shape {probs.shape}")

    # Precompute code -> column index for speed
    code_to_col = {str(code): int(i) for i, code in enumerate(class_names)}

    # Duration mask
    if max_duration_sec is not None:
        durations = _load_or_compute_durations(
            ecg_filenames,
            max_duration_sec=float(max_duration_sec),
            duration_cache_path=duration_cache_path,
        )
        dur_mask = np.isfinite(durations) & (durations <= float(max_duration_sec))
        print(f"[INFO] Duration filter: keeping {int(dur_mask.sum())}/{N} ECGs.")
        if int(dur_mask.sum()) == 0:
            print("[WARN] No ECGs passed the duration filter – selection will be empty.")
    else:
        durations = np.full(N, np.nan, dtype=float)
        dur_mask = np.ones(N, dtype=bool)

    rows: List[Dict[str, Any]] = []

    for meta_key, cfg in target_meta.items():
        meta_key = str(meta_key)
        name = str(cfg.get("name", meta_key))
        aliases = [str(a) for a in cfg.get("aliases", [])]

        # Resolve alias columns
        alias_idx = [code_to_col[a] for a in aliases if a in code_to_col]
        if not alias_idx:
            print(f"[WARN] No alias columns for meta-class {meta_key} ({name})")
            continue

        alias_idx_arr = np.asarray(alias_idx, dtype=int)

        # Meta-probability (max over aliases)
        prob_alias = probs[:, alias_idx_arr]        # (N, n_alias)
        prob_meta = prob_alias.max(axis=1)          # (N,)

        allowed_mask = dur_mask.copy()
        high_prob = prob_meta >= float(min_prob)

        if y_true is not None:
            y_alias = y_true[:, alias_idx_arr]          # (N, n_alias)
            y_meta = (y_alias.sum(axis=1) > 0)          # (N,)
            mask = high_prob & y_meta & allowed_mask
        else:
            mask = high_prob & allowed_mask

        cand_idx = np.where(mask)[0]

        # If not enough candidates, relax threshold but still respect duration
        if cand_idx.size < int(k_per_class):
            print(f"[INFO] relaxing selection for {meta_key} ({name})")
            eligible_idx = np.where(allowed_mask)[0]
            if eligible_idx.size == 0:
                print(f"[WARN] no eligible ECGs (duration) for {meta_key} ({name})")
                continue

            order_local = np.argsort(-prob_meta[eligible_idx])  # descending
            cand_idx = eligible_idx[order_local][: max(int(k_per_class), cand_idx.size)]

        if cand_idx.size == 0:
            print(f"[WARN] no candidates for {meta_key} ({name}) after filtering.")
            continue

        take = min(int(k_per_class), int(cand_idx.size))
        chosen = rng.choice(cand_idx, size=take, replace=False)

        print(f"[CLASS {meta_key} ({name})] picked {take} examples.")

        for i in chosen:
            rows.append(
                {
                    "group_class": meta_key,
                    "filename": ecg_filenames[int(i)],
                    "sel_idx": int(i),
                    "duration_sec": float(durations[int(i)]) if max_duration_sec is not None else None,
                    "prob_meta": float(prob_meta[int(i)]),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Label parsing -> y_true
# ---------------------------------------------------------------------
_DX_SPLIT_RE = re.compile(r"[,\s;]+")


def parse_dx_codes(raw_label: Any) -> List[str]:
    """Parse a PhysioNet-style DX string into a list of SNOMED codes.

    Examples:
        "426783006, 164889003"
        "426783006"
    """
    if raw_label is None:
        return []
    s = str(raw_label).strip()
    if not s:
        return []
    return [t for t in _DX_SPLIT_RE.split(s) if t]


def build_y_true_from_labels(labels: Sequence[Any], class_names: Sequence[str]) -> np.ndarray:
    """Build a multi-hot y_true matrix aligned with class_names.

    Args:
        labels: list of raw DX strings.
        class_names: SNOMED codes for each model output (length C).

    Returns:
        y_true: (N, C) int8 array with 0/1 entries.
    """
    labels = list(labels)
    class_names = np.asarray(class_names, dtype=str)

    N = len(labels)
    C = len(class_names)
    y_true = np.zeros((N, C), dtype=np.int8)

    code_to_col = {str(code): int(i) for i, code in enumerate(class_names)}

    for i, raw in enumerate(labels):
        for code in parse_dx_codes(raw):
            j = code_to_col.get(str(code))
            if j is not None:
                y_true[int(i), int(j)] = 1

    return y_true
