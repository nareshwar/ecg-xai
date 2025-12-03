from __future__ import annotations
from typing import Dict, Sequence, Tuple

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _estimate_duration_sec_from_header(mat_path: str, default_fs: float = 500.0) -> float | None:
    """
    Given a .mat path, look for the matching .hea file and estimate
    duration = nsamp / fs from the first header line.

    Returns:
        duration_sec (float) or None if anything is missing.
    """
    hea_path = Path(mat_path).with_suffix(".hea")
    if not hea_path.exists():
        return None

    try:
        with hea_path.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
    except OSError:
        return None

    parts = first.split()
    if len(parts) < 4:
        return None

    try:
        fs = float(parts[2])
        nsamp = int(parts[3])
    except Exception:
        # fall back to default fs if fs parsing fails
        try:
            nsamp = int(parts[3])
            fs = default_fs
        except Exception:
            return None

    if fs <= 0:
        return None

    return nsamp / fs


def build_selection_df_with_aliases(
    ecg_filenames: Sequence[str],
    probs: np.ndarray,
    class_names: Sequence[str],
    target_meta: Dict[str, dict],
    y_true: np.ndarray | None = None,
    k_per_class: int = 5,
    min_prob: float = 0.85,
    random_seed: int = 42,
    max_duration_sec: float | None = 20.0,
    duration_cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Select up to k_per_class ECGs per meta-class defined in target_meta.

    Only ECGs with duration <= max_duration_sec are considered
    (if max_duration_sec is not None).

    target_meta:
        meta_key -> {"name": ..., "aliases": [SNOMED codes in class_names]}

    If y_true is provided (N, C), we prefer true positives.
    Otherwise selection uses predictions only.

    Returns:
        sel_df with columns:
            group_class : meta key (e.g. '17338001' / SNOMED of interest)
            filename    : path to .mat
            sel_idx     : global index of ECG in ecg_filenames/probs
    """
    rng = np.random.default_rng(random_seed)

    ecg_filenames = np.asarray(ecg_filenames, dtype=object)
    probs = np.asarray(probs)
    class_names = np.asarray(class_names).astype(str)

    N = len(ecg_filenames)
    if probs.shape[0] != N:
        raise ValueError(f"probs.shape[0] ({probs.shape[0]}) != len(ecg_filenames) ({N})")

    if y_true is not None:
        y_true = np.asarray(y_true).astype(int)
        if y_true.shape[0] != N:
            raise ValueError(f"y_true.shape[0] ({y_true.shape[0]}) != len(ecg_filenames) ({N})")

    # -------------------------------------------------------
    # Duration filter: only keep ECGs with duration <= max_duration_sec
    # -------------------------------------------------------
    # -------------------------------------------------------
    # Duration filter: only keep ECGs with duration <= max_duration_sec
    # -------------------------------------------------------
    if max_duration_sec is not None:
        print(f"[INFO] Estimating durations and keeping ECGs <= {max_duration_sec:.1f} s...")

        durations = None

        # Try cache first, if provided
        if duration_cache_path is not None and os.path.exists(duration_cache_path):
            durations = np.load(duration_cache_path)
            if durations.shape[0] != N:
                print("[WARN] duration cache has wrong length; recomputing.")
                durations = None

        if durations is None:
            durations = np.empty(N, dtype=float)
            durations[:] = np.nan

            for i, fpath in enumerate(ecg_filenames):
                durations[i] = _estimate_duration_sec_from_header(str(fpath))

            if duration_cache_path is not None:
                np.save(duration_cache_path, durations)
                print(f"[INFO] Saved duration cache to {duration_cache_path}")

        # valid & <= max_duration_sec
        dur_mask = np.isfinite(durations) & (durations <= float(max_duration_sec))
        n_keep = int(dur_mask.sum())
        print(f"[INFO] Duration filter: keeping {n_keep}/{N} ECGs (<= {max_duration_sec:.1f} s).")

        if n_keep == 0:
            print("[WARN] No ECGs passed the duration filter – selection will be empty.")
    else:
        dur_mask = np.ones(N, dtype=bool)
        durations = np.full(N, np.nan, dtype=float)

    rows = []

    # -------------------------------------------------------
    # Loop over meta-classes (AF / sinus / VPB groups)
    # -------------------------------------------------------
    for meta_key, cfg in target_meta.items():
        aliases = [str(a) for a in cfg["aliases"]]

        # Which column indices in class_names correspond to these aliases?
        alias_idx = []
        for a in aliases:
            idx = np.where(class_names == a)[0]
            if idx.size > 0:
                alias_idx.append(int(idx[0]))

        if not alias_idx:
            print(f"[WARN] No alias columns for meta-class {meta_key} ({cfg['name']})")
            continue

        alias_idx = np.array(alias_idx, dtype=int)

        # meta-prob: max over all alias outputs
        prob_alias = probs[:, alias_idx]   # (N, n_alias)
        prob_meta = prob_alias.max(axis=1) # (N,)

        # Start with duration constraint
        allowed_mask = dur_mask.copy()

        # High-prob candidates
        high_prob = prob_meta >= float(min_prob)

        if y_true is not None:
            # meta-label: OR over alias labels
            y_alias = y_true[:, alias_idx]         # (N, n_alias)
            y_meta = (y_alias.sum(axis=1) > 0)     # (N,)
            # Prefer high-probabilities that are actually positive
            mask = high_prob & y_meta & allowed_mask
        else:
            # No ground truth? Use predictions + duration only.
            mask = high_prob & allowed_mask

        cand_idx = np.where(mask)[0]

        # ------------------------------------------------------------------
        # If not enough candidates, relax threshold but STILL respect duration
        # ------------------------------------------------------------------
        if cand_idx.size < k_per_class:
            print(f"[INFO] relaxing selection for {meta_key} ({cfg['name']})")
            # among allowed ones only
            eligible_idx = np.where(allowed_mask)[0]
            if eligible_idx.size == 0:
                print(f"[WARN] no eligible ECGs (duration) for {meta_key} ({cfg['name']})")
                continue

            # rank eligible ECGs by prob_meta descending
            order_local = np.argsort(-prob_meta[eligible_idx])
            order_global = eligible_idx[order_local]
            cand_idx = order_global[:max(k_per_class, cand_idx.size)]

        if cand_idx.size == 0:
            print(f"[WARN] no candidates at all for {meta_key} ({cfg['name']}) after duration filter.")
            continue

        take = min(k_per_class, cand_idx.size)
        chosen = rng.choice(cand_idx, size=take, replace=False)

        print(f"[CLASS {meta_key} ({cfg['name']})] picked {take} examples.")

        for i in chosen:
            rows.append({
                "group_class": meta_key,       # meta-key / primary SNOMED
                "filename": ecg_filenames[i],
                "sel_idx": int(i),
                "duration_sec": float(durations[i]) if max_duration_sec is not None else None,
                "prob_meta": float(prob_meta[i]),
            })

    sel_df = pd.DataFrame(rows)
    return sel_df
