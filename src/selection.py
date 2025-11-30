from __future__ import annotations
from typing import Dict, Sequence

import numpy as np
import pandas as pd


def build_selection_df_with_aliases(
    ecg_filenames: Sequence[str],
    probs: np.ndarray,
    class_names: Sequence[str],
    target_meta: Dict[str, dict],
    y_true: np.ndarray | None = None,
    k_per_class: int = 5,
    min_prob: float = 0.85,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Select up to k_per_class ECGs per meta-class defined in target_meta.

    target_meta:
        meta_key -> {"name": ..., "aliases": [SNOMED codes in class_names]}

    If y_true is provided (N, C), we prefer true positives.
    Otherwise selection uses predictions only.

    Returns:
        sel_df with columns:
            group_class : primary SNOMED (e.g. '17338001')
            filename    : path to .mat
            sel_idx     : global index of ECG in ecg_filenames/probs
    """
    rng = np.random.default_rng(random_seed)

    ecg_filenames = np.asarray(ecg_filenames, dtype=object)
    probs = np.asarray(probs)
    class_names = np.asarray(class_names).astype(str)

    if y_true is not None:
        y_true = np.asarray(y_true).astype(int)

    rows = []

    for meta_key, cfg in target_meta.items():
        aliases = [str(a) for a in cfg["aliases"]]

        # which indices in class_names?
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
        prob_alias = probs[:, alias_idx]      # (N, n_alias)
        prob_meta = prob_alias.max(axis=1)    # (N,)

        # Candidate mask = high prob
        high_prob = prob_meta >= float(min_prob)

        if y_true is not None:
            # meta-label: OR over alias labels
            y_alias = y_true[:, alias_idx]         # (N, n_alias)
            y_meta = (y_alias.sum(axis=1) > 0)     # (N,)
            # Prefer high-probabilities that are actually positive
            mask = high_prob & y_meta
        else:
            # No ground truth? Use predictions only.
            mask = high_prob

        cand_idx = np.where(mask)[0]

        # If not enough, fall back to just top predicted by prob_meta
        if cand_idx.size < k_per_class:
            print(f"[INFO] relaxing selection for {meta_key} ({cfg['name']})")
            # rank all by prob_meta descending
            order = np.argsort(-prob_meta)
            cand_idx = order[:max(k_per_class, cand_idx.size)]

        take = min(k_per_class, cand_idx.size)
        chosen = rng.choice(cand_idx, size=take, replace=False)

        print(f"[CLASS {meta_key} ({cfg['name']})] picked {take} examples.")

        for i in chosen:
            rows.append({
                "group_class": meta_key,       # primary SNOMED
                "filename": ecg_filenames[i],
                "sel_idx": int(i),
            })

    sel_df = pd.DataFrame(rows)
    return sel_df
