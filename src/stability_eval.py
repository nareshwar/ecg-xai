"""stability_eval.py — Extra-beat augmentation + regionwise stability ONLY.

- Adds ONE duplicate beat by copying a randomly-selected beat (seeded), and inserting it
  immediately after that beat ("middle" insertion). We intentionally avoid edge beats.
- Saves augmented MATs to AUGMENT_ROOT (NOT into your dataset folder)
- Runs fused explainer for original + augmented
- Computes regionwise stability (Spearman + Jaccard@K, plus extras) while ignoring the extra beat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from scipy.io import savemat

from config import MAXLEN
from preprocessing import ensure_paths, parse_fs_and_leads
from config_targets import TARGET_META
from explainer import run_fused_pipeline_for_classes

# Import what we need from eval.py (core)
from eval import (
    LEADS12,
    load_mat_TF,
    detect_rpeaks,
    REGISTRY,
    build_windows_from_rpeaks,
    build_tokens,
    integrate_attribution,
    apply_sinus_prior_blend,
    apply_af_prior_blend,
    apply_vpb_prior_blend,
)

ROOT = Path.cwd().parent
AUGMENT_ROOT = Path( ROOT / "outputs" / "extra_beat_aug")  # change if you want


def _init_rng(
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, np.random.Generator]:
    """Return (seed, rng) where seed is always an int suitable for reproducing choices."""
    if rng is not None:
        # If caller supplies rng, we still want a seed value to record in filenames.
        if seed is None:
            # Draw a deterministic-ish integer from the provided RNG.
            seed = int(rng.integers(0, 2**31 - 1))
        return int(seed), rng

    if seed is None:
        # Generate a seed and record it, so outputs remain reproducible if re-run.
        seed = int(np.random.SeedSequence().entropy)
    return int(seed), np.random.default_rng(int(seed))


def _choose_non_edge_beat_index(
    x: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    *,
    pre_sec: float = 0.30,
    post_sec: float = 0.40,
    edge_beats: int = 1,
    lead_names: Sequence[str] = LEADS12,
    prefer: Sequence[str] = ("II", "V2", "V3"),
) -> int:
    """Pick a beat index (by R-peak list index) while avoiding edge beats.

    Strategy:
      1) Exclude the first/last `edge_beats` beats.
      2) Filter to beats whose extraction window stays in-bounds (no clipping).
      3) Fall back to the middle beat if candidates are empty.
    """
    x = np.asarray(x, dtype=np.float32)
    T = int(x.shape[0])
    r_idx = detect_rpeaks(x, fs, prefer=prefer, lead_names=lead_names)
    n = int(r_idx.size)
    if n == 0:
        raise ValueError("No R-peaks detected; cannot choose beat_index.")

    lo = int(max(0, edge_beats))
    hi = int(min(n, n - max(0, edge_beats)))
    candidates = list(range(lo, hi))
    if not candidates:
        return int(n // 2)

    good: List[int] = []
    for bi in candidates:
        r = int(r_idx[bi])
        s = int(round(r - pre_sec * fs))
        e = int(round(r + post_sec * fs))
        if (s >= 0) and (e <= T) and (e > s):
            good.append(int(bi))

    pool = good if good else candidates
    return int(rng.choice(pool))


def add_extra_beat(
    x: np.ndarray,
    fs: float,
    *,
    location: str = "end",
    beat_index: Optional[int] = None,
    pre_sec: float = 0.30,
    post_sec: float = 0.40,
    lead_names: Sequence[str] = LEADS12,
    prefer: Sequence[str] = ("II", "V2", "V3"),
    max_len: Optional[int] = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    T, F = x.shape

    r_idx = detect_rpeaks(x, fs, prefer=prefer, lead_names=lead_names)
    if r_idx.size == 0:
        raise ValueError("No R-peaks detected; cannot add extra beat.")

    if beat_index is None:
        beat_index = int(len(r_idx) // 2) if location == "middle" else (len(r_idx) - 1)

    if not (0 <= beat_index < len(r_idx)):
        raise IndexError(f"beat_index {beat_index} out of range for {len(r_idx)} beats.")

    r = int(r_idx[beat_index])
    s = max(0, int(round(r - pre_sec * fs)))
    e = min(T, int(round(r + post_sec * fs)))
    if e <= s:
        raise RuntimeError("Computed empty beat segment when extracting heartbeat.")

    beat_seg = x[s:e, :]

    insert_at = e if location == "middle" else T
    x_new = np.concatenate([x[:insert_at, :], beat_seg, x[insert_at:, :]], axis=0)

    if max_len is not None and x_new.shape[0] > max_len:
        x_new = x_new[:max_len, :]
    return x_new

def _match_beats_by_time(
    r_sec_a: Sequence[float],
    r_sec_b: Sequence[float],
    max_diff_sec: float = 0.08,
) -> List[Tuple[int, int]]:
    r_a = np.asarray(r_sec_a, dtype=float)
    r_b = np.asarray(r_sec_b, dtype=float)

    pairs: List[Tuple[int, int]] = []
    used_b = set()

    for i, t in enumerate(r_a):
        diffs = [(j, abs(float(r_b[j]) - float(t))) for j in range(len(r_b)) if j not in used_b]
        if not diffs:
            break
        j, dt = min(diffs, key=lambda p: p[1])
        if dt <= max_diff_sec:
            pairs.append((i, j))
            used_b.add(j)

    return pairs

def _ranks_with_ties(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.size
    order = np.argsort(values)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    for v in np.unique(values):
        idx = np.where(values == v)[0]
        if idx.size > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks

def _effective_k(n: int, k: int = 20, k_frac: float = 0.10, k_min: int = 10) -> int:
    """
    Pick a K that scales with the number of regions.
    Defaults: top 10%, at least 10, at most k.
    """
    if n <= 0:
        return 0
    return max(k_min, min(int(k), int(round(k_frac * n))))

def _topk_hard(scores: np.ndarray, k: int) -> set[int]:
    scores = np.asarray(scores, float)
    n = scores.size
    if n == 0 or k <= 0:
        return set()
    k = min(k, n)
    idx = np.argsort(scores)[::-1][:k]
    return set(idx.tolist())

def _topk_tie_aware(scores: np.ndarray, k: int) -> set[int]:
    """
    Include all items with score >= score_at_rank_k (so ties at the boundary don't jitter).
    """
    scores = np.asarray(scores, float)
    n = scores.size
    if n == 0 or k <= 0:
        return set()
    k = min(k, n)
    order = np.argsort(scores)[::-1]
    kth = scores[order[k - 1]]
    return set(np.where(scores >= kth)[0].tolist())


def _jaccard(a: set[int], b: set[int]) -> float:
    u = a | b
    if not u:
        return float("nan")
    return float(len(a & b) / len(u))


def _weighted_jaccard(scores_a: np.ndarray, scores_b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Overlap of attribution *mass* (0..1). More stable than top-K membership.
    """
    a = np.maximum(np.asarray(scores_a, float), 0.0)
    b = np.maximum(np.asarray(scores_b, float), 0.0)
    if a.size == 0 or b.size != a.size:
        return float("nan")
    num = np.minimum(a, b).sum()
    den = np.maximum(a, b).sum() + eps
    return float(num / den)


def _rbo_from_scores(scores_a: np.ndarray, scores_b: np.ndarray, p: float = 0.90, k: int | None = None) -> float:
    """
    Rank-Biased Overlap (0..1), rank-sensitive stability.
    p near 1.0 looks deeper into the ranking; p=0.9 is a common default.
    """
    a = np.asarray(scores_a, float)
    b = np.asarray(scores_b, float)
    n = a.size
    if n == 0 or b.size != n:
        return float("nan")
    if k is None:
        k = n
    k = min(k, n)

    ra = np.argsort(a)[::-1][:k]
    rb = np.argsort(b)[::-1][:k]

    seen_a, seen_b = set(), set()
    rbo_sum = 0.0
    for d in range(1, k + 1):
        seen_a.add(int(ra[d - 1]))
        seen_b.add(int(rb[d - 1]))
        overlap = len(seen_a & seen_b)
        rbo_sum += (overlap / d) * (p ** (d - 1))

    return float((1 - p) * rbo_sum)


def explanation_stability(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    k: int = 20,
    k_frac: float = 0.10,
    k_min: int = 10,
    rbo_p: float = 0.90,
) -> Dict[str, float]:
    """
    Returns:
      - spearman
      - jaccard_topk            (tie-aware)
      - jaccard_topk_hard       (old behaviour)
      - rbo                     (rank-sensitive)
      - weighted_jaccard        (mass overlap)
      - k_eff                   (actual K used)
    """
    s1 = np.asarray(scores_a, dtype=float)
    s2 = np.asarray(scores_b, dtype=float)
    if s1.shape != s2.shape or s1.size == 0:
        return {
            "spearman": np.nan,
            "jaccard_topk": np.nan,
            "jaccard_topk_hard": np.nan,
            "rbo": np.nan,
            "weighted_jaccard": np.nan,
            "k_eff": 0,
        }

    # Spearman (rank stability over all regions)
    r1 = _ranks_with_ties(s1)
    r2 = _ranks_with_ties(s2)
    r1c = r1 - r1.mean()
    r2c = r2 - r2.mean()
    denom = (np.linalg.norm(r1c) * np.linalg.norm(r2c))
    spearman = float(np.dot(r1c, r2c) / denom) if denom != 0 else np.nan

    # Choose K adaptively
    k_eff = _effective_k(s1.size, k=k, k_frac=k_frac, k_min=k_min)

    # Jaccard top-K (tie-aware + hard baseline)
    set1_h = _topk_hard(s1, k_eff)
    set2_h = _topk_hard(s2, k_eff)
    jacc_h = _jaccard(set1_h, set2_h)

    set1_t = _topk_tie_aware(s1, k_eff)
    set2_t = _topk_tie_aware(s2, k_eff)
    jacc_t = _jaccard(set1_t, set2_t)

    # Extra stability metrics
    wj = _weighted_jaccard(s1, s2)
    rbo = _rbo_from_scores(s1, s2, p=rbo_p, k=k_eff)

    return {
        "spearman": spearman,
        "jaccard_topk": jacc_t,
        "jaccard_topk_hard": jacc_h,
        "rbo": rbo,
        "weighted_jaccard": wj,
        "k_eff": float(k_eff),
    }

def aggregate_scores_by_region(tokens, windows, r_sec, scores) -> Tuple[List[Tuple[str, str, int]], np.ndarray]:
    # map (start,end) -> window type
    win_map = {}
    for k, lst in (
        ("pre", windows.pre),
        ("qrs", windows.qrs),
        ("qrs_term", windows.qrs_term),
        ("qrs_on", windows.qrs_on),
        ("stt", windows.stt),
        ("tlate", windows.tlate),
        ("pace", windows.pace),
        ("beat", windows.beat),
    ):
        for w in lst:
            win_map[w] = k

    r_sec_arr = np.asarray(r_sec, dtype=float)
    region_to_scores: Dict[Tuple[str, str, int], List[float]] = {}

    for idx, (lead, (s, e)) in enumerate(tokens):
        wt = win_map.get((s, e))
        if wt is None:
            continue
        centre = 0.5 * (float(s) + float(e))
        beat_idx = int(np.argmin(np.abs(r_sec_arr - centre))) if r_sec_arr.size else 0
        key = (str(lead), str(wt), beat_idx)
        region_to_scores.setdefault(key, []).append(float(scores[idx]))

    region_keys = sorted(region_to_scores.keys())
    region_scores = np.array([np.mean(region_to_scores[k]) for k in region_keys], dtype=float)
    if region_scores.size and region_scores.max() > 0:
        region_scores /= region_scores.max()
    return region_keys, region_scores

def stability_with_extra_beat_regionwise(
    mat_path_a: str,
    mat_path_b: str,
    fs: float,
    payload_a: Dict,
    payload_b: Dict,
    class_name_eval: str,
    lead_names: Sequence[str] = LEADS12,
    k: int = 20,
    beat_tolerance_sec: float = 0.08,
) -> Dict[str, float]:
    x_a = load_mat_TF(mat_path_a)
    x_b = load_mat_TF(mat_path_b)

    r_idx_a = detect_rpeaks(x_a, fs, prefer=("II", "V2", "V3"), lead_names=lead_names)
    r_idx_b = detect_rpeaks(x_b, fs, prefer=("II", "V2", "V3"), lead_names=lead_names)
    r_sec_a = (r_idx_a / float(fs)).tolist()
    r_sec_b = (r_idx_b / float(fs)).tolist()

    cfg = REGISTRY[class_name_eval]

    windows_a = build_windows_from_rpeaks(r_sec_a, class_name=class_name_eval)
    tokens_a = build_tokens(lead_names, windows_a, which=cfg.window_keys)

    windows_b = build_windows_from_rpeaks(r_sec_b, class_name=class_name_eval)
    tokens_b = build_tokens(lead_names, windows_b, which=cfg.window_keys)

    def _spans(payload: Dict) -> Dict[str, List[Tuple[float, float, float]]]:
        raw = payload.get("perlead_spans", {})
        return {str(L): [(float(s), float(e), float(w)) for (s, e, w) in spans] for L, spans in raw.items()}

    spans_a = _spans(payload_a)
    spans_b = _spans(payload_b)

    scores_a = integrate_attribution(spans_a, tokens_a)
    scores_a = apply_sinus_prior_blend(class_name_eval, tokens_a, windows_a, scores_a, alpha=0.8)
    scores_a = apply_af_prior_blend(class_name_eval, tokens_a, windows_a, scores_a, alpha=0.8)
    scores_a = apply_vpb_prior_blend(class_name_eval, tokens_a, windows_a, scores_a, alpha=0.8)
    keys_a, reg_a = aggregate_scores_by_region(tokens_a, windows_a, r_sec_a, scores_a)

    scores_b = integrate_attribution(spans_b, tokens_b)
    scores_b = apply_sinus_prior_blend(class_name_eval, tokens_b, windows_b, scores_b, alpha=0.8)
    scores_b = apply_af_prior_blend(class_name_eval, tokens_b, windows_b, scores_b, alpha=0.8)
    scores_b = apply_vpb_prior_blend(class_name_eval, tokens_b, windows_b, scores_b, alpha=0.8)
    keys_b, reg_b = aggregate_scores_by_region(tokens_b, windows_b, r_sec_b, scores_b)

    beat_pairs = _match_beats_by_time(r_sec_a, r_sec_b, max_diff_sec=beat_tolerance_sec)
    if not beat_pairs:
        return {"spearman": np.nan, "jaccard_topk": np.nan}

    reg_dict_a = {k: v for k, v in zip(keys_a, reg_a)}
    reg_dict_b = {k: v for k, v in zip(keys_b, reg_b)}

    common_leads = sorted({k[0] for k in keys_a} & {k[0] for k in keys_b})
    common_wins = sorted({k[1] for k in keys_a} & {k[1] for k in keys_b})

    aligned_a: List[float] = []
    aligned_b: List[float] = []
    for ia, ib in beat_pairs:
        for lead in common_leads:
            for wt in common_wins:
                ka = (lead, wt, ia)
                kb = (lead, wt, ib)
                if ka in reg_dict_a and kb in reg_dict_b:
                    aligned_a.append(float(reg_dict_a[ka]))
                    aligned_b.append(float(reg_dict_b[kb]))

    if not aligned_a:
        return {"spearman": np.nan, "jaccard_topk": np.nan}

    return explanation_stability(np.asarray(aligned_a), np.asarray(aligned_b), k=k)

def make_augmented_sel_df_for_one_record(
    mat_path: str,
    class_code: str,
    fs: float,
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    edge_beats: int = 1,
    maxlen: int = MAXLEN,
    out_root: Path = AUGMENT_ROOT,
    overwrite: bool = False,
) -> pd.DataFrame:
    mat_path = Path(mat_path)
    rec_dir = out_root / mat_path.stem
    rec_dir.mkdir(parents=True, exist_ok=True)

    seed_used, rng_used = _init_rng(seed=seed, rng=rng)

    x_orig = load_mat_TF(str(mat_path))
    beat_index = _choose_non_edge_beat_index(
        x_orig,
        float(fs),
        rng_used,
        edge_beats=int(edge_beats),
    )
    x_extra = add_extra_beat(
        x_orig,
        float(fs),
        location="middle",
        beat_index=int(beat_index),
        max_len=maxlen,
    )
    suffix = f"_extra_seed{int(seed_used)}_beat{int(beat_index)}"

    def _save_variant(x_tf: np.ndarray, suffix: str, sel_idx: int) -> Dict:
        new_mat = rec_dir / f"{mat_path.stem}{suffix}{mat_path.suffix}"
        if overwrite or (not new_mat.exists()):
            savemat(new_mat, {"val": x_tf.T})
            src_hea = mat_path.with_suffix(".hea")
            dst_hea = new_mat.with_suffix(".hea")
            if src_hea.exists():
                copyfile(src_hea, dst_hea)
        return {
            "group_class": class_code,
            "filename": str(new_mat),
            "sel_idx": int(sel_idx),
            "augment_seed": int(seed_used),
            "augment_beat_index": int(beat_index),
        }

    rows = [
        {
            "group_class": class_code,
            "filename": str(mat_path),
            "sel_idx": 0,
            "augment_seed": np.nan,
            "augment_beat_index": np.nan,
        },
        _save_variant(x_extra, suffix, 1),
    ]
    return pd.DataFrame(rows)

def run_extra_beat_stability_experiment(
    mat_path: str,
    snomed_code: str,
    model,
    class_names: Sequence[str],
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    edge_beats: int = 1,
    overwrite: bool = False,
    maxlen: int = MAXLEN,
    beat_tolerance_sec: float = 0.08,
    k: int = 20,
    augment_root: Path = AUGMENT_ROOT,
) -> Tuple[Dict, pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame]:
    hea_path, _ = ensure_paths(mat_path)
    fs, _ = parse_fs_and_leads(hea_path, default_fs=500.0)

    sel_df = make_augmented_sel_df_for_one_record(
        mat_path=mat_path,
        class_code=str(snomed_code),
        fs=float(fs),
        seed=seed,
        rng=rng,
        edge_beats=edge_beats,
        maxlen=maxlen,
        out_root=augment_root,
        overwrite=bool(overwrite),
    )

    all_fused_payloads, df_lime_all, df_ts_all = run_fused_pipeline_for_classes(
        target_classes=[str(snomed_code)],
        sel_df=sel_df,
        model=model,
        class_names=class_names,
        max_examples_per_class=None,
        plot=False,
    )

    fused_for_cls = all_fused_payloads[str(snomed_code)]
    payload_orig = fused_for_cls[0]
    payload_extra = fused_for_cls[1]

    mat_path_p = Path(mat_path)
    mat_extra = Path(sel_df.loc[sel_df["sel_idx"] == 1, "filename"].iloc[0])

    class_name_eval = str(TARGET_META[str(snomed_code)]["name"])

    metrics_extra = stability_with_extra_beat_regionwise(
        mat_path_a=str(mat_path_p),
        mat_path_b=str(mat_extra),
        fs=float(fs),
        payload_a=payload_orig,
        payload_b=payload_extra,
        class_name_eval=class_name_eval,
        k=k,
        beat_tolerance_sec=beat_tolerance_sec,
    )

    # Surface the augmentation choice in the return value, too.
    aug_row = sel_df.loc[sel_df["sel_idx"] == 1].iloc[0]
    metrics = {
        "extra": metrics_extra,
        "augment_seed": int(aug_row["augment_seed"]),
        "augment_beat_index": int(aug_row["augment_beat_index"]),
        "augmented_file": str(mat_extra),
    }
    return metrics, sel_df, all_fused_payloads, df_lime_all, df_ts_all
