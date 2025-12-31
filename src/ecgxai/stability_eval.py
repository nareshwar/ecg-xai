"""
stability_eval.py — Extra-beat augmentation + regionwise stability.

What this module does
---------------------
1) Creates ONE augmented ECG by duplicating a beat (copy a beat segment around an R-peak)
    and inserting it immediately after that beat (middle insertion). Edge beats are avoided.
2) Runs the fused explainer (LIME + TimeSHAP + fusion) for original and augmented ECG.
3) Computes regionwise stability (Spearman + Jaccard@K + extras) while "ignoring"
    the extra beat by matching beats between recordings based on R-peak timestamps.

Key idea
--------
The augmented signal has one additional beat. We align beats between (A) original and (B)
augmented using R-peak times within a tolerance, and then compare region scores on matched beats.

Expected ECG format
-------------------
- load_mat_TF() returns ECG shaped (T, 12) in float.
- .mat PhysioNet format: savemat(..., {"val": x.T}) where x is (T, 12).

Outputs
-------
- run_extra_beat_stability_experiment() returns:
    metrics: dict with stability metrics + augmentation metadata
    sel_df: 2-row DataFrame (original + augmented)
    all_fused_payloads: fused payload dict (per class -> per sel_idx)
    df_lime_all: LIME dataframe (2 rows)
    df_ts_all: TimeSHAP dataframe (2 rows)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import savemat

from .config import MAXLEN
from .preprocessing import ensure_paths, parse_fs_and_leads

from .config_targets import TARGET_META
from .explainer import run_fused_pipeline_for_classes

# Import what we need from eval.py (core)
from .eval import (
    LEADS12,
    detect_rpeaks,
    REGISTRY,
    build_windows_from_rpeaks,
    build_tokens,
    integrate_attribution,
    apply_all_priors,
)

from preprocessing import load_mat_TF

ROOT = Path.cwd().parent
AUGMENT_ROOT = Path(ROOT / "outputs" / "extra_beat_aug")  # change if you want


__all__ = [
    "AUGMENT_ROOT",
    "add_extra_beat",
    "explanation_stability",
    "stability_with_extra_beat_regionwise",
    "make_augmented_sel_df_for_one_record",
    "run_extra_beat_stability_experiment",
]


@dataclass(frozen=True)
class ExtraBeatParams:
    """Parameters controlling how the extra beat is extracted/inserted."""
    location: str = "middle"  # "middle" or "end"
    pre_sec: float = 0.30
    post_sec: float = 0.40
    edge_beats: int = 1
    prefer: Tuple[str, ...] = ("II", "V2", "V3")


# ---------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------
def _init_rng(
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, np.random.Generator]:
    """Return (seed, rng). If rng is provided, we also produce a reproducible seed for filenames."""
    if rng is not None:
        if seed is None:
            seed = int(rng.integers(0, 2**31 - 1))
        return int(seed), rng

    if seed is None:
        # Generate a concrete integer seed (stable to record).
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    return int(seed), np.random.default_rng(int(seed))


# ---------------------------------------------------------------------
# Header copy with updated sample count (important correctness fix)
# ---------------------------------------------------------------------
def _copy_header_with_updated_nsamp(
    src_hea: Path,
    dst_hea: Path,
    *,
    new_record_name: Optional[str],
    new_nsamp: int,
) -> None:
    """Copy a PhysioNet header, updating record name (optional) and n_samples if parseable."""
    if not src_hea.exists():
        return

    try:
        lines = src_hea.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    except Exception:
        lines = src_hea.read_text(encoding="latin-1", errors="ignore").splitlines(True)

    if not lines:
        copyfile(src_hea, dst_hea)
        return

    first = lines[0].strip("\n")
    parts = first.split()

    # Typical: <rec> <n_sig> <fs> <n_samples> ...
    if len(parts) >= 4 and parts[1].isdigit():
        if new_record_name:
            parts[0] = str(new_record_name)
        # nsamp is commonly integer
        try:
            int(parts[3])
            parts[3] = str(int(new_nsamp))
        except Exception:
            pass

        lines[0] = " ".join(parts) + "\n"

    dst_hea.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Beat selection + augmentation
# ---------------------------------------------------------------------
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
    """Pick a beat index (index into R-peak list) while avoiding edge beats and clipping.

    Strategy:
      1) Detect R-peaks.
      2) Exclude first/last `edge_beats`.
      3) Keep only beats whose extraction window [r-pre, r+post] stays in-bounds.
      4) Fall back to the middle beat if candidates are empty.
    """
    x = np.asarray(x, dtype=np.float32)
    T = int(x.shape[0])

    r_idx = detect_rpeaks(x, fs, prefer=prefer, lead_names=lead_names)
    n = int(r_idx.size)
    if n == 0:
        raise ValueError("No R-peaks detected; cannot choose beat_index.")

    lo = max(0, int(edge_beats))
    hi = min(n, n - int(edge_beats))
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
    """Duplicate one beat and insert it either in the middle or at the end.

    Args:
        x: ECG array (T, F) float.
        fs: Sampling frequency.
        location: "middle" inserts after the selected beat; "end" appends.
        beat_index: index into the detected R-peak list. If None, chooses a sensible default.
        pre_sec/post_sec: beat extraction window around R-peak.
        max_len: if not None, truncate final signal to max_len samples.

    Returns:
        Augmented ECG array (T_new, F).
    """
    if location not in ("middle", "end"):
        raise ValueError("location must be 'middle' or 'end'")

    x = np.asarray(x, dtype=np.float32)
    T, F = x.shape

    r_idx = detect_rpeaks(x, fs, prefer=prefer, lead_names=lead_names)
    if r_idx.size == 0:
        raise ValueError("No R-peaks detected; cannot add extra beat.")

    if beat_index is None:
        beat_index = int(len(r_idx) // 2) if location == "middle" else (len(r_idx) - 1)

    if not (0 <= int(beat_index) < int(len(r_idx))):
        raise IndexError(f"beat_index {beat_index} out of range for {len(r_idx)} beats.")

    r = int(r_idx[int(beat_index)])
    s = max(0, int(round(r - pre_sec * fs)))
    e = min(T, int(round(r + post_sec * fs)))
    if e <= s:
        raise RuntimeError("Computed empty beat segment when extracting heartbeat.")

    beat_seg = x[s:e, :]

    insert_at = e if location == "middle" else T
    x_new = np.concatenate([x[:insert_at, :], beat_seg, x[insert_at:, :]], axis=0)

    if max_len is not None and x_new.shape[0] > int(max_len):
        x_new = x_new[: int(max_len), :]

    return x_new


# ---------------------------------------------------------------------
# Beat matching (to ignore the extra beat)
# ---------------------------------------------------------------------
def _match_beats_by_time(
    r_sec_a: Sequence[float],
    r_sec_b: Sequence[float],
    max_diff_sec: float = 0.08,
) -> List[Tuple[int, int]]:
    """Greedy matching of beats by timestamp; returns list of (idx_a, idx_b)."""
    r_a = np.asarray(r_sec_a, dtype=float)
    r_b = np.asarray(r_sec_b, dtype=float)

    pairs: List[Tuple[int, int]] = []
    used_b = set()

    for i, t in enumerate(r_a):
        candidates = [(j, abs(float(r_b[j]) - float(t))) for j in range(len(r_b)) if j not in used_b]
        if not candidates:
            break
        j, dt = min(candidates, key=lambda p: p[1])
        if dt <= float(max_diff_sec):
            pairs.append((int(i), int(j)))
            used_b.add(int(j))

    return pairs


# ---------------------------------------------------------------------
# Stability metrics on aligned score vectors
# ---------------------------------------------------------------------
def _ranks_with_ties(values: np.ndarray) -> np.ndarray:
    """Compute 1..n ranks with average ranks for ties."""
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
    """Pick K scaling with number of regions: top k_frac, at least k_min, at most k."""
    if n <= 0:
        return 0
    return max(int(k_min), min(int(k), int(round(float(k_frac) * n))))


def _topk_hard(scores: np.ndarray, k: int) -> set[int]:
    scores = np.asarray(scores, float)
    n = scores.size
    if n == 0 or k <= 0:
        return set()
    k = min(k, n)
    idx = np.argsort(scores)[::-1][:k]
    return set(idx.tolist())


def _topk_tie_aware(scores: np.ndarray, k: int) -> set[int]:
    """Include all items with score >= score_at_rank_k (ties don't jitter)."""
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
    """Overlap of attribution mass (0..1). Often smoother than top-K membership."""
    a = np.maximum(np.asarray(scores_a, float), 0.0)
    b = np.maximum(np.asarray(scores_b, float), 0.0)
    if a.size == 0 or b.size != a.size:
        return float("nan")
    num = np.minimum(a, b).sum()
    den = np.maximum(a, b).sum() + float(eps)
    return float(num / den)


def _rbo_from_scores(scores_a: np.ndarray, scores_b: np.ndarray, p: float = 0.90, k: Optional[int] = None) -> float:
    """Rank-Biased Overlap (0..1). p near 1 looks deeper into the ranking."""
    a = np.asarray(scores_a, float)
    b = np.asarray(scores_b, float)
    n = a.size
    if n == 0 or b.size != n:
        return float("nan")
    if k is None:
        k = n
    k = min(int(k), n)

    ra = np.argsort(a)[::-1][:k]
    rb = np.argsort(b)[::-1][:k]

    seen_a, seen_b = set(), set()
    rbo_sum = 0.0
    for d in range(1, k + 1):
        seen_a.add(int(ra[d - 1]))
        seen_b.add(int(rb[d - 1]))
        overlap = len(seen_a & seen_b)
        rbo_sum += (overlap / d) * (float(p) ** (d - 1))

    return float((1 - float(p)) * rbo_sum)


def explanation_stability(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    k: int = 20,
    k_frac: float = 0.10,
    k_min: int = 10,
    rbo_p: float = 0.90,
) -> Dict[str, float]:
    """Compute several stability measures between two aligned score vectors.

    Returns keys:
        - spearman
        - jaccard_topk            (tie-aware)
        - jaccard_topk_hard       (hard top-K membership)
        - rbo                     (rank-sensitive overlap)
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
            "k_eff": 0.0,
        }

    # Spearman over all regions (manual to avoid scipy dependency drift)
    r1 = _ranks_with_ties(s1)
    r2 = _ranks_with_ties(s2)
    r1c = r1 - r1.mean()
    r2c = r2 - r2.mean()
    denom = float(np.linalg.norm(r1c) * np.linalg.norm(r2c))
    spearman = float(np.dot(r1c, r2c) / denom) if denom != 0 else np.nan

    k_eff = _effective_k(s1.size, k=k, k_frac=k_frac, k_min=k_min)

    set1_h = _topk_hard(s1, k_eff)
    set2_h = _topk_hard(s2, k_eff)
    jacc_h = _jaccard(set1_h, set2_h)

    set1_t = _topk_tie_aware(s1, k_eff)
    set2_t = _topk_tie_aware(s2, k_eff)
    jacc_t = _jaccard(set1_t, set2_t)

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


# ---------------------------------------------------------------------
# Region aggregation (lead × window_type × beat_index)
# ---------------------------------------------------------------------
def aggregate_scores_by_region(tokens, windows, r_sec, scores) -> Tuple[List[Tuple[str, str, int]], np.ndarray]:
    """Aggregate token-level scores into region scores.

    Region key: (lead_name, window_type, beat_index)

    We map each token (lead, (start,end)) to:
      - window_type via exact lookup in the per-type window lists
      - beat_index via nearest R-peak time to token centre

    Returns:
      region_keys: sorted list of region keys
      region_scores: array aligned with region_keys, normalized to max=1 if possible
    """
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


def _region_scores_from_payload(mat_path: str, fs: float, payload: Dict, class_name_eval: str) -> Tuple[List[Tuple[str, str, int]], np.ndarray, List[float]]:
    """Compute normalized region scores for one record/payload."""
    x = load_mat_TF(mat_path)

    r_idx = detect_rpeaks(x, fs, prefer=("II", "V2", "V3"), lead_names=LEADS12)
    r_sec = (r_idx / float(fs)).tolist()

    cfg = REGISTRY[class_name_eval]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name_eval)
    tokens = build_tokens(LEADS12, windows, which=cfg.window_keys)

    raw = payload.get("perlead_spans", {}) or {}
    spans = {str(L): [(float(s), float(e), float(w)) for (s, e, w) in lst] for L, lst in raw.items()}

    scores = integrate_attribution(spans, tokens)

    # Apply priors (kept identical to your current logic)
    scores = apply_all_priors(class_name_eval, tokens, windows, scores, alpha=0.8)

    keys, reg_scores = aggregate_scores_by_region(tokens, windows, r_sec, scores)
    return keys, reg_scores, r_sec


def stability_with_extra_beat_regionwise(
    mat_path_a: str,
    mat_path_b: str,
    fs: float,
    payload_a: Dict,
    payload_b: Dict,
    class_name_eval: str,
    *,
    k: int = 20,
    beat_tolerance_sec: float = 0.08,
) -> Dict[str, float]:
    """Compute stability between (A) original and (B) augmented while ignoring the extra beat."""
    keys_a, reg_a, r_sec_a = _region_scores_from_payload(mat_path_a, fs, payload_a, class_name_eval)
    keys_b, reg_b, r_sec_b = _region_scores_from_payload(mat_path_b, fs, payload_b, class_name_eval)

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
                ka = (lead, wt, int(ia))
                kb = (lead, wt, int(ib))
                if ka in reg_dict_a and kb in reg_dict_b:
                    aligned_a.append(float(reg_dict_a[ka]))
                    aligned_b.append(float(reg_dict_b[kb]))

    if not aligned_a:
        return {"spearman": np.nan, "jaccard_topk": np.nan}

    return explanation_stability(np.asarray(aligned_a), np.asarray(aligned_b), k=k)


# ---------------------------------------------------------------------
# Build a 2-row sel_df (original + augmented)
# ---------------------------------------------------------------------
def make_augmented_sel_df_for_one_record(
    mat_path: str,
    class_code: str,
    fs: float,
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    params: ExtraBeatParams = ExtraBeatParams(),
    maxlen: int = MAXLEN,
    out_root: Path = AUGMENT_ROOT,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create selection df with original + augmented record stored under out_root/<stem>/."""
    mat_path = Path(mat_path)
    rec_dir = Path(out_root) / mat_path.stem
    rec_dir.mkdir(parents=True, exist_ok=True)

    seed_used, rng_used = _init_rng(seed=seed, rng=rng)

    x_orig = load_mat_TF(str(mat_path))

    beat_index = _choose_non_edge_beat_index(
        x_orig,
        float(fs),
        rng_used,
        pre_sec=float(params.pre_sec),
        post_sec=float(params.post_sec),
        edge_beats=int(params.edge_beats),
        prefer=params.prefer,
    )

    x_extra = add_extra_beat(
        x_orig,
        float(fs),
        location=str(params.location),
        beat_index=int(beat_index),
        pre_sec=float(params.pre_sec),
        post_sec=float(params.post_sec),
        max_len=int(maxlen),
    )

    suffix = f"_extra_seed{int(seed_used)}_beat{int(beat_index)}"

    def _save_variant(x_tf: np.ndarray, suffix_: str, sel_idx: int) -> Dict:
        new_mat = rec_dir / f"{mat_path.stem}{suffix_}{mat_path.suffix}"
        if overwrite or (not new_mat.exists()):
            savemat(new_mat, {"val": np.asarray(x_tf, dtype=np.float32).T})

            src_hea = mat_path.with_suffix(".hea")
            dst_hea = new_mat.with_suffix(".hea")

            if src_hea.exists():
                _copy_header_with_updated_nsamp(
                    src_hea,
                    dst_hea,
                    new_record_name=new_mat.stem,
                    new_nsamp=int(x_tf.shape[0]),
                )

        return {
            "group_class": str(class_code),
            "filename": str(new_mat),
            "sel_idx": int(sel_idx),
            "augment_seed": int(seed_used),
            "augment_beat_index": int(beat_index),
        }

    rows = [
        {
            "group_class": str(class_code),
            "filename": str(mat_path),
            "sel_idx": 0,
            "augment_seed": np.nan,
            "augment_beat_index": np.nan,
        },
        _save_variant(x_extra, suffix, 1),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Public experiment runner
# ---------------------------------------------------------------------
def run_extra_beat_stability_experiment(
    mat_path: str,
    snomed_code: str,
    model,
    class_names: Sequence[str],
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    params: ExtraBeatParams = ExtraBeatParams(),
    overwrite: bool = False,
    maxlen: int = MAXLEN,
    beat_tolerance_sec: float = 0.08,
    k: int = 20,
    augment_root: Path = AUGMENT_ROOT,
) -> Tuple[Dict, pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame]:
    """Run the full original-vs-augmented stability experiment for one record + one SNOMED code."""
    hea_path, _ = ensure_paths(mat_path)
    fs, _ = parse_fs_and_leads(hea_path, default_fs=500.0)

    sel_df = make_augmented_sel_df_for_one_record(
        mat_path=mat_path,
        class_code=str(snomed_code),
        fs=float(fs),
        seed=seed,
        rng=rng,
        params=params,
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

    # REGISTRY uses human-readable names; TARGET_META maps SNOMED -> {"name": ...}
    class_name_eval = str(TARGET_META[str(snomed_code)]["name"])

    metrics_extra = stability_with_extra_beat_regionwise(
        mat_path_a=str(mat_path_p),
        mat_path_b=str(mat_extra),
        fs=float(fs),
        payload_a=payload_orig,
        payload_b=payload_extra,
        class_name_eval=class_name_eval,
        k=int(k),
        beat_tolerance_sec=float(beat_tolerance_sec),
    )

    aug_row = sel_df.loc[sel_df["sel_idx"] == 1].iloc[0]
    metrics = {
        "extra": metrics_extra,
        "augment_seed": int(aug_row["augment_seed"]),
        "augment_beat_index": int(aug_row["augment_beat_index"]),
        "augmented_file": str(mat_extra),
        "fs": float(fs),
        "params": {
            "location": params.location,
            "pre_sec": params.pre_sec,
            "post_sec": params.post_sec,
            "edge_beats": params.edge_beats,
            "prefer": list(params.prefer),
        },
    }

    return metrics, sel_df, all_fused_payloads, df_lime_all, df_ts_all
