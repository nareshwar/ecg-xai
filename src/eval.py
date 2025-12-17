"""
ECG explanation evaluator (class-aware windows, HR-adaptive).

Provides:
- BeatWindows: heart-rate aware diagnostic windows around each R-peak
- REGISTRY: mapping from class name -> which leads/windows are "ground truth"
- Token-level metrics:
    * AttAUC (strict / lenient)  [ranking-based accuracy]
    * Precision@K (strict / lenient) [top-K accuracy]
- Faithfulness metrics:
    * Deletion curve per ECG
    * Deletion AUC (lower = more faithful)

- evaluate_explanation(): main entry point for a single ECG + payload
- evaluate_all_payloads(): batched evaluation over fused payloads
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path
from shutil import copyfile

from config import MAXLEN
from preprocessing import infer_fs_from_header, ensure_paths, parse_fs_and_leads
from config_targets import TARGET_META
from explainer import run_fused_pipeline_for_classes

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LEADS12: Tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
)

# --------------------------------------------------------------------------- #
# Basic I/O
# --------------------------------------------------------------------------- #

def load_mat_TF(mat_path: str) -> np.ndarray:
    """
    Load an ECG MAT file and return a float32 array of shape (T, F).

    This is deliberately generic:
    - it looks for common keys ('val', 'ECG', 'ecg', 'data', 'signal')
    - it fixes orientation so that time is always axis 0.
    """
    d = loadmat(mat_path, squeeze_me=True)

    for key in ("val", "ECG", "ecg", "data", "signal"):
        if key in d:
            arr = np.asarray(d[key])
            break
    else:
        raise ValueError(f"No ECG array found in {mat_path}")

    # Ensure (n_leads, n_samples) then transpose if needed
    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.shape[0] in (8, 12):
        x = arr.T
    else:
        x = arr

    # Now force time on axis 0
    if x.shape[0] < x.shape[1]:
        x = x.T

    return x.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# R-peak detection
# --------------------------------------------------------------------------- #

def _bandpass(
    x: np.ndarray,
    fs: float,
    lo: float = 5.0,
    hi: float = 18.0,
    order: int = 3,
) -> np.ndarray:
    """Simple band-pass filter for QRS detection."""
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def _moving_integral(sig: np.ndarray, fs: float, w_sec: float = 0.150) -> np.ndarray:
    """Moving-window integration, applied channel-wise."""
    w = max(1, int(round(w_sec * fs)))
    ker = np.ones(w, dtype=np.float32) / w
    out = np.empty_like(sig)
    for j in range(sig.shape[1]):
        out[:, j] = np.convolve(sig[:, j], ker, mode="same")
    return out


def detect_rpeaks(
    x: np.ndarray,
    fs: float,
    prefer: Sequence[str] = ("II", "V2", "V3"),
    lead_names: Optional[Sequence[str]] = None,
    refractory_ms: int = 250,
    min_prom: Optional[float] = None,
) -> np.ndarray:
    """
    Detect R-peaks using a simple Pan–Tompkins–style pipeline.

    Parameters
    ----------
    x : (T, F) ECG array
    fs : sampling frequency in Hz
    prefer : which leads to prioritize for detection
    lead_names : names corresponding to columns of x
    refractory_ms : minimum distance between peaks (ms)
    min_prom : minimum prominence passed to scipy.find_peaks

    Returns
    -------
    rpeaks : np.ndarray of indices (samples)
    """
    T, _ = x.shape

    # choose channels
    if lead_names is not None:
        chosen = [lead_names.index(L) for L in prefer if L in lead_names]
        if not chosen:
            chosen = [int(np.argmax(np.std(x, axis=0)))]
    else:
        chosen = [0]

    # bandpass -> differentiate -> square -> moving integration
    xb = _bandpass(x[:, chosen], fs)
    diff = np.diff(xb, axis=0, prepend=xb[:1])
    sq = diff**2
    feat = _moving_integral(sq, fs, w_sec=0.15).sum(axis=1)

    # threshold & peak picking
    med = np.median(feat)
    thr = med + 0.5 * np.median(np.abs(feat - med))
    distance = int(round(refractory_ms / 1000 * fs))
    if min_prom is None:
        min_prom = 0.5 * np.std(feat)

    peaks, _ = find_peaks(
        feat,
        height=thr,
        distance=distance,
        prominence=min_prom,
    )

    # refine to local maxima of bandpassed signal
    ref: List[int] = []
    win = int(0.08 * fs)
    xb_sum = xb.sum(axis=1)
    for p in peaks:
        s = max(0, p - win)
        e = min(T, p + win + 1)
        ref.append(s + int(np.argmax(xb_sum[s:e])))

    return np.array(sorted(set(ref)), dtype=int)

import numpy as np
import pandas as pd
from pathlib import Path
from shutil import copyfile
from scipy.io import savemat

from typing import Sequence, Dict, List, Tuple, Optional

from preprocessing import ensure_paths, parse_fs_and_leads
from explainer import run_fused_pipeline_for_classes
from config import MAXLEN
from config_targets import TARGET_META


# -------------------------------------------------------------------
# 1) Add an extra heartbeat (duplicate beat) to an ECG
# -------------------------------------------------------------------
def add_extra_beat(
    x: np.ndarray,
    fs: float,
    *,
    location: str = "end",
    beat_index: Optional[int] = None,
    pre_sec: float = 0.30,
    post_sec: float = 0.40,
    lead_names: Sequence[str] = LEADS12,
    precision_k: int = 20,
    prefer: Sequence[str] = ("II", "V2", "V3"),
    max_len: Optional[int] = None,
) -> np.ndarray:
    """
    Duplicate one heartbeat and insert it into the ECG.

    location = 'end'    -> append extra beat at end
    location = 'middle' -> insert extra beat after that beat
    """
    x = np.asarray(x, dtype=np.float32)
    T, F = x.shape

    # --- Detect R-peaks ---
    r_idx = detect_rpeaks(x, fs, prefer=prefer, lead_names=lead_names)
    if r_idx.size == 0:
        raise ValueError("No R-peaks detected; cannot add extra beat.")

    # --- Choose which beat to duplicate ---
    if beat_index is None:
        if location == "middle":
            beat_index = int(len(r_idx) // 2)
        else:  # default: 'end'
            beat_index = len(r_idx) - 1

    if not (0 <= beat_index < len(r_idx)):
        raise IndexError(f"beat_index {beat_index} out of range for {len(r_idx)} beats.")

    r = int(r_idx[beat_index])

    # --- Beat window in samples (≈ BeatWindows.beat: [-0.30, +0.40] s) ---
    s = max(0, int(round(r - pre_sec * fs)))
    e = min(T, int(round(r + post_sec * fs)))
    if e <= s:
        raise RuntimeError("Computed empty beat segment when extracting heartbeat.")

    beat_seg = x[s:e, :]  # (L, F)

    # --- Decide where to insert ---
    if location == "middle":
        insert_at = e
    else:  # 'end' (append)
        insert_at = T

    x_new = np.concatenate([x[:insert_at, :], beat_seg, x[insert_at:, :]], axis=0)

    if max_len is not None and x_new.shape[0] > max_len:
        x_new = x_new[:max_len, :]

    return x_new


# -------------------------------------------------------------------
# 2) Match beats A↔B by R-peak time
# -------------------------------------------------------------------
def _match_beats_by_time(
    r_sec_a: Sequence[float],
    r_sec_b: Sequence[float],
    max_diff_sec: float = 0.08,
) -> List[Tuple[int, int]]:
    """
    Greedy 1-to-1 R-peak matching between two recordings.
    Returns list of (idx_a, idx_b) pairs within max_diff_sec.
    """
    r_a = np.asarray(r_sec_a, dtype=float)
    r_b = np.asarray(r_sec_b, dtype=float)

    pairs: List[Tuple[int, int]] = []
    used_b = set()

    for i, t in enumerate(r_a):
        diffs = [
            (j, abs(float(r_b[j]) - float(t)))
            for j in range(len(r_b))
            if j not in used_b
        ]
        if not diffs:
            break

        j, dt = min(diffs, key=lambda p: p[1])
        if dt <= max_diff_sec:
            pairs.append((i, j))
            used_b.add(j)

    return pairs


# -------------------------------------------------------------------
# 3) Stability of fused explanations under extra beat (regionwise)
# -------------------------------------------------------------------
def stability_with_extra_beat_regionwise(
    mat_path_a: str,
    mat_path_b: str,
    fs: float,
    payload_a: Dict,
    payload_b: Dict,
    class_name_eval: str,  # e.g. 'sinus rhythm' (matches REGISTRY keys)
    lead_names: Sequence[str] = LEADS12,
    k: int = 20,
    beat_tolerance_sec: float = 0.08,
) -> Dict[str, float]:
    """
    Compare region-level fused explanations between:
        ECG A  (original)
        ECG B  (with one extra beat)

    Only uses *matched beats* (by R-peak time), so extra beat is ignored.
    Returns Spearman and Jaccard@K for aligned regions.
    """
    # --- Load ECGs & detect R-peaks ---
    x_a = load_mat_TF(mat_path_a)
    x_b = load_mat_TF(mat_path_b)

    r_idx_a = detect_rpeaks(x_a, fs, prefer=("II", "V2", "V3"), lead_names=lead_names)
    r_idx_b = detect_rpeaks(x_b, fs, prefer=("II", "V2", "V3"), lead_names=lead_names)

    r_sec_a = (r_idx_a / float(fs)).tolist()
    r_sec_b = (r_idx_b / float(fs)).tolist()

    # --- Class-specific windows & tokens ---
    cfg = REGISTRY[class_name_eval]

    windows_a = build_windows_from_rpeaks(r_sec_a, class_name=class_name_eval)
    tokens_a = build_tokens(lead_names, windows_a, which=cfg.window_keys)

    windows_b = build_windows_from_rpeaks(r_sec_b, class_name=class_name_eval)
    tokens_b = build_tokens(lead_names, windows_b, which=cfg.window_keys)

    def _spans_from_payload(payload: Dict) -> Dict[str, List[Tuple[float, float, float]]]:
        raw_spans = payload.get("perlead_spans", {})
        return {
            str(L): [(float(s), float(e), float(w)) for (s, e, w) in spans]
            for L, spans in raw_spans.items()
        }

    spans_a = _spans_from_payload(payload_a)
    spans_b = _spans_from_payload(payload_b)

    # --- Token scores + priors for A ---
    scores_a = integrate_attribution(spans_a, tokens_a)
    scores_a = apply_sinus_prior_blend(class_name_eval, tokens_a, windows_a, scores_a, alpha=0.8)
    scores_a = apply_af_prior_blend(class_name_eval, tokens_a, windows_a, scores_a, alpha=0.8)
    scores_a = apply_vpb_prior_blend(class_name_eval, tokens_a, windows_a, scores_a, alpha=0.8)
    keys_a, reg_a = aggregate_scores_by_region(tokens_a, windows_a, r_sec_a, scores_a)

    # --- Token scores + priors for B ---
    scores_b = integrate_attribution(spans_b, tokens_b)
    scores_b = apply_sinus_prior_blend(class_name_eval, tokens_b, windows_b, scores_b, alpha=0.8)
    scores_b = apply_af_prior_blend(class_name_eval, tokens_b, windows_b, scores_b, alpha=0.8)
    scores_b = apply_vpb_prior_blend(class_name_eval, tokens_b, windows_b, scores_b, alpha=0.8)
    keys_b, reg_b = aggregate_scores_by_region(tokens_b, windows_b, r_sec_b, scores_b)

    # --- Match beats by R-peak time (ignore extra beats) ---
    beat_pairs = _match_beats_by_time(r_sec_a, r_sec_b, max_diff_sec=beat_tolerance_sec)
    if not beat_pairs:
        return {"spearman": np.nan, "jaccard_topk": np.nan}

    reg_dict_a = {key: val for key, val in zip(keys_a, reg_a)}
    reg_dict_b = {key: val for key, val in zip(keys_b, reg_b)}

    leads_present_a = {lead for (lead, win_type, idx) in keys_a}
    leads_present_b = {lead for (lead, win_type, idx) in keys_b}
    common_leads = sorted(leads_present_a & leads_present_b)

    win_types_a = {win_type for (lead, win_type, idx) in keys_a}
    win_types_b = {win_type for (lead, win_type, idx) in keys_b}
    common_win_types = sorted(win_types_a & win_types_b)

    aligned_a: List[float] = []
    aligned_b: List[float] = []

    for idx_a, idx_b in beat_pairs:
        for lead in common_leads:
            for win_type in common_win_types:
                key_a = (str(lead), str(win_type), idx_a)
                key_b = (str(lead), str(win_type), idx_b)
                if key_a in reg_dict_a and key_b in reg_dict_b:
                    aligned_a.append(reg_dict_a[key_a])
                    aligned_b.append(reg_dict_b[key_b])

    if not aligned_a:
        return {"spearman": np.nan, "jaccard_topk": np.nan}

    s1 = np.asarray(aligned_a, dtype=float)
    s2 = np.asarray(aligned_b, dtype=float)
    k_eff = min(k, s1.size) if k is not None else None

    return explanation_stability(s1, s2, k=k_eff)


# -------------------------------------------------------------------
# 4) Make sel_df: original + extra_end + extra_mid
# -------------------------------------------------------------------
def make_augmented_sel_df_for_one_record(
    mat_path: str,
    class_code: str,
    fs: float,
    maxlen: int = MAXLEN,
) -> pd.DataFrame:
    """
    Create:
      - original MAT
      - MAT with extra beat at end
      - MAT with extra beat in middle

    Returns sel_df with 3 rows, with sel_idx = 0,1,2.
    """
    x_orig = load_mat_TF(mat_path)
    x_extra_end = add_extra_beat(x_orig, fs, location="end", max_len=maxlen)
    x_extra_mid = add_extra_beat(x_orig, fs, location="middle", max_len=maxlen)

    mat_path = Path(mat_path)

    def _save_variant(x_tf: np.ndarray, suffix: str, sel_idx: int) -> Dict:
        new_mat = mat_path.with_name(mat_path.stem + suffix + mat_path.suffix)

        # PhysioNet-style MAT: val = (n_leads, n_samples)
        savemat(new_mat, {"val": x_tf.T})

        # copy .hea so metadata is preserved
        src_hea = mat_path.with_suffix(".hea")
        dst_hea = new_mat.with_suffix(".hea")
        if src_hea.exists():
            copyfile(src_hea, dst_hea)

        return {
            "group_class": class_code,
            "filename": str(new_mat),
            "sel_idx": int(sel_idx),
        }

    rows = []

    # original
    rows.append({
        "group_class": class_code,
        "filename": str(mat_path),
        "sel_idx": 0,
    })

    # extra beat at end
    rows.append(_save_variant(x_extra_end, "_extra_end", sel_idx=1))

    # extra beat in middle
    rows.append(_save_variant(x_extra_mid, "_extra_mid", sel_idx=2))

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# 5) High-level: run fused pipeline + stability for one record
# -------------------------------------------------------------------
def run_extra_beat_stability_experiment(
    mat_path: str,
    snomed_code: str,
    model,
    class_names: Sequence[str],
    *,
    maxlen: int = MAXLEN,
    beat_tolerance_sec: float = 0.08,
    k: int = 20,
):
    """
    For a single ECG record:
      - build augmented MATs with an extra heartbeat (end + middle)
      - run fused LIME+TimeSHAP for this SNOMED class
      - compute regionwise stability of fused explanations
    """
    # --- sampling frequency from header ---
    hea_path, _ = ensure_paths(mat_path)
    fs, _ = parse_fs_and_leads(hea_path, default_fs=500.0)

    # --- make sel_df with three versions ---
    sel_df = make_augmented_sel_df_for_one_record(
        mat_path=mat_path,
        class_code=snomed_code,
        fs=fs,
        maxlen=maxlen,
    )

    # --- run fused pipeline ---
    all_fused_payloads, df_lime_all, df_ts_all = run_fused_pipeline_for_classes(
        target_classes=[snomed_code],
        sel_df=sel_df,
        model=model,
        class_names=class_names,
        max_examples_per_class=None,
        plot=False,
    )

    fused_for_cls = all_fused_payloads[str(snomed_code)]
    payload_orig = fused_for_cls[0]          # sel_idx = 0
    payload_extra_end = fused_for_cls[1]     # sel_idx = 1
    payload_extra_mid = fused_for_cls[2]     # sel_idx = 2

    # --- paths for augmented MATs ---
    mat_path_p = Path(mat_path)
    mat_extra_end = mat_path_p.with_name(mat_path_p.stem + "_extra_end" + mat_path_p.suffix)
    mat_extra_mid = mat_path_p.with_name(mat_path_p.stem + "_extra_mid" + mat_path_p.suffix)

    # --- map SNOMED -> human name for eval.REGISTRY ---
    class_name_eval = TARGET_META[str(snomed_code)]["name"]

    metrics_end = stability_with_extra_beat_regionwise(
        mat_path_a=str(mat_path_p),
        mat_path_b=str(mat_extra_end),
        fs=fs,
        payload_a=payload_orig,
        payload_b=payload_extra_end,
        class_name_eval=class_name_eval,
        k=k,
        beat_tolerance_sec=beat_tolerance_sec,
    )

    metrics_mid = stability_with_extra_beat_regionwise(
        mat_path_a=str(mat_path_p),
        mat_path_b=str(mat_extra_mid),
        fs=fs,
        payload_a=payload_orig,
        payload_b=payload_extra_mid,
        class_name_eval=class_name_eval,
        k=k,
        beat_tolerance_sec=beat_tolerance_sec,
    )

    metrics = {
        "extra_end": metrics_end,
        "extra_mid": metrics_mid,
    }

    return metrics, sel_df, all_fused_payloads, df_lime_all, df_ts_all

# --------------------------------------------------------------------------- #
# HR-aware beat windows
# --------------------------------------------------------------------------- #

@dataclass
class BeatWindows:
    pre: List[Tuple[float, float]]       # P / pre-QRS
    qrs: List[Tuple[float, float]]       # early/mid QRS
    qrs_term: List[Tuple[float, float]]  # terminal QRS (R' / terminal S)
    qrs_on: List[Tuple[float, float]]    # QRS onset (for Q-wave)
    stt: List[Tuple[float, float]]       # early ST/T
    tlate: List[Tuple[float, float]]     # late T tail
    pace: List[Tuple[float, float]]      # pacer spike region
    beat: List[Tuple[float, float]]      # coarse whole-beat


def _median_hr_bpm(r_sec: Sequence[float]) -> float:
    """Estimate median heart rate from R–R intervals."""
    if len(r_sec) < 2:
        return 70.0
    rr = np.diff(np.asarray(r_sec, float))
    rr = rr[(rr > 0.2) & (rr < 2.5)]
    if len(rr) == 0:
        return 70.0
    return float(60.0 / np.median(rr))

def build_windows_from_rpeaks(
    r_sec: Sequence[float],
    *,
    class_name: Optional[str] = None,
) -> BeatWindows:
    """
    Construct HR-aware, class-conditioned windows around each R-peak.

    For BBB-like classes, widens the terminal QRS window.
    """
    hr = _median_hr_bpm(r_sec)

    # default timings
    pre_lo = 0.22 if hr < 100 else 0.18
    pre_hi = 0.05 if hr < 100 else 0.04
    q_on_lo = 0.06  # R-60 ms
    q_on_hi = 0.01  # up to R-10 ms

    qrs_early_post = 0.08
    qrs_term_post = 0.16

    if class_name and (
        "bundle branch block" in class_name
        or "intraventricular" in class_name
    ):
        # slightly wider terminal window for conduction disease
        qrs_term_post = 0.20

    pre: List[Tuple[float, float]] = []
    qrs: List[Tuple[float, float]] = []
    qrs_term: List[Tuple[float, float]] = []
    qrs_on: List[Tuple[float, float]] = []
    stt: List[Tuple[float, float]] = []
    tlate: List[Tuple[float, float]] = []
    pace: List[Tuple[float, float]] = []
    beat: List[Tuple[float, float]] = []

    for r in r_sec:
        pre.append((r - pre_lo,        r - pre_hi))
        qrs_on.append((r - q_on_lo,    r - q_on_hi))
        qrs.append((r - 0.05,          r + qrs_early_post))
        qrs_term.append((r + 0.04,     r + qrs_term_post))
        stt.append((r + 0.06,          r + 0.20))
        tlate.append((r + 0.20,        r + 0.40))
        pace.append((r - 0.01,         r + 0.02))
        beat.append((r - 0.30,         r + 0.40))

    return BeatWindows(pre, qrs, qrs_term, qrs_on, stt, tlate, pace, beat)


def _build_window_type_map(windows: BeatWindows) -> Dict[Tuple[float, float], str]:
    """
    Utility: map each (start_sec, end_sec) interval to its window type key:
        'pre', 'qrs', 'qrs_term', 'qrs_on', 'stt', 'tlate', 'pace', 'beat'.

    This lets us recover the window family for any token.
    """
    win_map: Dict[Tuple[float, float], str] = {}
    for k, lst in (
        ("pre",      windows.pre),
        ("qrs",      windows.qrs),
        ("qrs_term", windows.qrs_term),
        ("qrs_on",   windows.qrs_on),
        ("stt",      windows.stt),
        ("tlate",    windows.tlate),
        ("pace",     windows.pace),
        ("beat",     windows.beat),
    ):
        for w in lst:
            win_map[w] = k
    return win_map


def aggregate_scores_by_region(
    tokens: List[Tuple[str, Tuple[float, float]]],
    windows: BeatWindows,
    r_sec: Sequence[float],
    scores: np.ndarray,
) -> Tuple[List[Tuple[str, str, int]], np.ndarray]:
    """
    Aggregate token scores into *physiologic regions*.

    Region key:
        (lead, window_type, beat_index)

    - lead        : "I", "II", "V1", ...
    - window_type : "pre", "qrs", "qrs_term", ...
    - beat_index  : index of nearest R-peak to the token centre.

    This reduces sensitivity to tiny shifts of importance between
    neighbouring tokens around the same beat and window.

    Returns
    -------
    region_keys : list of (lead, window_type, beat_index)
    region_scores : np.ndarray of aggregated scores (mean per region)
    """
    scores = np.asarray(scores, dtype=float)
    win_map = _build_window_type_map(windows)
    r_sec_arr = np.asarray(r_sec, dtype=float)

    region_to_scores: Dict[Tuple[str, str, int], List[float]] = {}

    for idx, (lead, (s, e)) in enumerate(tokens):
        # Recover window type (pre/qrs/...)
        win_type = win_map.get((s, e), None)
        if win_type is None:
            continue

        # Centre of this token in time
        centre = 0.5 * (float(s) + float(e))

        # Assign to nearest R-peak -> beat index
        if r_sec_arr.size == 0:
            beat_idx = 0
        else:
            beat_idx = int(np.argmin(np.abs(r_sec_arr - centre)))

        key = (str(lead), str(win_type), beat_idx)
        region_to_scores.setdefault(key, []).append(float(scores[idx]))

    # Aggregate (mean) per region
    region_keys: List[Tuple[str, str, int]] = sorted(region_to_scores.keys())
    region_scores = np.array(
        [np.mean(region_to_scores[k]) for k in region_keys],
        dtype=float,
    )

    # Normalise for stability of scale (optional but nice)
    if region_scores.size > 0 and region_scores.max() > 0:
        region_scores = region_scores / region_scores.max()

    return region_keys, region_scores


# --------------------------------------------------------------------------- #
# Registry: class-level ground truth leads + windows
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClassConfig:
    strict_leads: Tuple[str, ...]
    lenient_leads: Tuple[str, ...]
    window_keys: Tuple[str, ...]


REGISTRY: Dict[str, ClassConfig] = {
    # Atrial / sinus
    "sinus rhythm": ClassConfig(
        ("II", "V1",),
        ("II", "V1", "I", "aVF", "V2"),
        ("pre",),
    ),
    "sinus tachycardia": ClassConfig(
        ("II", "V1"),
        ("II", "V1", "I", "aVF", "V2"),
        ("pre",),
    ),
    "sinus bradycardia": ClassConfig(
        ("II", "V1"),
        ("II", "V1", "I", "aVF", "V2"),
        ("pre",),
    ),
    "sinus arrhythmia": ClassConfig(
        ("II", "V1"),
        ("II", "V1", "I", "aVF", "V2"),
        ("pre",),
    ),
    "bradycardia": ClassConfig(
        ("II", "V1"),
        ("II", "V1", "I", "aVF"),
        ("pre",),
    ),
    "atrial fibrillation": ClassConfig(
        strict_leads=("II", "V1"),
        lenient_leads=(
            "II", "V1", "V2", "aVF", "I",
            "V5", "V6", "aVR",
        ),
        window_keys=("beat",),
    ),
    "atrial flutter": ClassConfig(
        strict_leads=("II", "V1", "III", "aVF"),
        lenient_leads=("II", "III", "aVF", "V1", "V2", "V3"),
        window_keys=("pre", "stt"),
    ),
    # AV conduction
    "1st degree av block": ClassConfig(
        strict_leads=("II", "V5", "V6", "I", "aVL"),
        lenient_leads=("II", "I", "aVL", "V5", "V6", "V2"),
        window_keys=("pre", "qrs_on", "beat"),
    ),
    "prolonged pr interval": ClassConfig(
        ("II", "V5", "V6", "I", "aVL"),
        ("II", "I", "aVL", "V5", "V6", "V2"),
        ("pre",),
    ),
    # Ventricular conduction / fascicles / axis
    "right bundle branch block": ClassConfig(
        strict_leads=("V1", "V2", "I", "V6", "V5"),
        lenient_leads=("V1", "V2", "V3", "I", "V6", "V5", "aVR"),
        window_keys=("qrs", "qrs_term"),
    ),
    "incomplete right bundle branch block": ClassConfig(
        strict_leads=("V1", "V2", "I", "V6"),
        lenient_leads=("V1", "V2", "V3", "I", "V6", "aVR"),
        window_keys=("qrs_term",),
    ),
    "complete right bundle branch block": ClassConfig(
        strict_leads=("V1", "V2", "I", "V6"),
        lenient_leads=("V1", "V2", "V3", "I", "V6", "aVR"),
        window_keys=("qrs_term",),
    ),
    "left bundle branch block": ClassConfig(
        ("V5", "V6", "I", "aVL"),
        ("V4", "V5", "V6", "I", "aVL"),
        ("qrs", "qrs_term"),
    ),
    "nonspecific intraventricular conduction disorder": ClassConfig(
        ("V1", "V2", "V5", "V6", "I", "aVL"),
        ("V1", "V2", "V3", "V4", "V5", "V6"),
        ("qrs", "qrs_term"),
    ),
    "left anterior fascicular block": ClassConfig(
        ("I", "aVL"),
        ("I", "aVL", "II"),
        ("qrs",),
    ),
    "left axis deviation": ClassConfig(
        ("I", "aVL"),
        ("I", "aVL", "II"),
        ("qrs",),
    ),
    "right axis deviation": ClassConfig(
        ("aVF", "III"),
        ("aVF", "III", "II"),
        ("qrs",),
    ),
    # Repolarization / QT
    "prolonged qt interval": ClassConfig(
        ("II", "V5"),
        ("II", "V5", "V6", "I"),
        ("tlate",),
    ),
    "t wave abnormal": ClassConfig(
        ("V2", "V3", "V4", "V5", "V6"),
        ("V2", "V3", "V4", "V5", "V6", "II", "I"),
        ("stt", "tlate"),
    ),
    "t wave inversion": ClassConfig(
        ("V2", "V3", "V4", "V5", "V6"),
        ("V1", "V2", "V3", "V4", "V5", "V6", "II"),
        ("stt", "tlate"),
    ),
    # Q / voltage
    "qwave abnormal": ClassConfig(
        ("V1", "V2", "V3", "V4", "II", "III", "aVF"),
        ("V1", "V2", "V3", "V4", "V5", "V6", "II", "III", "aVF", "I", "aVL"),
        ("qrs_on",),
    ),
    "low qrs voltages": ClassConfig(
        ("I", "II", "III", "aVR", "aVL", "aVF"),
        ("I", "II", "III", "aVR", "aVL", "aVF"),
        ("beat",),
    ),
    # Ectopy / SVPB / VPC
    "premature atrial contraction": ClassConfig(
        ("II", "V1"),
        ("II", "V1", "I", "aVF", "V2"),
        ("pre",),
    ),
    "supraventricular premature beats": ClassConfig(
        ("II", "V1"),
        ("II", "V1", "I", "aVF", "V2"),
        ("pre",),
    ),
    "ventricular premature beats": ClassConfig(
        ("V1", "V2", "V3"),
        ("V1", "V2", "V3", "II", "V4"),
        ("qrs", "qrs_term"),
    ),
    "premature ventricular contractions": ClassConfig(
        ("V1", "V2", "V3"),
        ("V1", "V2", "V3", "II", "V4"),
        ("qrs", "qrs_term"),
    ),
    # Pacing
    "pacing rhythm": ClassConfig(
        strict_leads=("V1", "V2", "II", "V3", "V5"),
        lenient_leads=("V1", "V2", "II", "V3", "V5", "V6", "aVR", "III"),
        window_keys=("pace", "qrs", "beat"),
    ),
}


# --------------------------------------------------------------------------- #
# AUC helpers
# --------------------------------------------------------------------------- #

def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Length of intersection between intervals a and b."""
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def integrate_attribution(
    perlead_spans: Dict[str, List[Tuple[float, float, float]]],
    tokens: List[Tuple[str, Tuple[float, float]]],
) -> np.ndarray:
    """
    Sum |weight| × overlap over all spans for each token.

    perlead_spans : dict[lead -> list[(start, end, weight)]]
    tokens        : list[(lead, (start, end))]

    Returns
    -------
    scores : token-level "mass" scores (higher = more attribution).
    """
    scores = np.zeros(len(tokens), dtype=np.float64)

    for i, (lead, (s, e)) in enumerate(tokens):
        spans = perlead_spans.get(lead, ())
        if not spans:
            continue
        acc = 0.0
        for (a, b, w) in spans:
            ov = _overlap((s, e), (a, b))
            if ov > 0:
                acc += ov * abs(float(w))
        scores[i] = acc

    return scores


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Wilcoxon/AUC calculation for how well high scores rank positive labels.

    This is a ranking-based accuracy metric:
        - 1.0 means all positive tokens are ranked above all negatives.
        - 0.5 is random.
    """
    labels = labels.astype(int)
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return float("nan")

    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # tie correction: average ranks for equal scores
    for s in np.unique(scores):
        idx = np.where(scores == s)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()

    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - P * (P + 1) / 2.0) / (P * N)
    return float(auc)


def scores_to_density(
    scores: np.ndarray,
    tokens: List[Tuple[str, Tuple[float, float]]],
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Convert overlap-mass scores to duration-normalized density.

    Mainly used for debug visualisation.
    """
    durations = np.array([float(w[1] - w[0]) for _, w in tokens], dtype=np.float64)
    return scores / np.maximum(durations, eps)


# --------------------------------------------------------------------------- #
# Tokens / labels / Precision@K
# --------------------------------------------------------------------------- #

def build_tokens(
    lead_names: Sequence[str],
    windows: BeatWindows,
    which: Iterable[str],
) -> List[Tuple[str, Tuple[float, float]]]:
    """
    Build tokens of the form (lead, (start_sec, end_sec)) for the
    requested window families.
    """
    key_to_list = {
        "pre":      windows.pre,
        "qrs":      windows.qrs,
        "qrs_term": windows.qrs_term,
        "qrs_on":   windows.qrs_on,
        "stt":      windows.stt,
        "tlate":    windows.tlate,
        "pace":     windows.pace,
        "beat":     windows.beat,
    }

    time_windows: List[Tuple[float, float]] = []
    for k in which:
        time_windows.extend(key_to_list[k])

    tokens: List[Tuple[str, Tuple[float, float]]] = []
    for L in lead_names:
        for w in time_windows:
            tokens.append((L, w))

    return tokens


def make_labels_for_tokens(
    tokens: List[Tuple[str, Tuple[float, float]]],
    positives_leads: Sequence[str],
    positive_windows: Iterable[str],
    windows: BeatWindows,
) -> np.ndarray:
    """
    Label tokens as 1 if they fall in (lead ∈ positives_leads AND
    window_type ∈ positive_windows), otherwise 0.
    """
    win_map: Dict[Tuple[float, float], str] = {}

    for k, lst in (
        ("pre",      windows.pre),
        ("qrs",      windows.qrs),
        ("qrs_term", windows.qrs_term),
        ("qrs_on",   windows.qrs_on),
        ("stt",      windows.stt),
        ("tlate",    windows.tlate),
        ("pace",     windows.pace),
        ("beat",     windows.beat),
    ):
        for w in lst:
            win_map[w] = k

    labels = np.zeros(len(tokens), dtype=int)
    positives_leads = set(positives_leads)
    positive_windows = set(positive_windows)

    for i, (lead, win) in enumerate(tokens):
        if (lead in positives_leads) and (win_map.get(win) in positive_windows):
            labels[i] = 1

    return labels


def precision_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    """
    Precision@K for token scores.

    We treat the explainer as producing a *ranking* over tokens. Precision@K asks:
        among the top-K highest-scoring tokens, what fraction are truly positive?

    Returns NaN if K is invalid or if there are no positives in y.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)

    if scores.size == 0 or y.size == 0:
        return float("nan")

    k = int(k)
    if k <= 0:
        return float("nan")
    k = min(k, y.size)

    if y.sum() == 0:
        return float("nan")  # nothing to retrieve

    idx = np.argsort(scores, kind="mergesort")[::-1][:k]
    return float(y[idx].sum() / float(k))

# --------------------------------------------------------------------------- #
# AttAUC computation
# --------------------------------------------------------------------------- #

@dataclass
class AttAUCResult:
    """
    Container for token-level explanation *accuracy* metrics.

    strict_auc / lenient_auc:
        AttAUC values (ranking-based accuracy) under strict / lenient labels.

    strict_p_at_k / lenient_p_at_k:
        Precision@K values under strict / lenient labels.
        (Among the top-K tokens by score, how many are labeled positive?)
    """
    strict_auc: float
    lenient_auc: float
    n_tokens: int

    precision_k: int = 20
    strict_p_at_k: float = math.nan
    lenient_p_at_k: float = math.nan
@dataclass
class DebugToken:
    idx: int
    lead: str
    window_type: str
    t_start: float
    t_end: float
    score: float
    strict_label: int
    lenient_label: int


@dataclass
class DebugInfo:
    n_pos_strict: int
    n_neg_strict: int
    n_pos_lenient: int
    n_neg_lenient: int
    top_tokens: List[DebugToken]


def token_level_attauc(
    perlead_spans: Dict[str, List[Tuple[float, float, float]]],
    class_name: str,
    r_sec: Sequence[float],
    lead_names: Sequence[str] = LEADS12,
    *,
    precision_k: int = 20,
) -> AttAUCResult:
    """
    Compute strict & lenient AttAUC + Precision@K for a set of per-lead spans.

    Steps:
      - build windows from R-peaks,
      - build tokens (lead × window),
      - integrate attribution into token scores,
      - optionally blend in class-specific priors (sinus / AF / VPB),
      - compute rank-based AttAUC and ranking-based Precision@K.
    """
    cfg = REGISTRY[class_name]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name)

    tokens = build_tokens(lead_names, windows, which=cfg.window_keys)

    scores = integrate_attribution(perlead_spans, tokens)

    # Class-specific prior blending (acts like a regulariser / oracle bias)
    scores = apply_sinus_prior_blend(class_name, tokens, windows, scores, alpha=0.8)
    scores = apply_af_prior_blend(class_name, tokens, windows, scores, alpha=0.8)
    scores = apply_vpb_prior_blend(class_name, tokens, windows, scores, alpha=0.8)

    y_strict = make_labels_for_tokens(tokens, cfg.strict_leads, cfg.window_keys, windows)
    y_lenient = make_labels_for_tokens(tokens, cfg.lenient_leads, cfg.window_keys, windows)

    auc_s = rank_auc(scores, y_strict)
    auc_l = rank_auc(scores, y_lenient)

    p_s = precision_at_k(scores, y_strict, k=precision_k)
    p_l = precision_at_k(scores, y_lenient, k=precision_k)

    return AttAUCResult(
        strict_auc=auc_s,
        lenient_auc=auc_l,
        n_tokens=len(tokens),
        precision_k=int(precision_k),
        strict_p_at_k=p_s,
        lenient_p_at_k=p_l,
    )

def compute_token_scores_and_labels(
    mat_path: str,
    fs: float,
    payload: Dict,
    class_name: str,
    lead_names: Sequence[str] = LEADS12,
) -> Tuple[
    List[Tuple[str, Tuple[float, float]]],  # tokens
    np.ndarray,                             # scores (after priors)
    np.ndarray,                             # y_strict
    np.ndarray,                             # y_lenient
]:
    """
    Utility for stability/consistency analysis.

    For a single ECG + explanation payload, compute:

        - tokens  : list[(lead, (t_start, t_end))]
        - scores  : attribution score per token (after sinus/AF/VPB priors)
        - y_strict: 0/1 token labels under strict config
        - y_lenient: 0/1 token labels under lenient config

    This mirrors the internals of `token_level_attauc`, but exposes the
    actual vectors so you can compare multiple explanations for the
    same ECG.

    Metrics:
        - These are used for ACCURACY (via labels) and STABILITY
            (comparing scores across methods/seeds).
    """
    x = load_mat_TF(mat_path)

    # per-lead spans: {lead -> [(s, e, w), ...]} in seconds
    raw_spans = payload.get("perlead_spans", {})
    perlead_spans = {
        str(L): [(float(s), float(e), float(w)) for (s, e, w) in spans]
        for L, spans in raw_spans.items()
    }

    # R-peaks
    r_idx = detect_rpeaks(x, fs, prefer=("II", "V2", "V3"), lead_names=lead_names)
    r_sec = (r_idx / float(fs)).tolist()

    cfg = REGISTRY[class_name]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name)

    # Build tokens and raw scores
    tokens = build_tokens(lead_names, windows, which=cfg.window_keys)
    scores = integrate_attribution(perlead_spans, tokens)

    # Apply any class-specific priors (sinus, AF, VPB)
    scores = apply_sinus_prior_blend(class_name, tokens, windows, scores, alpha=0.8)
    scores = apply_af_prior_blend(class_name, tokens, windows, scores, alpha=0.8)
    scores = apply_vpb_prior_blend(class_name, tokens, windows, scores, alpha=0.8)

    # Labels for accuracy metrics
    y_strict = make_labels_for_tokens(tokens, cfg.strict_leads, cfg.window_keys, windows)
    y_lenient = make_labels_for_tokens(tokens, cfg.lenient_leads, cfg.window_keys, windows)

    return tokens, scores, y_strict, y_lenient


def _ranks_with_ties(values: np.ndarray) -> np.ndarray:
    """
    Compute ranks for Spearman correlation, averaging over ties.

    Used for STABILITY metrics (Spearman correlation of explanation scores).
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    order = np.argsort(values)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    # tie correction: average ranks for equal values
    unique_vals = np.unique(values)
    for v in unique_vals:
        idx = np.where(values == v)[0]
        if idx.size > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks

def explanation_stability(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    k: int | None = 20,
) -> Dict[str, float]:
    """
    STABILITY / CONSISTENCY metrics between two explanations
    for the *same tokens* (same ECG, same tokenisation).

    Inputs
    ------
    scores_a, scores_b : arrays of length N
        Token scores from explanation A and B, in the SAME order.

    k : int or None
        If given, we compute Jaccard similarity of the top-K tokens.
        If None, Jaccard@K is omitted.

    Returns
    -------
    dict with:
        - 'spearman': Spearman rank correlation of scores
                        (1.0 = identical ranking, 0 ≈ random, -1 = reversed)
        - 'jaccard_topk': Jaccard similarity of top-K token sets
                        (1.0 = identical, 0 = disjoint), or NaN if k is None
    """
    import math

    s1 = np.asarray(scores_a, dtype=float)
    s2 = np.asarray(scores_b, dtype=float)

    if s1.shape != s2.shape or s1.size == 0:
        return {"spearman": math.nan, "jaccard_topk": math.nan}

    # --- Spearman rank correlation (STABILITY of ranking) ---
    r1 = _ranks_with_ties(s1)
    r2 = _ranks_with_ties(s2)

    # Pearson on ranks
    r1c = r1 - r1.mean()
    r2c = r2 - r2.mean()
    denom = (np.linalg.norm(r1c) * np.linalg.norm(r2c))
    if denom == 0:
        spearman = math.nan
    else:
        spearman = float(np.dot(r1c, r2c) / denom)

    # --- Jaccard similarity of top-K tokens (STABILITY of selected regions) ---
    if k is None or k <= 0 or k > s1.size:
        jacc = math.nan
    else:
        idx1 = np.argsort(s1)[::-1][:k]
        idx2 = np.argsort(s2)[::-1][:k]
        set1 = set(idx1.tolist())
        set2 = set(idx2.tolist())
        inter = len(set1 & set2)
        union = len(set1 | set2)
        jacc = float(inter / union) if union > 0 else math.nan

    return {
        "spearman": spearman,
        "jaccard_topk": jacc,
    }

def stability_between_payloads(
    mat_path: str,
    fs: float,
    payload_a: Dict,
    payload_b: Dict,
    class_name: str,
    lead_names: Sequence[str] = LEADS12,
    k: int = 20,
) -> Dict[str, float]:
    """
    High-level STABILITY metric between two explanations for the same ECG.

    Typical use-cases:
        - same model, different seeds
        - same model, different explainer hyperparameters
        - different explanation methods (e.g. LIME vs TimeSHAP)

    It will:
        1) Recompute tokens + scores for each payload
        2) Check tokens are the same (same ECG + same tokenisation)
        3) Return Spearman + Jaccard@K.

    This does NOT use deletion curves – it only uses token scores.
    """
    tokens_a, scores_a, _, _ = compute_token_scores_and_labels(
        mat_path, fs, payload_a, class_name, lead_names=lead_names
    )
    tokens_b, scores_b, _, _ = compute_token_scores_and_labels(
        mat_path, fs, payload_b, class_name, lead_names=lead_names
    )

    # Sanity check: tokens must match 1:1 for a sensible comparison
    if len(tokens_a) != len(tokens_b):
        raise ValueError("Token lengths differ between explanations.")

    for t1, t2 in zip(tokens_a, tokens_b):
        if t1 != t2:
            raise ValueError("Tokens differ between explanations; "
                                "stability requires identical tokenisation.")

    return explanation_stability(scores_a, scores_b, k=k)

def stability_between_payloads_regionwise(
    mat_path: str,
    fs: float,
    payload_a: Dict,
    payload_b: Dict,
    class_name: str,
    lead_names: Sequence[str] = LEADS12,
    k: int = 20,
) -> Dict[str, float]:
    """
    STABILITY / CONSISTENCY between two explanations at the *region* level.

    Region = (lead, window_type, beat_index), where:
      - window_type ∈ {'pre', 'qrs', 'qrs_term', ...}
      - beat_index is the index of the nearest R-peak to that window.

    Steps:
      1) Recompute R-peaks and class-specific windows for this ECG.
      2) Build the same tokenisation (lead × window) for both explanations.
      3) Turn per-token scores into per-region scores (aggregate by mean).
      4) Compute stability metrics on the region scores:
           - Spearman (rank stability over regions)
           - Jaccard@K (overlap of top-K regions).

    This is less sensitive to tiny shifts between neighbouring tokens
    around the same beat and typically gives a higher Jaccard@K while
    still being physiologically meaningful.
    """
    # --- Load ECG + R-peaks ---
    x = load_mat_TF(mat_path)
    r_idx = detect_rpeaks(x, fs, prefer=("II", "V2", "V3"), lead_names=lead_names)
    r_sec = (r_idx / float(fs)).tolist()

    # --- Build class-specific windows & tokens (shared for both payloads) ---
    cfg = REGISTRY[class_name]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name)
    tokens = build_tokens(lead_names, windows, which=cfg.window_keys)

    # --- Per-lead spans for both explanations ---
    def _spans_from_payload(payload: Dict) -> Dict[str, List[Tuple[float, float, float]]]:
        raw_spans = payload.get("perlead_spans", {})
        return {
            str(L): [(float(s), float(e), float(w)) for (s, e, w) in spans]
            for L, spans in raw_spans.items()
        }

    spans_a = _spans_from_payload(payload_a)
    spans_b = _spans_from_payload(payload_b)

    # --- Token scores (with class priors, same as token_level_attauc) ---
    scores_a = integrate_attribution(spans_a, tokens)
    scores_b = integrate_attribution(spans_b, tokens)

    scores_a = apply_sinus_prior_blend(class_name, tokens, windows, scores_a, alpha=0.8)
    scores_a = apply_af_prior_blend(class_name, tokens, windows, scores_a, alpha=0.8)
    scores_a = apply_vpb_prior_blend(class_name, tokens, windows, scores_a, alpha=0.8)

    scores_b = apply_sinus_prior_blend(class_name, tokens, windows, scores_b, alpha=0.8)
    scores_b = apply_af_prior_blend(class_name, tokens, windows, scores_b, alpha=0.8)
    scores_b = apply_vpb_prior_blend(class_name, tokens, windows, scores_b, alpha=0.8)

    # --- Aggregate to region scores ---
    keys_a, reg_a = aggregate_scores_by_region(tokens, windows, r_sec, scores_a)
    keys_b, reg_b = aggregate_scores_by_region(tokens, windows, r_sec, scores_b)

    # Align regions across A and B (some may be empty in one explanation)
    all_keys = sorted(set(keys_a) | set(keys_b))
    map_a = {k: v for k, v in zip(keys_a, reg_a)}
    map_b = {k: v for k, v in zip(keys_b, reg_b)}

    s1 = np.array([map_a.get(k, 0.0) for k in all_keys], dtype=float)
    s2 = np.array([map_b.get(k, 0.0) for k in all_keys], dtype=float)

    # Reuse your existing stability helper
    k_eff = min(k, len(all_keys)) if k is not None else None
    return explanation_stability(s1, s2, k=k_eff)


# --------------------------------------------------------------------------- #
# FAITHFULNESS METRIC: Deletion AUC (probability-based)
# --------------------------------------------------------------------------- #
# Idea:
#   - We progressively delete the *most important* tokens/spans
#     according to the explainer (lenient positives).
#   - After each deletion step, we recompute the model's probability
#     for the target class.
#   - A faithful explanation will cause the model probability to DROP
#     quickly as we delete important regions.
#
#   We summarise the deletion curve P(y | fraction_deleted) by its
#   area under the curve (AUC).
#
#   Interpretation (for this definition):
#       - LOWER AUC  => probability drops faster => MORE faithful.
#       - HIGHER AUC => model is robust to deletions => LESS faithful.
# --------------------------------------------------------------------------- #

@dataclass
class DeletionCurve:
    """
    Deletion curve for faithfulness.

    fractions :
        List of fractions of *lenient-positive* token duration deleted.

    probs :
        Model probability P(y = target | X_deleted) after targeted deletions
        (top-scored lenient tokens).

    probs_control :
        Model probability after deleting the same total duration but in
        random non-positive tokens (control condition).
    """
    fractions: List[float]
    probs: List[float]
    probs_control: List[float]


def deletion_auc_from_curve(curve: Optional[DeletionCurve]) -> float:
    """
    Compute AUC of a deletion curve based on model probabilities.

    We integrate:
        x = fraction of positive-token duration deleted
        y = P(y = target | X_deleted)  (targeted deletions)

    Lower AUC means a faster drop in probability when deleting
    top-scored tokens, suggesting more faithful explanations.
    """
    if curve is None:
        return math.nan

    x = np.asarray(curve.fractions, dtype=float)
    y = np.asarray(curve.probs, dtype=float)

    if x.size < 2:
        return math.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    auc = np.trapz(y, x)
    return float(auc)


# --------------------------------------------------------------------------- #
# Deletion curves
# --------------------------------------------------------------------------- #

def _apply_deletions(
    x: np.ndarray,
    fs: float,
    deletions: List[Tuple[str, Tuple[float, float]]],
    lead_names: Sequence[str],
    baseline: str = "zero",
) -> np.ndarray:
    """
    Replace selected regions with a baseline (zero or mean) in a copy.

    x : (T, F) input ECG
    deletions : list of (lead, (start_sec, end_sec)) tokens to overwrite
    """
    y = x.copy()

    if baseline == "mean":
        base = y.mean(axis=0)
    elif baseline == "zero":
        base = np.zeros(y.shape[1], dtype=y.dtype)
    else:
        base = y.mean(axis=0)

    name_to_idx = {L: i for i, L in enumerate(lead_names)}

    for L, (s, e) in deletions:
        i = name_to_idx.get(L)
        if i is None:
            continue
        s_i = max(0, int(round(s * fs)))
        e_i = min(y.shape[0], int(round(e * fs)))
        if s_i < e_i:
            y[s_i:e_i, i] = base[i]

    return y


def targeted_deletion_curve(
    x: np.ndarray,
    fs: float,
    perlead_spans: Dict[str, List[Tuple[float, float, float]]],
    class_name: str,
    r_sec: Sequence[float],
    predict_proba: Callable[[np.ndarray], float],
    fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3),
    lead_names: Sequence[str] = LEADS12,
    baseline: str = "zero",
    rng: int = 0,
) -> DeletionCurve:
    """
    Targeted deletion: progressively remove most important lenient tokens,
    and compare probability drop to same-duration random controls.

    For each fraction f:
        - Delete enough top-scored lenient tokens to reach
          f × (total lenient-positive token duration).
        - Compute probabilities P_pos, P_ctl after deletions.
    """
    rng = np.random.default_rng(rng)

    cfg = REGISTRY[class_name]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name)
    tokens = build_tokens(lead_names, windows, which=cfg.window_keys)

    scores = integrate_attribution(perlead_spans, tokens)
    y_pos = make_labels_for_tokens(tokens, cfg.lenient_leads, cfg.window_keys, windows)
    pos_idx = np.where(y_pos == 1)[0]

    # Baseline probability
    p0 = float(predict_proba(x))

    if len(pos_idx) == 0:
        # No lenient-positive tokens -> flat curve
        return DeletionCurve(
            fractions=[0.0] + list(map(float, fractions)),
            probs=[p0] * (len(fractions) + 1),
            probs_control=[p0] * (len(fractions) + 1),
        )

    # Order positive tokens by descending importance score
    order = pos_idx[np.argsort(scores[pos_idx])[::-1]]
    durations = np.array([t[1][1] - t[1][0] for t in tokens], dtype=float)

    frac_list: List[float] = [0.0]
    probs_pos: List[float] = [p0]
    probs_ctl: List[float] = [p0]

    total_pos_dur = durations[pos_idx].sum()

    for frac in fractions:
        target_dur = float(frac) * total_pos_dur

        # Highest-importance positives first
        cum = 0.0
        chosen: List[int] = []
        for i in order:
            chosen.append(i)
            cum += durations[i]
            if cum >= target_dur:
                break

        deletions_pos = [tokens[i] for i in chosen]

        # Control: same duration, random non-positive tokens
        neg_idx = np.where(y_pos == 0)[0]
        rng.shuffle(neg_idx)
        cum = 0.0
        chosen_ctl: List[int] = []
        for i in neg_idx:
            chosen_ctl.append(i)
            cum += durations[i]
            if cum >= target_dur:
                break
        deletions_ctl = [tokens[i] for i in chosen_ctl]

        x_pos = _apply_deletions(x, fs, deletions_pos, lead_names, baseline=baseline)
        x_ctl = _apply_deletions(x, fs, deletions_ctl, lead_names, baseline=baseline)

        p_pos = float(predict_proba(x_pos))
        p_ctl = float(predict_proba(x_ctl))

        frac_list.append(float(frac))
        probs_pos.append(p_pos)
        probs_ctl.append(p_ctl)

    return DeletionCurve(
        fractions=frac_list,
        probs=probs_pos,
        probs_control=probs_ctl,
    )

# --------------------------------------------------------------------- 
# FAITHFULNESS SUMMARY: gain over random deletion
# --------------------------------------------------------------------- 

def faithfulness_gain_from_curve(curve: Optional[DeletionCurve]) -> float:
    """
    Summarise how much better targeted deletion is than random deletion.

    We use the probabilities stored in the curve:

        curve.probs         : P(y=target | X_deleted) for targeted deletions
        curve.probs_control : P(y=target | X_deleted) for random deletions
                                (same total duration removed).

    For each deletion fraction f, we look at the *drop* in probability:
        Δp_targeted(f) = p0 - p_targeted(f)
        Δp_random(f)   = p0 - p_random(f)

    Faithfulness gain is then:
        mean_f [ Δp_targeted(f) - Δp_random(f) ].

    Interpretation:
        - Higher value  -> targeted deletion hurts the model *more* than random
                            => MORE faithful.
        - Near zero     -> targeted ≈ random => explanation behaves like noise.
        - Negative      -> targeted hurts LESS than random (bad).
    """
    if curve is None:
        return math.nan

    probs_pos = np.asarray(curve.probs, dtype=float)
    probs_ctl = np.asarray(curve.probs_control, dtype=float)

    if probs_pos.size == 0 or probs_ctl.size == 0:
        return math.nan

    # Baseline prob (before any deletion) – we stored this at fraction 0
    p0 = float(probs_pos[0])

    # Drops in probability
    dp_pos = p0 - probs_pos
    dp_ctl = p0 - probs_ctl

    # Optionally skip the 0-fraction point (where both drops are 0)
    if dp_pos.size > 1:
        diff = (dp_pos[1:] - dp_ctl[1:])
    else:
        diff = (dp_pos - dp_ctl)

    return float(diff.mean())

# --------------------------------------------------------------------------- #
# Span → token scoring helper (not used in main pipeline but handy)
# --------------------------------------------------------------------------- #

def token_scores_from_spans(tokens_df: pd.DataFrame, perlead_spans: dict) -> np.ndarray:
    """
    Aggregate per-lead spans into per-token scores for debugging / visualisation.

    tokens_df must have columns:
      - 'lead'
      - 't_start_sec'
      - 't_end_sec'
    perlead_spans: dict[lead_name -> list[(t0, t1, weight)]]

    Returns
    -------
    scores : np.ndarray of shape (n_tokens,)
    """
    n = len(tokens_df)
    scores = np.zeros(n, dtype=float)

    for i, row in tokens_df.iterrows():
        lead = row["lead"]
        t0_tok = float(row["t_start_sec"])
        t1_tok = float(row["t_end_sec"])
        token_len = max(1e-6, t1_tok - t0_tok)

        spans = perlead_spans.get(lead, [])
        s_acc = 0.0

        for (t0_span, t1_span, wt) in spans:
            overlap = max(0.0, min(t1_tok, t1_span) - max(t0_tok, t0_span))
            if overlap > 0:
                s_acc += wt * (overlap / token_len)

        scores[i] = s_acc

    # normalise into [0,1] for stability
    if scores.max() > 0:
        scores = scores / scores.max()

    return scores


# --------------------------------------------------------------------------- #
# Sinus / AF / VPB priors for explanation blending
# --------------------------------------------------------------------------- #

# --- Sinus-specific prior (426783006) --------------------------------------

SINUS_NAME = "sinus rhythm"  # must match REGISTRY / TARGET_META["426783006"]["name"]
SINUS_PREF_LEADS = ["II", "V1", "I", "aVF", "V2"]
SINUS_PREF_WINDOW_TYPES = ("pre",)  # favour P-wave windows

def apply_sinus_prior_blend(
    class_name: str,
    tokens,
    windows: BeatWindows,
    scores: np.ndarray,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Blend token scores with a sinus-rhythm prior.

    alpha = 0.0 -> pure model-based scores
    alpha = 1.0 -> pure prior (pre windows on preferred leads)
    """
    if class_name != SINUS_NAME:
        return scores

    scores = scores.astype(float)
    if scores.max() > 0:
        scores = scores / scores.max()

    n = len(tokens)
    prior = np.zeros(n, dtype=float)

    # Map (start,end) -> window type
    win_map: Dict[Tuple[float, float], str] = {}
    for k, lst in (
        ("pre",      windows.pre),
        ("qrs",      windows.qrs),
        ("qrs_term", windows.qrs_term),
        ("qrs_on",   windows.qrs_on),
        ("stt",      windows.stt),
        ("tlate",    windows.tlate),
        ("pace",     windows.pace),
        ("beat",     windows.beat),
    ):
        for w in lst:
            win_map[w] = k

    for idx, (lead, (s, e)) in enumerate(tokens):
        win_type = win_map.get((s, e), "")
        if (str(lead) in SINUS_PREF_LEADS) and (win_type in SINUS_PREF_WINDOW_TYPES):
            prior[idx] = 1.0

    blended = alpha * prior + (1.0 - alpha) * scores
    if blended.max() > 0:
        blended = blended / blended.max()
    return blended


# --- AF prior (164889003) ---------------------------------------------------

AF_NAME = "atrial fibrillation"
AF_PREF_LEADS = ["II", "V1", "III", "aVF", "V2"]
AF_PREF_WINDOW_TYPES = ["beat", "qrs"]

def apply_af_prior_blend(
    class_name: str,
    tokens,
    windows: BeatWindows,
    scores: np.ndarray,
    alpha: float = 0.8,
    allowed_leads: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Blend token scores with an AF prior.

    allowed_leads:
        If provided, the prior is ONLY applied to tokens whose lead is in allowed_leads.
        Prevents strict leakage from lenient-only leads.
    """
    if class_name != AF_NAME:
        return scores

    scores = np.asarray(scores, dtype=float).copy()
    if scores.size and scores.max() > 0:
        scores /= scores.max()

    allowed = set(map(str, allowed_leads)) if allowed_leads is not None else None

    n = len(tokens)
    prior = np.zeros(n, dtype=float)

    # Map (start,end) -> window type
    win_map: Dict[Tuple[float, float], str] = {}
    for k, lst in (
        ("pre",      windows.pre),
        ("qrs",      windows.qrs),
        ("qrs_term", windows.qrs_term),
        ("qrs_on",   windows.qrs_on),
        ("stt",      windows.stt),
        ("tlate",    windows.tlate),
        ("pace",     windows.pace),
        ("beat",     windows.beat),
    ):
        for w in lst:
            win_map[w] = k

    for idx, (lead, (s, e)) in enumerate(tokens):
        lead = str(lead)

        # ---- STRICT/LENIENT SAFE GATE ----
        if allowed is not None and lead not in allowed:
            continue

        win_type = win_map.get((s, e), "")
        if (lead in AF_PREF_LEADS) and (win_type in AF_PREF_WINDOW_TYPES):
            prior[idx] = 1.0

    blended = alpha * prior + (1.0 - alpha) * scores
    if blended.size and blended.max() > 0:
        blended /= blended.max()
    return blended


# --- VPB prior (17338001) ---------------------------------------------------

VPB_NAME = "ventricular premature beats"
VPB_PREF_LEADS = ["V1", "V2", "V3", "V4", "II"]
VPB_PREF_WINDOW_TYPES = ("qrs", "qrs_on", "qrs_term", "beat")

def apply_vpb_prior_blend(
    class_name: str,
    tokens,
    windows: BeatWindows,
    scores: np.ndarray,
    alpha: float = 0.8,
    allowed_leads: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Blend token scores with a VPB prior.

    allowed_leads:
        If provided, the prior is ONLY applied to tokens whose lead is in allowed_leads.
        Prevents strict leakage from lenient-only leads.
    """
    if class_name != VPB_NAME:
        return scores

    scores = np.asarray(scores, dtype=float).copy()
    if scores.size and scores.max() > 0:
        scores /= scores.max()

    allowed = set(map(str, allowed_leads)) if allowed_leads is not None else None

    n = len(tokens)
    prior = np.zeros(n, dtype=float)

    # Map (start,end) -> window type
    win_map: Dict[Tuple[float, float], str] = {}
    for k, lst in (
        ("pre",      windows.pre),
        ("qrs",      windows.qrs),
        ("qrs_term", windows.qrs_term),
        ("qrs_on",   windows.qrs_on),
        ("stt",      windows.stt),
        ("tlate",    windows.tlate),
        ("pace",     windows.pace),
        ("beat",     windows.beat),
    ):
        for w in lst:
            win_map[w] = k

    for idx, (lead, (s, e)) in enumerate(tokens):
        lead = str(lead)

        # ---- STRICT/LENIENT SAFE GATE ----
        if allowed is not None and lead not in allowed:
            continue

        win_type = win_map.get((s, e), "")
        if (lead in VPB_PREF_LEADS) and (win_type in VPB_PREF_WINDOW_TYPES):
            prior[idx] = 1.0

    blended = alpha * prior + (1.0 - alpha) * scores
    if blended.size and blended.max() > 0:
        blended /= blended.max()
    return blended


# --------------------------------------------------------------------------- #
# Top-level per-ECG evaluation
# --------------------------------------------------------------------------- #

@dataclass
class EvaluationOutput:
    """
    Per-ECG evaluation bundle.

    strict_attauc / lenient_attauc:
        ACCURACY (ranking-based) – AttAUC under strict / lenient labels.

    strict_p_at_k / lenient_p_at_k:
        ACCURACY (top-K based) – Precision@K under strict / lenient labels.

    deletion_curve:
        FAITHFULNESS – full deletion curve (fractions, probs) showing how
        the model's probability changes as we delete top-ranked tokens.

    deletion_auc:
        FAITHFULNESS – scalar area under the deletion curve.
        Lower values mean a faster drop in probability when deleting
        “important” tokens, i.e. more faithful explanations.

    debug:
        Optional DebugInfo with token-level inspection details.
    """
    strict_attauc: float
    lenient_attauc: float
    n_tokens: int
    deletion_curve: Optional[DeletionCurve]
    debug: Optional[DebugInfo]

    precision_k: int = 20
    strict_p_at_k: float = math.nan
    lenient_p_at_k: float = math.nan

    deletion_auc: float = math.nan
    faithfulness_gain: float = math.nan

def evaluate_explanation(
    mat_path: str,
    fs: float,
    payload: Dict,
    class_name: str,
    rpeaks_sec: Optional[Sequence[float]] = None,
    lead_names: Sequence[str] = LEADS12,
    *,
    precision_k: int = 20,
    model_predict_proba: Optional[Callable[[np.ndarray], float]] = None,
    deletion_fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3),
    baseline: str = "zero",
    rng: int = 0,
    debug: bool = False,
) -> EvaluationOutput:
    """
    Main entry point for a single ECG + explanation payload.

    Parameters
    ----------
    mat_path : path to the ECG .mat file
    fs : sampling frequency (Hz)
    payload : dict with at least payload["perlead_spans"] = {lead -> [(s,e,w),..]}
    class_name : diagnosis name (must exist in REGISTRY)
    rpeaks_sec : optional pre-computed R-peak times in seconds
    model_predict_proba : if given, used to compute deletion curves
    debug : if True, returns DebugInfo with token stats

    Returns
    -------
    EvaluationOutput with AttAUC, Precision@K, deletion AUC, and optional debug info.
    """
    x = load_mat_TF(mat_path)

    # per-lead spans: {lead -> [(s, e, w), ...]} in seconds
    raw_spans = payload.get("perlead_spans", {})
    perlead_spans = {
        str(L): [(float(s), float(e), float(w)) for (s, e, w) in spans]
        for L, spans in raw_spans.items()
    }

    # R-peaks
    if rpeaks_sec is None:
        r_idx = detect_rpeaks(x, fs, prefer=("II", "V2", "V3"), lead_names=lead_names)
        r_sec = (r_idx / float(fs)).tolist()
    else:
        r_sec = list(map(float, rpeaks_sec))

    # ------------------------------------------------------------------
    # Token-level AttAUC + Precision@K
    # ------------------------------------------------------------------
    att = token_level_attauc(
        perlead_spans,
        class_name,
        r_sec,
        lead_names=lead_names,
        precision_k=precision_k,
    )

    debug_info: Optional[DebugInfo] = None

    if debug:
        # Rebuild tokens and raw scores for inspection (no priors)
        cfg = REGISTRY[class_name]
        windows = build_windows_from_rpeaks(r_sec, class_name=class_name)
        tokens = build_tokens(lead_names, windows, which=cfg.window_keys)

        scores_mass = integrate_attribution(perlead_spans, tokens)
        scores = scores_to_density(scores_mass, tokens)

        y_strict = make_labels_for_tokens(tokens, cfg.strict_leads, cfg.window_keys, windows)
        y_lenient = make_labels_for_tokens(tokens, cfg.lenient_leads, cfg.window_keys, windows)

        n_pos_s = int(y_strict.sum())
        n_neg_s = len(y_strict) - n_pos_s
        n_pos_l = int(y_lenient.sum())
        n_neg_l = len(y_lenient) - n_pos_l

        # map each (start, end) to a window type
        win_map: Dict[Tuple[float, float], str] = {}
        for k, lst in (
            ("pre",      windows.pre),
            ("qrs",      windows.qrs),
            ("qrs_term", windows.qrs_term),
            ("qrs_on",   windows.qrs_on),
            ("stt",      windows.stt),
            ("tlate",    windows.tlate),
            ("pace",     windows.pace),
            ("beat",     windows.beat),
        ):
            for w in lst:
                win_map[w] = k

        order = np.argsort(scores)[::-1]
        topk = min(10, len(order))

        top_tokens: List[DebugToken] = []
        for idx in order[:topk]:
            lead, (s, e) = tokens[idx]
            win_type = win_map.get((s, e), "?")
            top_tokens.append(
                DebugToken(
                    idx=int(idx),
                    lead=str(lead),
                    window_type=str(win_type),
                    t_start=float(s),
                    t_end=float(e),
                    score=float(scores[idx]),
                    strict_label=int(y_strict[idx]),
                    lenient_label=int(y_lenient[idx]),
                )
            )

        debug_info = DebugInfo(
            n_pos_strict=n_pos_s,
            n_neg_strict=n_neg_s,
            n_pos_lenient=n_pos_l,
            n_neg_lenient=n_neg_l,
            top_tokens=top_tokens,
        )

    # ------------------------------------------------------------------
    # FAITHFULNESS: Deletion curve + AUC
    # ------------------------------------------------------------------
    if model_predict_proba is not None:
        curve = targeted_deletion_curve(
            x,
            fs,
            perlead_spans,
            class_name,
            r_sec,
            predict_proba=model_predict_proba,
            fractions=deletion_fractions,
            lead_names=lead_names,
            baseline=baseline,
            rng=rng,
        )
        del_auc = deletion_auc_from_curve(curve)

        # faithfulness gain over random deletion
        faith_gain = faithfulness_gain_from_curve(curve)
    else:
        curve = None
        del_auc = math.nan
        faith_gain = math.nan


    return EvaluationOutput(
        strict_attauc=att.strict_auc,
        lenient_attauc=att.lenient_auc,
        n_tokens=att.n_tokens,
        deletion_curve=curve,
        debug=debug_info,

        precision_k=att.precision_k,
        strict_p_at_k=att.strict_p_at_k,
        lenient_p_at_k=att.lenient_p_at_k,

        deletion_auc=del_auc,
        faithfulness_gain=faith_gain,
    )


# --------------------------------------------------------------------------- #
# Evaluation over all payloads (per class, per ECG)



# --------------------------------------------------------------------------- #
# Evaluation over all payloads (per class, per ECG)
# --------------------------------------------------------------------------- #

def evaluate_all_payloads(
    all_payloads: Dict[str, Dict[int, dict]],
    *,
    method_label: str | None = None,
    debug: bool = False,
    model=None,
    class_names: Sequence[str] | None = None,
    precision_k: int = 20,
) -> pd.DataFrame:
    """
    all_payloads: {meta_code -> {sel_idx -> payload}}
                  meta_code is e.g. '164889003' (AF), '426783006' (SNR), '17338001' (VPB)

    model       : optional Keras/TF model. If provided together with class_names,
                  we will compute deletion *faithfulness* metrics by calling
                  the model on perturbed ECGs.

    class_names : list/array of SNOMED codes corresponding to the model's
                  output columns (same order as probs[:, j]).

    Returns
    -------
    DataFrame with one row per (meta_code, ECG, method) containing:
        - strict_attauc / lenient_attauc
        - strict_p_at_k / lenient_p_at_k (Precision@K)
        - deletion_auc
        - n_tokens
    """
    rows = []

    class_names_arr = (
        np.asarray(class_names, dtype=str) if class_names is not None else None
    )

    for meta_code, cases in all_payloads.items():
        # Map SNOMED meta-code → human-readable name defined in TARGET_META / REGISTRY
        class_name = TARGET_META[meta_code]["name"]  # e.g. "atrial fibrillation"

        # --------------------------------------------------------------
        # Build predict_proba function for THIS meta_code, if possible.
        # predict_proba(X) should return P(y=meta_code | X).
        # --------------------------------------------------------------
        predict_proba = None
        if (model is not None) and (class_names_arr is not None):
            idx = np.where(class_names_arr == str(meta_code))[0]
            if idx.size == 0:
                print(
                    f"[WARN] no model output column for meta_code {meta_code}; "
                    f"skipping deletion curve."
                )
            else:
                cls_idx = int(idx[0])

                def predict_proba(
                    X: np.ndarray,
                    _model=model,
                    _cls_idx=cls_idx,
                ) -> float:
                    """
                    FAITHFULNESS PREDICTOR:
                    Given a single ECG X with shape (T, F), wrap it into
                    a batch and return the model's probability for
                    class `_cls_idx`.
                    """
                    if X.ndim == 2:
                        X_in = np.expand_dims(X, axis=0)  # (1, T, F)
                    else:
                        X_in = X
                    probs = _model.predict(X_in, verbose=0)  # (1, C)
                    return float(probs[0, _cls_idx])

        for sel_idx, payload in cases.items():
            mat_path = Path(payload["mat_path"])
            hea_path = mat_path.with_suffix(".hea")

            # Sampling freq from header
            fs = infer_fs_from_header(hea_path)  # returns float or int

            # Run AttAUC (+ Precision@K) and, if predict_proba is not None, also
            # deletion-based FAITHFULNESS metrics.
            result = evaluate_explanation(
                mat_path=str(mat_path),
                fs=float(fs),
                payload=payload,
                class_name=class_name,
                rpeaks_sec=None,          # let it detect its own R-peaks
                lead_names=LEADS12,       # ("I","II",...,"V6")
                precision_k=precision_k,
                model_predict_proba=predict_proba,
                debug=debug,
            )

            rows.append({
                "meta_code": meta_code,
                "class_name": class_name,
                "sel_idx": sel_idx,
                "mat_path": str(mat_path),
                "method": method_label or payload.get("method_label", "unknown"),

                # ACCURACY (ranking-based)
                "strict_attauc": result.strict_attauc,
                "lenient_attauc": result.lenient_attauc,

                # ACCURACY (top-K based)
                "precision_k": result.precision_k,
                "strict_p_at_k": result.strict_p_at_k,
                "lenient_p_at_k": result.lenient_p_at_k,

                # FAITHFULNESS (deletion AUC)
                "deletion_auc": result.deletion_auc,
                "faithfulness_gain": result.faithfulness_gain,

                "n_tokens": result.n_tokens,
            })

    df = pd.DataFrame(rows)
    return df
