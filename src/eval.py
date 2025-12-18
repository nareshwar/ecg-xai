"""
eval.py — Core ECG explanation evaluation (NO stability / NO augmentation)

Provides:
- BeatWindows: heart-rate aware diagnostic windows around each R-peak
- REGISTRY: mapping from class name -> which leads/windows are "ground truth"
- Token-level metrics:
    * AttAUC (strict / lenient)  [ranking-based accuracy]
    * Precision@K (strict / lenient) [top-K accuracy]
- Faithfulness metrics:
    * Deletion curve per ECG
    * Deletion AUC (lower = more faithful)
    * Faithfulness gain over random deletion

Main entry points:
- evaluate_explanation(): single ECG + payload
- evaluate_all_payloads(): batch evaluation over payload dict/list
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path

from config import MAXLEN
from preprocessing import infer_fs_from_header
from config_targets import TARGET_META

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

    Generic:
    - looks for common keys ('val', 'ECG', 'ecg', 'data', 'signal')
    - fixes orientation so that time is always axis 0.
    """
    d = loadmat(mat_path, squeeze_me=True)

    for key in ("val", "ECG", "ecg", "data", "signal"):
        if key in d:
            arr = np.asarray(d[key])
            break
    else:
        raise ValueError(f"No ECG array found in {mat_path}")

    if arr.ndim == 1:
        arr = arr[None, :]

    # If PhysioNet-style: (n_leads, n_samples) -> transpose
    if arr.shape[0] in (8, 12):
        x = arr.T
    else:
        x = arr

    # Force time on axis 0
    if x.ndim == 2 and x.shape[0] < x.shape[1]:
        x = x.T

    return x.astype(np.float32, copy=False)

def _ensure_len(x: np.ndarray, maxlen: int = MAXLEN) -> np.ndarray:
    """
    Pad/truncate ECG to (maxlen, F). This helps when model expects fixed length.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected (T,F), got shape {x.shape}")

    T, F = x.shape
    if T == maxlen:
        return x
    if T > maxlen:
        return x[:maxlen, :]
    y = np.zeros((maxlen, F), dtype=np.float32)
    y[:T, :] = x
    return y

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
    nyq = fs / 2.0
    lo_n = max(1e-6, lo / nyq)
    hi_n = min(0.999999, hi / nyq)
    if hi_n <= lo_n:
        hi_n = min(0.999999, lo_n + 0.05)
    b, a = butter(order, [lo_n, hi_n], btype="band")
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

    Returns
    -------
    rpeaks : np.ndarray of indices (samples)
    """
    x = np.asarray(x, dtype=np.float32)
    T, F = x.shape

    # choose channels
    if lead_names is not None:
        chosen = [lead_names.index(L) for L in prefer if L in lead_names]
        if not chosen:
            chosen = [int(np.argmax(np.std(x, axis=0)))]
    else:
        chosen = [0]

    xb = _bandpass(x[:, chosen], fs)
    diff = np.diff(xb, axis=0, prepend=xb[:1])
    sq = diff ** 2
    feat = _moving_integral(sq, fs, w_sec=0.15).sum(axis=1)

    med = np.median(feat)
    mad = np.median(np.abs(feat - med))
    thr = med + 0.5 * mad

    distance = int(round(refractory_ms / 1000 * fs))
    if min_prom is None:
        min_prom = 0.5 * np.std(feat)

    peaks, _ = find_peaks(feat, height=thr, distance=distance, prominence=min_prom)

    # refine to local maxima
    ref: List[int] = []
    win = int(0.08 * fs)
    xb_sum = xb.sum(axis=1)
    for p in peaks:
        s = max(0, p - win)
        e = min(T, p + win + 1)
        ref.append(s + int(np.argmax(xb_sum[s:e])))

    return np.array(sorted(set(ref)), dtype=int)

# --------------------------------------------------------------------------- #
# HR-aware beat windows
# --------------------------------------------------------------------------- #

@dataclass
class BeatWindows:
    pre: List[Tuple[float, float]]
    qrs: List[Tuple[float, float]]
    qrs_term: List[Tuple[float, float]]
    qrs_on: List[Tuple[float, float]]
    stt: List[Tuple[float, float]]
    tlate: List[Tuple[float, float]]
    pace: List[Tuple[float, float]]
    beat: List[Tuple[float, float]]

def _median_hr_bpm(r_sec: Sequence[float]) -> float:
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
    hr = _median_hr_bpm(r_sec)

    pre_lo = 0.22 if hr < 100 else 0.18
    pre_hi = 0.05 if hr < 100 else 0.04
    q_on_lo = 0.06
    q_on_hi = 0.01

    qrs_early_post = 0.08
    qrs_term_post = 0.16

    if class_name and ("bundle branch block" in class_name or "intraventricular" in class_name):
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
        pre.append((r - pre_lo,     r - pre_hi))
        qrs_on.append((r - q_on_lo, r - q_on_hi))
        qrs.append((r - 0.05,       r + qrs_early_post))
        qrs_term.append((r + 0.04,  r + qrs_term_post))
        stt.append((r + 0.06,       r + 0.20))
        tlate.append((r + 0.20,     r + 0.40))
        pace.append((r - 0.01,      r + 0.02))
        beat.append((r - 0.30,      r + 0.40))

    return BeatWindows(pre, qrs, qrs_term, qrs_on, stt, tlate, pace, beat)

# --------------------------------------------------------------------------- #
# Registry: class-level ground truth leads + windows
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClassConfig:
    strict_leads: Tuple[str, ...]
    lenient_leads: Tuple[str, ...]
    window_keys: Tuple[str, ...]

REGISTRY: Dict[str, ClassConfig] = {
    "sinus rhythm": ClassConfig(
        ("II", "V1"),
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
        ("II", "V1"),
        ("II", "V1", "V2", "aVF", "I", "V5", "V6", "aVR"),
        ("beat",),
    ),
    "atrial flutter": ClassConfig(
        ("II", "V1", "III", "aVF"),
        ("II", "III", "aVF", "V1", "V2", "V3"),
        ("pre", "stt"),
    ),
    "1st degree av block": ClassConfig(
        ("II", "V5", "V6", "I", "aVL"),
        ("II", "I", "aVL", "V5", "V6", "V2"),
        ("pre", "qrs_on", "beat"),
    ),
    "prolonged pr interval": ClassConfig(
        ("II", "V5", "V6", "I", "aVL"),
        ("II", "I", "aVL", "V5", "V6", "V2"),
        ("pre",),
    ),
    "right bundle branch block": ClassConfig(
        ("V1", "V2", "I", "V6", "V5"),
        ("V1", "V2", "V3", "I", "V6", "V5", "aVR"),
        ("qrs", "qrs_term"),
    ),
    "incomplete right bundle branch block": ClassConfig(
        ("V1", "V2", "I", "V6"),
        ("V1", "V2", "V3", "I", "V6", "aVR"),
        ("qrs_term",),
    ),
    "complete right bundle branch block": ClassConfig(
        ("V1", "V2", "I", "V6"),
        ("V1", "V2", "V3", "I", "V6", "aVR"),
        ("qrs_term",),
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
    "pacing rhythm": ClassConfig(
        ("V1", "V2", "II", "V3", "V5"),
        ("V1", "V2", "II", "V3", "V5", "V6", "aVR", "III"),
        ("pace", "qrs", "beat"),
    ),
}

# --------------------------------------------------------------------------- #
# AUC helpers
# --------------------------------------------------------------------------- #

def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)

def integrate_attribution(
    perlead_spans: Dict[str, List[Tuple[float, float, float]]],
    tokens: List[Tuple[str, Tuple[float, float]]],
) -> np.ndarray:
    """
    Sum |weight| × overlap over all spans for each token.
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
    Wilcoxon/AUC for how well high scores rank positive labels.
    """
    labels = labels.astype(int)
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return float("nan")

    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    for s in np.unique(scores):
        idx = np.where(scores == s)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()

    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - P * (P + 1) / 2.0) / (P * N)
    return float(auc)

# --------------------------------------------------------------------------- #
# Tokens / labels / Precision@K
# --------------------------------------------------------------------------- #

def build_tokens(
    lead_names: Sequence[str],
    windows: BeatWindows,
    which: Iterable[str],
) -> List[Tuple[str, Tuple[float, float]]]:
    key_to_list = {
        "pre": windows.pre,
        "qrs": windows.qrs,
        "qrs_term": windows.qrs_term,
        "qrs_on": windows.qrs_on,
        "stt": windows.stt,
        "tlate": windows.tlate,
        "pace": windows.pace,
        "beat": windows.beat,
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
    win_map: Dict[Tuple[float, float], str] = {}
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

    labels = np.zeros(len(tokens), dtype=int)
    posL = set(map(str, positives_leads))
    posW = set(map(str, positive_windows))
    for i, (lead, win) in enumerate(tokens):
        if (str(lead) in posL) and (win_map.get(win) in posW):
            labels[i] = 1
    return labels

def precision_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)

    if scores.size == 0 or y.size == 0:
        return float("nan")
    k = int(k)
    if k <= 0:
        return float("nan")
    k = min(k, y.size)
    if y.sum() == 0:
        return float("nan")

    idx = np.argsort(scores, kind="mergesort")[::-1][:k]
    return float(y[idx].sum() / float(k))

# --------------------------------------------------------------------------- #
# Priors (optional blending)
# --------------------------------------------------------------------------- #

SINUS_NAME = "sinus rhythm"
AF_NAME = "atrial fibrillation"
VPB_NAME = "ventricular premature beats"

SINUS_PREF_LEADS = {"II", "V1", "I", "aVF", "V2"}
SINUS_PREF_WINDOW_TYPES = {"pre"}

AF_PREF_LEADS = {"II", "V1", "III", "aVF", "V2"}
AF_PREF_WINDOW_TYPES = {"beat", "qrs"}

VPB_PREF_LEADS = {"V1", "V2", "V3", "V4", "II"}
VPB_PREF_WINDOW_TYPES = {"qrs", "qrs_on", "qrs_term", "beat"}

def _win_type_map(windows: BeatWindows) -> Dict[Tuple[float, float], str]:
    m: Dict[Tuple[float, float], str] = {}
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
            m[w] = k
    return m

def apply_sinus_prior_blend(class_name: str, tokens, windows: BeatWindows, scores: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    if class_name != SINUS_NAME:
        return scores
    scores = np.asarray(scores, float).copy()
    if scores.size and scores.max() > 0:
        scores /= scores.max()

    prior = np.zeros(len(tokens), dtype=float)
    wmap = _win_type_map(windows)
    for i, (lead, (s, e)) in enumerate(tokens):
        if str(lead) in SINUS_PREF_LEADS and wmap.get((s, e), "") in SINUS_PREF_WINDOW_TYPES:
            prior[i] = 1.0
    blended = alpha * prior + (1 - alpha) * scores
    if blended.size and blended.max() > 0:
        blended /= blended.max()
    return blended

def apply_af_prior_blend(class_name: str, tokens, windows: BeatWindows, scores: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    if class_name != AF_NAME:
        return scores
    scores = np.asarray(scores, float).copy()
    if scores.size and scores.max() > 0:
        scores /= scores.max()

    prior = np.zeros(len(tokens), dtype=float)
    wmap = _win_type_map(windows)
    for i, (lead, (s, e)) in enumerate(tokens):
        if str(lead) in AF_PREF_LEADS and wmap.get((s, e), "") in AF_PREF_WINDOW_TYPES:
            prior[i] = 1.0
    blended = alpha * prior + (1 - alpha) * scores
    if blended.size and blended.max() > 0:
        blended /= blended.max()
    return blended

def apply_vpb_prior_blend(class_name: str, tokens, windows: BeatWindows, scores: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    if class_name != VPB_NAME:
        return scores
    scores = np.asarray(scores, float).copy()
    if scores.size and scores.max() > 0:
        scores /= scores.max()

    prior = np.zeros(len(tokens), dtype=float)
    wmap = _win_type_map(windows)
    for i, (lead, (s, e)) in enumerate(tokens):
        if str(lead) in VPB_PREF_LEADS and wmap.get((s, e), "") in VPB_PREF_WINDOW_TYPES:
            prior[i] = 1.0
    blended = alpha * prior + (1 - alpha) * scores
    if blended.size and blended.max() > 0:
        blended /= blended.max()
    return blended

# --------------------------------------------------------------------------- #
# Token-level AttAUC computation
# --------------------------------------------------------------------------- #

@dataclass
class AttAUCResult:
    strict_auc: float
    lenient_auc: float
    n_tokens: int
    precision_k: int = 20
    strict_p_at_k: float = math.nan
    lenient_p_at_k: float = math.nan

def token_level_attauc(
    perlead_spans: Dict[str, List[Tuple[float, float, float]]],
    class_name: str,
    r_sec: Sequence[float],
    lead_names: Sequence[str] = LEADS12,
    *,
    precision_k: int = 20,
) -> AttAUCResult:
    cfg = REGISTRY[class_name]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name)
    tokens = build_tokens(lead_names, windows, which=cfg.window_keys)

    scores = integrate_attribution(perlead_spans, tokens)
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

# --------------------------------------------------------------------------- #
# Faithfulness: deletion curves
# --------------------------------------------------------------------------- #

@dataclass
class DeletionCurve:
    fractions: List[float]
    probs: List[float]
    probs_control: List[float]

def deletion_auc_from_curve(curve: Optional[DeletionCurve]) -> float:
    if curve is None:
        return math.nan
    x = np.asarray(curve.fractions, dtype=float)
    y = np.asarray(curve.probs, dtype=float)
    if x.size < 2:
        return math.nan
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))

def faithfulness_gain_from_curve(curve: Optional[DeletionCurve]) -> float:
    if curve is None:
        return math.nan
    probs_pos = np.asarray(curve.probs, dtype=float)
    probs_ctl = np.asarray(curve.probs_control, dtype=float)
    if probs_pos.size == 0 or probs_ctl.size == 0:
        return math.nan
    p0 = float(probs_pos[0])
    dp_pos = p0 - probs_pos
    dp_ctl = p0 - probs_ctl
    diff = (dp_pos[1:] - dp_ctl[1:]) if dp_pos.size > 1 else (dp_pos - dp_ctl)
    return float(np.mean(diff))

def _apply_deletions(
    x: np.ndarray,
    fs: float,
    deletions: List[Tuple[str, Tuple[float, float]]],
    lead_names: Sequence[str],
    baseline: str = "zero",
) -> np.ndarray:
    y = x.copy()
    if baseline == "mean":
        base = y.mean(axis=0)
    else:
        base = np.zeros(y.shape[1], dtype=y.dtype)

    name_to_idx = {L: i for i, L in enumerate(lead_names)}
    for L, (s, e) in deletions:
        i = name_to_idx.get(str(L))
        if i is None:
            continue
        s_i = max(0, int(round(float(s) * fs)))
        e_i = min(y.shape[0], int(round(float(e) * fs)))
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
    rngg = np.random.default_rng(rng)

    cfg = REGISTRY[class_name]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name)
    tokens = build_tokens(lead_names, windows, which=cfg.window_keys)

    scores = integrate_attribution(perlead_spans, tokens)
    y_pos = make_labels_for_tokens(tokens, cfg.lenient_leads, cfg.window_keys, windows)
    pos_idx = np.where(y_pos == 1)[0]

    p0 = float(predict_proba(x))

    if pos_idx.size == 0:
        return DeletionCurve(
            fractions=[0.0] + list(map(float, fractions)),
            probs=[p0] * (len(fractions) + 1),
            probs_control=[p0] * (len(fractions) + 1),
        )

    order = pos_idx[np.argsort(scores[pos_idx])[::-1]]
    durations = np.array([t[1][1] - t[1][0] for t in tokens], dtype=float)
    total_pos_dur = float(durations[pos_idx].sum())

    frac_list = [0.0]
    probs_pos = [p0]
    probs_ctl = [p0]

    for frac in fractions:
        target_dur = float(frac) * total_pos_dur

        cum = 0.0
        chosen: List[int] = []
        for i in order:
            chosen.append(int(i))
            cum += float(durations[i])
            if cum >= target_dur:
                break
        deletions_pos = [tokens[i] for i in chosen]

        neg_idx = np.where(y_pos == 0)[0]
        rngg.shuffle(neg_idx)
        cum = 0.0
        chosen_ctl: List[int] = []
        for i in neg_idx:
            chosen_ctl.append(int(i))
            cum += float(durations[i])
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

    return DeletionCurve(fractions=frac_list, probs=probs_pos, probs_control=probs_ctl)

# --------------------------------------------------------------------------- #
# Top-level per-ECG evaluation
# --------------------------------------------------------------------------- #

@dataclass
class EvaluationOutput:
    strict_attauc: float
    lenient_attauc: float
    n_tokens: int

    precision_k: int = 20
    strict_p_at_k: float = math.nan
    lenient_p_at_k: float = math.nan

    deletion_curve: Optional[DeletionCurve] = None
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
) -> EvaluationOutput:
    x = load_mat_TF(mat_path)

    raw_spans = payload.get("perlead_spans", {})
    perlead_spans = {
        str(L): [(float(s), float(e), float(w)) for (s, e, w) in spans]
        for L, spans in raw_spans.items()
    }

    if rpeaks_sec is None:
        r_idx = detect_rpeaks(x, fs, prefer=("II", "V2", "V3"), lead_names=lead_names)
        r_sec = (r_idx / float(fs)).tolist()
    else:
        r_sec = list(map(float, rpeaks_sec))

    att = token_level_attauc(
        perlead_spans,
        class_name,
        r_sec,
        lead_names=lead_names,
        precision_k=precision_k,
    )

    curve = None
    del_auc = math.nan
    faith_gain = math.nan

    if model_predict_proba is not None:
        curve = targeted_deletion_curve(
            x=x,
            fs=fs,
            perlead_spans=perlead_spans,
            class_name=class_name,
            r_sec=r_sec,
            predict_proba=model_predict_proba,
            fractions=deletion_fractions,
            lead_names=lead_names,
            baseline=baseline,
            rng=rng,
        )
        del_auc = deletion_auc_from_curve(curve)
        faith_gain = faithfulness_gain_from_curve(curve)

    return EvaluationOutput(
        strict_attauc=att.strict_auc,
        lenient_attauc=att.lenient_auc,
        n_tokens=att.n_tokens,
        precision_k=att.precision_k,
        strict_p_at_k=att.strict_p_at_k,
        lenient_p_at_k=att.lenient_p_at_k,
        deletion_curve=curve,
        deletion_auc=del_auc,
        faithfulness_gain=faith_gain,
    )

# --------------------------------------------------------------------------- #
# Batch evaluation
# --------------------------------------------------------------------------- #

def _payload_mat_path(payload: Dict) -> Optional[str]:
    # common keys across your pipeline
    for k in ("mat_path", "filename", "path"):
        if k in payload and payload[k]:
            return str(payload[k])
    return None

def evaluate_all_payloads(
    all_payloads: Dict[str, Union[Dict[int, dict], List[dict]]],
    *,
    method_label: str | None = None,
    model=None,
    class_names: Sequence[str] | None = None,
    precision_k: int = 20,
    baseline: str = "zero",
    deletion_fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3),
    skip_missing_files: bool = True,
) -> pd.DataFrame:
    """
    all_payloads:
      - dict[meta_code -> dict[sel_idx -> payload]]
        OR
      - dict[meta_code -> list[payload]]  (where list index is sel_idx)

    Returns DataFrame with one row per (meta_code, sel_idx).
    """
    rows = []

    class_names_arr = np.asarray(class_names, dtype=str) if class_names is not None else None

    for meta_code, cases in all_payloads.items():
        meta_code_str = str(meta_code)
        if meta_code_str not in TARGET_META:
            print(f"[WARN] meta_code {meta_code_str} not in TARGET_META, skipping")
            continue

        class_name = str(TARGET_META[meta_code_str]["name"])

        predict_proba = None
        if model is not None and class_names_arr is not None:
            idx = np.where(class_names_arr == meta_code_str)[0]
            if idx.size == 0:
                print(f"[WARN] No model output column for {meta_code_str}; skipping deletion metrics.")
            else:
                cls_idx = int(idx[0])

                def predict_proba(X: np.ndarray, _model=model, _cls_idx=cls_idx) -> float:
                    X2 = _ensure_len(X, MAXLEN)
                    X_in = np.expand_dims(X2, axis=0)  # (1, T, F)
                    probs = _model.predict(X_in, verbose=0)
                    return float(probs[0, _cls_idx])

        # support dict or list for cases
        if isinstance(cases, dict):
            items = list(cases.items())
        else:
            items = list(enumerate(cases))

        for sel_idx, payload in items:
            mp = _payload_mat_path(payload)
            if mp is None:
                print(f"[WARN] payload missing mat_path/filename for {meta_code_str}, sel_idx={sel_idx}, skipping")
                continue

            mat_path = Path(mp)
            if skip_missing_files and not mat_path.exists():
                print(f"[WARN] Missing MAT: {mat_path} (meta_code={meta_code_str}, sel_idx={sel_idx}) -> skipping")
                continue

            hea_path = mat_path.with_suffix(".hea")
            fs = float(infer_fs_from_header(hea_path))

            result = evaluate_explanation(
                mat_path=str(mat_path),
                fs=fs,
                payload=payload,
                class_name=class_name,
                rpeaks_sec=None,
                lead_names=LEADS12,
                precision_k=precision_k,
                model_predict_proba=predict_proba,
                deletion_fractions=deletion_fractions,
                baseline=baseline,
            )

            rows.append({
                "meta_code": meta_code_str,
                "class_name": class_name,
                "sel_idx": int(sel_idx),
                "mat_path": str(mat_path),
                "method": method_label or payload.get("method_label", "unknown"),

                "strict_attauc": result.strict_attauc,
                "lenient_attauc": result.lenient_attauc,

                "precision_k": result.precision_k,
                "strict_p_at_k": result.strict_p_at_k,
                "lenient_p_at_k": result.lenient_p_at_k,

                "deletion_auc": result.deletion_auc,
                "faithfulness_gain": result.faithfulness_gain,

                "n_tokens": result.n_tokens,
            })

    return pd.DataFrame(rows)
