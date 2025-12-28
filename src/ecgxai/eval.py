"""
ecgxai.eval — Core ECG explanation evaluation (NO stability / NO augmentation)

This module evaluates explanation payloads against ECG-aware "plausibility" and
(optional) faithfulness metrics.

Provided:
- BeatWindows: HR-aware diagnostic windows around each R-peak.
- REGISTRY: mapping from class name -> "ground truth" lead sets and window types.
- Token-level plausibility:
    * AttAUC (strict / lenient): rank-based alignment between explanation and
      clinically plausible lead×window tokens.
    * Precision@K (strict / lenient): top-K alignment.
- Faithfulness (optional; requires model_predict_proba):
    * Targeted deletion curve over positive tokens vs control deletions.
    * Deletion AUC (lower => more faithful; probability drops faster).
    * Faithfulness gain (drop faster than control on average).

Main entry points:
- evaluate_explanation(): evaluate a single ECG + payload
- evaluate_all_payloads(): batch evaluation over dict/list payload collections

Expected payload format (minimum):
payload["perlead_spans"] = {
    "II": [(start_sec, end_sec, weight), ...],
    "V1": [(start_sec, end_sec, weight), ...],
    ...
}
Weights can be positive/negative; plausibility uses |weight| by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks

from .config import MAXLEN
from .preprocessing import infer_fs_from_header
from .config_targets import TARGET_META


# --------------------------------------------------------------------------- #
# Public exports
# --------------------------------------------------------------------------- #

__all__ = [
    "LEADS12",
    "BeatWindows",
    "ClassConfig",
    "REGISTRY",
    "detect_rpeaks",
    "build_windows_from_rpeaks",
    "integrate_attribution",
    "rank_auc",
    "precision_at_k",
    "token_level_attauc",
    "DeletionCurve",
    "targeted_deletion_curve",
    "deletion_auc_from_curve",
    "faithfulness_gain_from_curve",
    "EvaluationOutput",
    "evaluate_explanation",
    "evaluate_all_payloads",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LEADS12: Tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
)


# --------------------------------------------------------------------------- #
# Basic I/O helpers (generic MAT loading)
# --------------------------------------------------------------------------- #

def load_ecg_mat_tf(mat_path: Union[str, Path]) -> np.ndarray:
    """Load an ECG MAT file as float32 shaped (T, F).

    This loader is intentionally *generic*:
    - Looks for common keys: ('val', 'ECG', 'ecg', 'data', 'signal')
    - Fixes orientation so that time is always axis 0
    - Handles PhysioNet-style arrays shaped (F, T_raw) by transposing

    Args:
        mat_path: Path to a .mat file.

    Returns:
        x: ECG array of shape (T, F), dtype float32.

    Raises:
        ValueError: If no array-like ECG key is found.
    """
    d = loadmat(str(mat_path), squeeze_me=True)

    for key in ("val", "ECG", "ecg", "data", "signal"):
        if key in d:
            arr = np.asarray(d[key])
            break
    else:
        raise ValueError(f"No ECG array found in {mat_path}")

    if arr.ndim == 1:
        arr = arr[None, :]

    # If PhysioNet-style: (n_leads, n_samples) -> transpose
    if arr.ndim == 2 and arr.shape[0] in (8, 12):
        x = arr.T
    else:
        x = arr

    # Force time on axis 0 if it looks swapped
    if x.ndim == 2 and x.shape[0] < x.shape[1]:
        x = x.T

    return x.astype(np.float32, copy=False)


def _ensure_len(x: np.ndarray, maxlen: int = MAXLEN) -> np.ndarray:
    """Pad/truncate ECG to (maxlen, F) with zero-padding at end."""
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
    """Detect R-peaks using a simple Pan–Tompkins–style pipeline.

    Pipeline:
    1) choose 1+ preferred leads (fallback to highest-variance lead)
    2) bandpass filter (QRS band)
    3) differentiate → square → moving integration → sum across chosen leads
    4) find peaks with refractory constraint
    5) refine each detected peak to local maximum within ±80ms window

    Args:
        x: ECG array (T, F).
        fs: Sampling frequency in Hz.
        prefer: Lead names to prefer when lead_names is available.
        lead_names: Lead names aligned to columns of x; if None, uses channel 0.
        refractory_ms: Minimum time between peaks.
        min_prom: Optional peak prominence. If None, uses 0.5*std(feature).

    Returns:
        rpeaks: 1D array of peak indices (samples), sorted and unique.
    """
    x = np.asarray(x, dtype=np.float32)
    T, F = x.shape

    # Choose channels
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
        min_prom = 0.5 * float(np.std(feat))

    peaks, _ = find_peaks(feat, height=thr, distance=distance, prominence=min_prom)

    # Refine to local maxima on the filtered composite
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
    """Beat-aligned windows (all in seconds)."""
    pre: List[Tuple[float, float]]
    qrs: List[Tuple[float, float]]
    qrs_term: List[Tuple[float, float]]
    qrs_on: List[Tuple[float, float]]
    stt: List[Tuple[float, float]]
    tlate: List[Tuple[float, float]]
    pace: List[Tuple[float, float]]
    beat: List[Tuple[float, float]]


def _median_hr_bpm(r_sec: Sequence[float]) -> float:
    """Robust median HR from R-peak times (seconds)."""
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
    """Build HR-aware windows around each R-peak.

    Windows are chosen to be sensible across heart rates (pre window slightly
    shrinks at high HR). Some classes may extend QRS termination.

    Args:
        r_sec: R-peak times in seconds.
        class_name: Optional class label (used for special-case window lengths).

    Returns:
        BeatWindows with lists aligned to r_sec length.
    """
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
    """Defines plausible tokens for a class (strict vs lenient)."""
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
# Overlap + scoring
# --------------------------------------------------------------------------- #

def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return overlap duration between two [start,end] windows in seconds."""
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def integrate_attribution(
    perlead_spans: Dict[str, List[Tuple[float, float, float]]],
    tokens: List[Tuple[str, Tuple[float, float]]],
) -> np.ndarray:
    """Integrate explanation attribution over each lead×window token.

    For a token (lead, [s,e]):
        score = Σ_{spans on that lead} overlap([s,e], [a,b]) * |w|

    Args:
        perlead_spans: {lead: [(start_sec, end_sec, weight), ...]}
        tokens: list of (lead, (start_sec, end_sec))

    Returns:
        scores: float64 array of shape (len(tokens),)
    """
    scores = np.zeros(len(tokens), dtype=np.float64)
    for i, (lead, (s, e)) in enumerate(tokens):
        spans = perlead_spans.get(str(lead), ())
        if not spans:
            continue
        acc = 0.0
        for (a, b, w) in spans:
            ov = _overlap((s, e), (float(a), float(b)))
            if ov > 0:
                acc += ov * abs(float(w))
        scores[i] = acc
    return scores


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Wilcoxon rank-sum AUC: how well scores rank positive labels higher.

    Returns NaN if there are no positives or no negatives.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    P = int(labels.sum())
    N = int(len(labels) - P)
    if P == 0 or N == 0:
        return float("nan")

    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # tie-handling: average ranks for equal scores
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
    """Create tokens for all leads across selected window types."""
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
            tokens.append((str(L), w))
    return tokens


def make_labels_for_tokens(
    tokens: List[Tuple[str, Tuple[float, float]]],
    positives_leads: Sequence[str],
    positive_windows: Iterable[str],
    windows: BeatWindows,
) -> np.ndarray:
    """Assign 1/0 labels to tokens based on lead set and window type set."""
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
    """Precision@K for ranking scores (ties broken stably)."""
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
# Optional priors (blending)
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


def _apply_prior_blend(
    class_name: str,
    target_name: str,
    pref_leads: set,
    pref_windows: set,
    tokens: List[Tuple[str, Tuple[float, float]]],
    windows: BeatWindows,
    scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if class_name != target_name:
        return scores

    scores = np.asarray(scores, float).copy()
    if scores.size and scores.max() > 0:
        scores /= scores.max()

    prior = np.zeros(len(tokens), dtype=float)
    wmap = _win_type_map(windows)
    for i, (lead, (s, e)) in enumerate(tokens):
        if str(lead) in pref_leads and wmap.get((s, e), "") in pref_windows:
            prior[i] = 1.0

    blended = alpha * prior + (1 - alpha) * scores
    if blended.size and blended.max() > 0:
        blended /= blended.max()
    return blended


def apply_sinus_prior_blend(class_name: str, tokens, windows: BeatWindows, scores: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    return _apply_prior_blend(class_name, SINUS_NAME, SINUS_PREF_LEADS, SINUS_PREF_WINDOW_TYPES, tokens, windows, scores, alpha)


def apply_af_prior_blend(class_name: str, tokens, windows: BeatWindows, scores: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    return _apply_prior_blend(class_name, AF_NAME, AF_PREF_LEADS, AF_PREF_WINDOW_TYPES, tokens, windows, scores, alpha)


def apply_vpb_prior_blend(class_name: str, tokens, windows: BeatWindows, scores: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    return _apply_prior_blend(class_name, VPB_NAME, VPB_PREF_LEADS, VPB_PREF_WINDOW_TYPES, tokens, windows, scores, alpha)

def apply_all_priors(
    class_name: str,
    tokens,
    windows: BeatWindows,
    scores: np.ndarray,
    alpha: float = 0.8,
) -> np.ndarray:
    """Apply all built-in diagnosis priors in a consistent order.

    This is a convenience wrapper to avoid repeating the same 3 lines across modules.

    Behaviour is intentionally identical to:
        scores = apply_sinus_prior_blend(...)
        scores = apply_af_prior_blend(...)
        scores = apply_vpb_prior_blend(...)

    Notes:
    - `class_name` here is the *human label* used in eval.py (e.g. "sinus rhythm"),
        not the SNOMED code.
    """
    scores = apply_sinus_prior_blend(class_name, tokens, windows, scores, alpha=alpha)
    scores = apply_af_prior_blend(class_name, tokens, windows, scores, alpha=alpha)
    scores = apply_vpb_prior_blend(class_name, tokens, windows, scores, alpha=alpha)
    return scores


# --------------------------------------------------------------------------- #
# Token-level AttAUC computation
# --------------------------------------------------------------------------- #

@dataclass
class AttAUCResult:
    """Plausibility results for strict/lenient token sets."""
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
    use_priors: bool = True,
    prior_alpha: float = 0.8,
) -> AttAUCResult:
    """Compute token-level plausibility metrics (AttAUC + Precision@K).

    Token construction:
      tokens = {(lead, window) for lead in lead_names, window in class windows}

    Labels:
      - strict labels: leads in cfg.strict_leads AND window types in cfg.window_keys
      - lenient labels: leads in cfg.lenient_leads AND window types in cfg.window_keys

    Scoring:
      score(token) = Σ overlap(token_window, span_window) * |span_weight|

    Returns NaN for AUC/precision when degenerate (no positives or no negatives).

    Args:
        perlead_spans: {lead: [(start_sec, end_sec, weight), ...]}
        class_name: Must exist in REGISTRY.
        r_sec: R-peak times in seconds.
        lead_names: Lead name list for token generation.
        precision_k: K for Precision@K.
        use_priors: If True, blend in simple class priors (optional).
        prior_alpha: Blend factor for priors.

    Returns:
        AttAUCResult
    """
    if class_name not in REGISTRY:
        raise KeyError(f"class_name {class_name!r} not in REGISTRY")

    cfg = REGISTRY[class_name]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name)
    tokens = build_tokens(lead_names, windows, which=cfg.window_keys)

    scores = integrate_attribution(perlead_spans, tokens)

    if use_priors:
        scores = apply_all_priors(class_name, tokens, windows, scores, alpha=prior_alpha)

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
    """Deletion curves for explanation-based vs control deletions."""
    fractions: List[float]
    probs: List[float]
    probs_control: List[float]


def deletion_auc_from_curve(curve: Optional[DeletionCurve]) -> float:
    """Trapezoidal AUC under deletion curve (lower => faster drop => more faithful)."""
    if curve is None:
        return math.nan
    x = np.asarray(curve.fractions, dtype=float)
    y = np.asarray(curve.probs, dtype=float)
    if x.size < 2:
        return math.nan
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def faithfulness_gain_from_curve(curve: Optional[DeletionCurve]) -> float:
    """Mean extra probability drop vs control (higher => explanation beats control)."""
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
    """Apply deletions by replacing values in lead×time windows with baseline."""
    y = x.copy()
    if baseline == "mean":
        base = y.mean(axis=0)
    else:
        base = np.zeros(y.shape[1], dtype=y.dtype)

    name_to_idx = {str(L): i for i, L in enumerate(lead_names)}
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
    """Compute targeted deletion curve for a class.

    Strategy:
    - Build lenient-positive tokens for the class (more forgiving "plausible set").
    - Rank positive tokens by integrated attribution score.
    - For each fraction f:
        delete top tokens until cumulative token duration reaches f × total_positive_duration
      Control:
        delete randomly sampled negative tokens matched by duration.

    Args:
        x: ECG (T,F)
        fs: sampling frequency
        perlead_spans: explanation spans per lead
        class_name: class to evaluate (must exist in REGISTRY)
        r_sec: R-peaks in seconds
        predict_proba: function mapping ECG -> probability for this class
        fractions: fractions of total positive token duration to delete
        baseline: "zero" or "mean"
        rng: random seed for control deletions

    Returns:
        DeletionCurve with aligned fractions, probs, probs_control.
    """
    if class_name not in REGISTRY:
        raise KeyError(f"class_name {class_name!r} not in REGISTRY")

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

        # Pick top-scoring positive tokens until duration is reached
        cum = 0.0
        chosen: List[int] = []
        for i in order:
            chosen.append(int(i))
            cum += float(durations[i])
            if cum >= target_dur:
                break
        deletions_pos = [tokens[i] for i in chosen]

        # Control deletions: random negatives duration-matched
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

        probs_pos.append(float(predict_proba(x_pos)))
        probs_ctl.append(float(predict_proba(x_ctl)))
        frac_list.append(float(frac))

    return DeletionCurve(fractions=frac_list, probs=probs_pos, probs_control=probs_ctl)


# --------------------------------------------------------------------------- #
# Top-level per-ECG evaluation
# --------------------------------------------------------------------------- #

@dataclass
class EvaluationOutput:
    """Combined plausibility + (optional) faithfulness outputs for one ECG."""
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
    mat_path: Union[str, Path],
    fs: float,
    payload: Dict,
    class_name: str,
    rpeaks_sec: Optional[Sequence[float]] = None,
    lead_names: Sequence[str] = LEADS12,
    *,
    precision_k: int = 20,
    use_priors: bool = True,
    prior_alpha: float = 0.8,
    model_predict_proba: Optional[Callable[[np.ndarray], float]] = None,
    deletion_fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3),
    baseline: str = "zero",
    rng: int = 0,
) -> EvaluationOutput:
    """Evaluate one explanation payload on one ECG.

    Steps:
    1) Load ECG (T,F).
    2) Detect R-peaks (if not provided), then build beat-aligned windows.
    3) Compute token-level plausibility: AttAUC + Precision@K.
    4) If model_predict_proba is given, compute targeted deletion faithfulness.

    Args:
        mat_path: Path to ECG MAT file.
        fs: Sampling frequency (Hz).
        payload: Explanation payload with "perlead_spans".
        class_name: Must exist in REGISTRY.
        rpeaks_sec: Optional R-peak times in seconds.
        model_predict_proba: Optional function ECG -> prob (for faithfulness).
    """
    x = load_ecg_mat_tf(mat_path)

    raw_spans = payload.get("perlead_spans", {}) or {}
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
        use_priors=use_priors,
        prior_alpha=prior_alpha,
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
    """Try common keys used in your pipeline to locate the MAT path."""
    for k in ("mat_path", "filename", "path"):
        if k in payload and payload[k]:
            return str(payload[k])
    return None


def evaluate_all_payloads(
    all_payloads: Dict[str, Union[Dict[int, dict], List[dict]]],
    *,
    method_label: Optional[str] = None,
    model=None,
    class_names: Optional[Sequence[str]] = None,
    precision_k: int = 20,
    use_priors: bool = True,
    prior_alpha: float = 0.8,
    baseline: str = "zero",
    deletion_fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3),
    skip_missing_files: bool = True,
) -> pd.DataFrame:
    """Evaluate a collection of payloads across selected targets.

    all_payloads supports:
      - dict[meta_code -> dict[sel_idx -> payload]]
        OR
      - dict[meta_code -> list[payload]]  (list index treated as sel_idx)

    Faithfulness requires `model` and `class_names`, and will be skipped if the
    meta_code isn't in model outputs.

    Returns:
        DataFrame with one row per (meta_code, sel_idx).
    """
    rows: List[Dict[str, object]] = []
    class_names_arr = np.asarray(class_names, dtype=str) if class_names is not None else None

    for meta_code, cases in all_payloads.items():
        meta_code_str = str(meta_code)

        if meta_code_str not in TARGET_META:
            print(f"[WARN] meta_code {meta_code_str} not in TARGET_META, skipping")
            continue

        class_name = str(TARGET_META[meta_code_str]["name"])
        if class_name not in REGISTRY:
            print(f"[WARN] class_name {class_name!r} not in REGISTRY, skipping")
            continue

        # Optional deletion metric function
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
        items: List[Tuple[int, dict]]
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
            fs = float(infer_fs_from_header(str(hea_path)))

            result = evaluate_explanation(
                mat_path=mat_path,
                fs=fs,
                payload=payload,
                class_name=class_name,
                rpeaks_sec=None,
                lead_names=LEADS12,
                precision_k=precision_k,
                use_priors=use_priors,
                prior_alpha=prior_alpha,
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
