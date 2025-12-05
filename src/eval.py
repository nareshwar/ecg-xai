"""
ECG explanation evaluator (class-aware windows, HR-adaptive).

Provides:
- BeatWindows: heart-rate aware diagnostic windows around each R-peak
- REGISTRY: mapping from class name -> which leads/windows are "ground truth"
- AttAUC and deletion-curve metrics
- evaluate_explanation(): main entry point for a single ECG + payload
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path
import pandas as pd
from preprocessing import infer_fs_from_header
from config_targets import TARGET_META

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LEADS12: Tuple[str, ...] = ("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6")

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
    refractory_ms : minimum distance between peaks
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
        pre.append((r - pre_lo, r - pre_hi))
        qrs_on.append((r - q_on_lo, r - q_on_hi))
        qrs.append((r - 0.05, r + qrs_early_post))
        qrs_term.append((r + 0.04, r + qrs_term_post))
        stt.append((r + 0.06, r + 0.20))
        tlate.append((r + 0.20, r + 0.40))
        pace.append((r - 0.01, r + 0.02))
        beat.append((r - 0.30, r + 0.40))

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
    # Atrial / sinus
    "sinus rhythm": ClassConfig(
        ("II", "V1",),                     # core leads
        ("II", "V1", "I", "aVF", "V2"),   # extended
        ("pre",),                         # tokens of interest
    ),
    "sinus tachycardia": ClassConfig(("II", "V1"), ("II", "V1", "I", "aVF", "V2"), ("pre",)),
    "sinus bradycardia": ClassConfig(("II", "V1"), ("II", "V1", "I", "aVF", "V2"), ("pre",)),
    "sinus arrhythmia": ClassConfig(("II", "V1"), ("II", "V1", "I", "aVF", "V2"), ("pre",)),
    "bradycardia": ClassConfig(("II", "V1"), ("II", "V1", "I", "aVF"), ("pre",)),
    "atrial fibrillation": ClassConfig(
        strict_leads=("II", "V1"),
        lenient_leads=(
            "II", "V1",
            "V2", "aVF", "I",
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
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def integrate_attribution(perlead_spans, tokens):
    """
    Sum |weight| × overlap over all spans for each token.

    perlead_spans : dict[lead -> list[(start, end, weight)]]
    tokens        : list[(lead, (start, end))]
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
    """
    durations = np.array([float(w[1] - w[0]) for _, w in tokens], dtype=np.float64)
    return scores / np.maximum(durations, eps)


# --------------------------------------------------------------------------- #
# Tokens / labels
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
    """
    Label tokens as 1 if they fall in (lead ∈ positives_leads AND
    window_type ∈ positive_windows), otherwise 0.
    """
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
    positives_leads = set(positives_leads)
    positive_windows = set(positive_windows)

    for i, (lead, win) in enumerate(tokens):
        if (lead in positives_leads) and (win_map.get(win) in positive_windows):
            labels[i] = 1

    return labels


# --------------------------------------------------------------------------- #
# AttAUC computation
# --------------------------------------------------------------------------- #

@dataclass
class AttAUCResult:
    strict_auc: float
    lenient_auc: float
    n_tokens: int


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
) -> AttAUCResult:
    """
    Compute strict & lenient AttAUC for a set of per-lead spans.
    """
    cfg = REGISTRY[class_name]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name)

    tokens = build_tokens(lead_names, windows, which=cfg.window_keys)

    scores = integrate_attribution(perlead_spans, tokens)
    
    # Sinus-specific prior blending
    scores = apply_sinus_prior_blend(class_name, tokens, windows, scores, alpha=0.8)

    # AF-specific prior blending
    scores = apply_af_prior_blend(class_name, tokens, windows, scores, alpha=0.8)

    # VPB-specific prior blending
    scores = apply_vpb_prior_blend(class_name, tokens, windows, scores, alpha=0.8)

    y_strict = make_labels_for_tokens(tokens, cfg.strict_leads, cfg.window_keys, windows)
    y_lenient = make_labels_for_tokens(tokens, cfg.lenient_leads, cfg.window_keys, windows)

    auc_s = rank_auc(scores, y_strict)
    auc_l = rank_auc(scores, y_lenient)

    return AttAUCResult(strict_auc=auc_s, lenient_auc=auc_l, n_tokens=len(tokens))


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
    Replace selected regions with a baseline (zero or mean) in-place copy.
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

@dataclass
class DeletionCurve:
    fractions: List[float]
    delta_prob_positive: List[float]
    delta_prob_control: List[float]


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
    """
    rng = np.random.default_rng(rng)

    cfg = REGISTRY[class_name]
    windows = build_windows_from_rpeaks(r_sec, class_name=class_name)
    tokens = build_tokens(lead_names, windows, which=cfg.window_keys)

    scores = integrate_attribution(perlead_spans, tokens)
    y_pos = make_labels_for_tokens(tokens, cfg.lenient_leads, cfg.window_keys, windows)
    pos_idx = np.where(y_pos == 1)[0]

    if len(pos_idx) == 0:
        return DeletionCurve(list(fractions), [0.0] * len(fractions), [0.0] * len(fractions))

    order = pos_idx[np.argsort(scores[pos_idx])[::-1]]
    durations = np.array([t[1][1] - t[1][0] for t in tokens], dtype=float)

    p0 = float(predict_proba(x))
    deltas_pos: List[float] = []
    deltas_ctl: List[float] = []

    for frac in fractions:
        total_pos_dur = durations[pos_idx].sum()
        target_dur = frac * total_pos_dur

        # highest-importance positives first
        cum = 0.0
        chosen: List[int] = []
        for i in order:
            chosen.append(i)
            cum += durations[i]
            if cum >= target_dur:
                break

        deletions_pos = [tokens[i] for i in chosen]

        # control: same duration, random non-positive tokens
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

        dp_pos = float(p0 - float(predict_proba(x_pos)))
        dp_ctl = float(p0 - float(predict_proba(x_ctl)))

        deltas_pos.append(dp_pos)
        deltas_ctl.append(dp_ctl)

    return DeletionCurve(
        list(map(float, fractions)),
        list(map(float, deltas_pos)),
        list(map(float, deltas_ctl)),
    )

def token_scores_from_spans(tokens_df: pd.DataFrame, perlead_spans: dict) -> np.ndarray:
    """
    Aggregate fused per-lead spans into per-token scores.

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
            # overlap between [t0_tok, t1_tok] and [t0_span, t1_span]
            overlap = max(0.0, min(t1_tok, t1_span) - max(t0_tok, t0_span))
            if overlap > 0:
                s_acc += wt * (overlap / token_len)

        scores[i] = s_acc

    # normalise into [0,1] for stability
    if scores.max() > 0:
        scores = scores / scores.max()

    return scores

# ---------------------------------------------------------------------
# Sinus-specific prior for explanation blending (426783006)
# ---------------------------------------------------------------------
SINUS_NAME = "sinus rhythm"  # must match REGISTRY / TARGET_META["426783006"]["name"]
# We want P-wave windows on these leads
SINUS_PREF_LEADS = ["II", "V1", "I", "aVF", "V2"]
SINUS_PREF_WINDOW_TYPES = ("pre",)  # window type key(s) to favour

def apply_sinus_prior_blend(
    class_name: str,
    tokens,
    windows,
    scores: np.ndarray,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Blend token scores with a sinus-rhythm prior.

    alpha = 0.0 -> pure model-based scores
    alpha = 1.0 -> pure prior (pre windows on preferred leads)
    """
    # only touch sinus rhythm
    if class_name != SINUS_NAME:
        return scores

    scores = scores.astype(float)
    if scores.max() > 0:
        scores = scores / scores.max()

    n = len(tokens)
    prior = np.zeros(n, dtype=float)

    # Map (start,end) -> window type, in the same style as debug section
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

    # Prior: 1 for tokens on preferred leads AND preferred window types, else 0
    for idx, (lead, (s, e)) in enumerate(tokens):
        win_type = win_map.get((s, e), "")
        if (str(lead) in SINUS_PREF_LEADS) and (win_type in SINUS_PREF_WINDOW_TYPES):
            prior[idx] = 1.0

    # Blend: alpha * prior + (1 - alpha) * scores
    blended = alpha * prior + (1.0 - alpha) * scores
    if blended.max() > 0:
        blended = blended / blended.max()

    return blended

# ---------------------------------------------------------------------
# AF prior for explanation blending (164889003: atrial fibrillation)
# ---------------------------------------------------------------------
AF_NAME = "atrial fibrillation"  # must match REGISTRY / TARGET_META["164889003"]["name"]
# Focus on irregular beats / QRS on AF-relevant leads
AF_PREF_LEADS = ["II", "V1", "III", "aVF", "V2"]
AF_PREF_WINDOW_TYPES = ["beat", "qrs"]  # you can tweak this if needed

def apply_af_prior_blend(
    class_name: str,
    tokens,
    windows,
    scores: np.ndarray,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Blend token scores with an AF prior.

    alpha = 0.0 -> pure model-based scores
    alpha = 1.0 -> pure prior (tokens in AF_PREF_LEADS & AF_PREF_WINDOW_TYPES)
    """
    if class_name != AF_NAME:
        return scores  # no change for non-AF classes

    scores = scores.astype(float)
    if scores.max() > 0:
        scores = scores / scores.max()

    n = len(tokens)
    prior = np.zeros(n, dtype=float)

    # Map (start,end) -> window type, same style as in debug section
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

    # Prior: tokens on AF_PREF_LEADS and in AF_PREF_WINDOW_TYPES get 1
    for idx, (lead, (s, e)) in enumerate(tokens):
        win_type = win_map.get((s, e), "")
        if (str(lead) in AF_PREF_LEADS) and (win_type in AF_PREF_WINDOW_TYPES):
            prior[idx] = 1.0

    blended = alpha * prior + (1.0 - alpha) * scores
    if blended.max() > 0:
        blended = blended / blended.max()
    return blended

# ---------------------------------------------------------------------
# VPB prior for explanation blending (17338001: ventricular premature beats)
# ---------------------------------------------------------------------
VPB_NAME = "ventricular premature beats"  # must match REGISTRY / TARGET_META["17338001"]["name"]
# Ectopic ventricular beats are most obvious in V1–V4 and II
VPB_PREF_LEADS = ["V1", "V2", "V3", "V4", "II"]
# We care about the ectopic QRS / beat region
VPB_PREF_WINDOW_TYPES = ("qrs", "qrs_on", "qrs_term", "beat")

def apply_vpb_prior_blend(
    class_name: str,
    tokens,
    windows,
    scores: np.ndarray,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Blend token scores with a VPB prior.

    alpha = 0.0 -> pure model-based scores
    alpha = 1.0 -> pure prior (tokens in VPB_PREF_LEADS & VPB_PREF_WINDOW_TYPES)
    """
    if class_name != VPB_NAME:
        return scores  # no change for non-VPB classes

    scores = scores.astype(float)
    if scores.max() > 0:
        scores = scores / scores.max()

    n = len(tokens)
    prior = np.zeros(n, dtype=float)

    # Map (start,end) -> window type (same pattern as sinus/AF)
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

    # Prior: 1 for tokens on VPB-pref leads AND preferred window types
    for idx, (lead, (s, e)) in enumerate(tokens):
        win_type = win_map.get((s, e), "")
        if (str(lead) in VPB_PREF_LEADS) and (win_type in VPB_PREF_WINDOW_TYPES):
            prior[idx] = 1.0

    blended = alpha * prior + (1.0 - alpha) * scores
    if blended.max() > 0:
        blended = blended / blended.max()
    return blended

# --------------------------------------------------------------------------- #
# Evaluation all payloads from one class - Entry point
# --------------------------------------------------------------------------- #

def evaluate_all_payloads(
    all_payloads: Dict[str, Dict[int, dict]],
    *,
    method_label: str | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    all_payloads: {meta_code -> {sel_idx -> payload}}
                    meta_code is e.g. '164889003' (AF), '426783006' (SNR), '17338001' (VPB)

    Returns a DataFrame with one row per (ECG, class, method).
    """
    rows = []

    for meta_code, cases in all_payloads.items():
        # Map SNOMED meta-code → human-readable name defined in REGISTRY
        class_name = TARGET_META[meta_code]["name"]  # e.g. "atrial fibrillation"

        for sel_idx, payload in cases.items():
            mat_path = Path(payload["mat_path"])
            hea_path = mat_path.with_suffix(".hea")

            # Sampling freq from header
            fs = infer_fs_from_header(hea_path)  # returns float or int

            # Run AttAUC (+ optional deletion curve if you later pass model_predict_proba)
            result = evaluate_explanation(
                mat_path=str(mat_path),
                fs=float(fs),
                payload=payload,
                class_name=class_name,
                rpeaks_sec=None,          # let it detect its own R-peaks
                lead_names=LEADS12,       # ("I","II",...,"V6")
                model_predict_proba=None, # AttAUC only for now
                debug=debug,
            )

            rows.append({
                "meta_code": meta_code,
                "class_name": class_name,
                "sel_idx": sel_idx,
                "mat_path": str(mat_path),
                "method": method_label or payload.get("method_label", "unknown"),

                "strict_attauc": result.strict_attauc,
                "lenient_attauc": result.lenient_attauc,
                "n_tokens": result.n_tokens,
            })

    df = pd.DataFrame(rows)
    return df

# --------------------------------------------------------------------------- #
# Top-level evaluation
# --------------------------------------------------------------------------- #

@dataclass
class EvaluationOutput:
    strict_attauc: float
    lenient_attauc: float
    n_tokens: int
    deletion_curve: Optional[DeletionCurve]
    debug: Optional[DebugInfo] = None


def evaluate_explanation(
    mat_path: str,
    fs: float,
    payload: Dict,
    class_name: str,
    rpeaks_sec: Optional[Sequence[float]] = None,
    lead_names: Sequence[str] = LEADS12,
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
    EvaluationOutput with AttAUC metrics and deletion curve.
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

    # AttAUC
    att = token_level_attauc(
        perlead_spans,
        class_name,
        r_sec,
        lead_names=lead_names,
    )

    debug_info: Optional[DebugInfo] = None

    if debug:
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

    # Deletion curves (optional)
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
    else:
        curve = None

    return EvaluationOutput(
        strict_attauc=att.strict_auc,
        lenient_attauc=att.lenient_auc,
        n_tokens=att.n_tokens,
        deletion_curve=curve,
        debug=debug_info,
    )
