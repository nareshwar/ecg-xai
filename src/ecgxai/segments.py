"""
ecgxai.segments

Create time segments ("events") over an ECG for event-level explanations.

The explainers (LIME/TimeSHAP) operate on a list of segments:
    segments = [(start_sample, end_sample), ...]

Each segment is a half-open interval [start_sample, end_sample) in *samples*.

Supported segmentation modes (params["event_kind"]):
- "uniform": non-overlapping fixed windows of length params["window_sec"]
- "beat_qrs": beat-centered windows around detected R-peaks (pre/post in seconds)
- "qrs_family": two windows per beat approximating QRS and terminal QRS timing

All functions assume x_tf is shaped (T, F):
- T: time samples
- F: leads/channels (typically 12)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

Segment = Tuple[int, int]  # (start_sample, end_sample)


__all__ = [
    "make_uniform_segments",
    "make_event_segments",
    "detect_rpeaks",
    "beat_segments",
    "qrs_family_segments",
]


# ---------------------------------------------------------------------
# Segment builders
# ---------------------------------------------------------------------
def make_uniform_segments(x_tf: np.ndarray, fs: float, window_sec: float) -> Tuple[List[Segment], int]:
    """Split an ECG into non-overlapping fixed-length windows.

    Args:
        x_tf: ECG array (T, F).
        fs: Sampling frequency in Hz.
        window_sec: Window length in seconds.

    Returns:
        segments: List[(start_sample, end_sample)] covering [0, T).
        win_samples: Window length in samples.
    """
    if window_sec <= 0:
        raise ValueError(f"window_sec must be > 0, got {window_sec}")

    T = int(x_tf.shape[0])
    win = max(1, int(round(float(window_sec) * float(fs))))
    segments = _segments(T, win)
    return segments, win


def make_event_segments(
    x_tf: np.ndarray,
    fs: float,
    params: dict,
    lead_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Segment], Optional[int]]:
    """Return segments for event-level explanation.

    Args:
        x_tf: ECG array (T, F).
        fs: Sampling frequency in Hz.
        params: Dict containing "event_kind" and associated parameters.
        lead_names: Optional names aligned with x_tf columns.

    Returns:
        segments: List[(start_sample, end_sample)].
        win_samples: Window size in samples for uniform segmentation, else None.

    Raises:
        ValueError: if params["event_kind"] is unknown or required params are missing.
    """
    kind = str(params.get("event_kind", "uniform"))
    T = int(x_tf.shape[0])

    if kind == "uniform":
        window_sec = float(params["window_sec"])
        win_samp = max(1, int(round(window_sec * float(fs))))
        return _segments(T, win_samp), win_samp

    if kind == "beat_qrs":
        pre = float(params.get("pre_qrs_sec", 0.08))
        post = float(params.get("post_qrs_sec", 0.20))
        rpeaks = detect_rpeaks(
            x_tf, fs,
            prefer_leads=("II", "V1", "V2"),
            lead_names=lead_names,
        )
        return beat_segments(rpeaks, fs, pre=pre, post=post, T=T), None

    if kind == "qrs_family":
        rpeaks = detect_rpeaks(
            x_tf, fs,
            prefer_leads=("II", "V1", "V2"),
            lead_names=lead_names,
        )
        segments = qrs_family_segments(
            rpeaks,
            fs,
            T,
            qrs_pre=float(params.get("qrs_pre_sec", 0.05)),
            qrs_post=float(params.get("qrs_post_sec", 0.08)),
            term_pre=float(params.get("term_pre_sec", 0.08)),
            term_post=float(params.get("term_post_sec", 0.16)),
        )
        return segments, None

    raise ValueError(f"Unknown event_kind: {kind!r}")


def _segments(T: int, win_samp: int) -> List[Segment]:
    """Split [0, T) into non-overlapping segments of size win_samp."""
    if win_samp <= 0:
        raise ValueError(f"win_samp must be > 0, got {win_samp}")
    starts = np.arange(0, int(T), int(win_samp))
    return [(int(s), int(min(T, s + win_samp))) for s in starts]


# ---------------------------------------------------------------------
# R-peak detection (Pan–Tompkins style, simplified)
# ---------------------------------------------------------------------
def detect_rpeaks(
    x_tf: np.ndarray,
    fs: float,
    *,
    prefer_leads: Sequence[str] = ("II", "V2", "V3"),
    lead_names: Optional[Sequence[str]] = None,
    refractory_ms: int = 250,
    min_prom: Optional[float] = None,
) -> np.ndarray:
    """Detect R-peak indices in samples.

    Strategy:
    - choose preferred leads if lead_names provided, else channel 0
    - bandpass filter (5–18 Hz) to isolate QRS energy
    - differentiate → square → moving integration → sum across chosen leads
    - find peaks with refractory period constraint
    - refine each peak to local maximum within ±80ms on filtered signal

    Args:
        x_tf: ECG (T, F)
        fs: Sampling frequency (Hz)
        prefer_leads: Preferred lead names to use for detection
        lead_names: Optional lead names aligned with columns of x_tf
        refractory_ms: Minimum separation between peaks
        min_prom: Optional prominence threshold; if None uses 0.5*std(feature)

    Returns:
        rpeaks: sorted unique peak indices (int array)
    """
    x = np.asarray(x_tf, dtype=np.float32)
    T, F = x.shape

    if lead_names is not None:
        chosen = [lead_names.index(L) for L in prefer_leads if L in lead_names]
        if not chosen:
            chosen = [int(np.argmax(np.std(x, axis=0)))]
    else:
        chosen = [0]

    xb = _bandpass(x[:, chosen], float(fs))
    diff = np.diff(xb, axis=0, prepend=xb[:1])
    sq = diff ** 2
    feat = _moving_integral(sq, float(fs), w_sec=0.15).sum(axis=1)

    med = float(np.median(feat))
    mad = float(np.median(np.abs(feat - med)))
    thr = med + 0.5 * mad

    distance = int(round(float(refractory_ms) / 1000.0 * float(fs)))
    if min_prom is None:
        min_prom = 0.5 * float(np.std(feat))

    peaks, _ = find_peaks(feat, height=thr, distance=distance, prominence=float(min_prom))

    # refine peaks to local maximum on filtered composite
    ref: List[int] = []
    win = int(round(0.08 * float(fs)))
    xb_sum = xb.sum(axis=1)
    for p in peaks:
        s = max(0, int(p) - win)
        e = min(T, int(p) + win + 1)
        ref.append(s + int(np.argmax(xb_sum[s:e])))

    return np.array(sorted(set(ref)), dtype=int)


def beat_segments(
    rpeaks: Sequence[int],
    fs: float,
    *,
    pre: float = 0.20,
    post: float = 0.50,
    T: Optional[int] = None,
) -> List[Segment]:
    """Beat-centered windows around each R-peak.

    Args:
        rpeaks: R-peak indices (samples)
        fs: Sampling frequency (Hz)
        pre: Seconds before R-peak
        post: Seconds after R-peak
        T: Optional signal length in samples to clip end

    Returns:
        List of (start_sample, end_sample) windows.
    """
    pre_s = int(round(float(pre) * float(fs)))
    post_s = int(round(float(post) * float(fs)))

    segs: List[Segment] = []
    for r in rpeaks:
        s = max(0, int(r) - pre_s)
        e = int(r) + post_s
        if T is not None:
            e = min(int(T), e)
        if e > s:
            segs.append((s, e))
    return segs


def qrs_family_segments(
    rpeaks: Sequence[int],
    fs: float,
    T: int,
    *,
    qrs_pre: float = 0.05,
    qrs_post: float = 0.08,
    term_pre: float = 0.08,
    term_post: float = 0.16,
) -> List[Segment]:
    """Build per-beat windows approximating 'qrs' and 'qrs_term' families.

    For each R-peak r:
      - qrs:      [r - qrs_pre,  r + qrs_post]
      - qrs_term: [r + term_pre, r + term_post]

    Args:
        rpeaks: R-peak indices (samples)
        fs: Sampling frequency (Hz)
        T: Signal length (samples)
        qrs_pre/qrs_post: QRS window around R-peak in seconds
        term_pre/term_post: terminal QRS window after R-peak in seconds

    Returns:
        List of segments (start_sample, end_sample).
    """
    segs: List[Segment] = []
    fs = float(fs)

    for r in rpeaks:
        r = int(r)

        # QRS
        s1 = max(0, int(round(r - qrs_pre * fs)))
        e1 = min(int(T), int(round(r + qrs_post * fs)))
        if e1 > s1:
            segs.append((s1, e1))

        # Terminal QRS
        s2 = max(0, int(round(r + term_pre * fs)))
        e2 = min(int(T), int(round(r + term_post * fs)))
        if e2 > s2:
            segs.append((s2, e2))

    return segs


# ---------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------
def _bandpass(
    x: np.ndarray,
    fs: float,
    lo: float = 5.0,
    hi: float = 18.0,
    order: int = 3,
) -> np.ndarray:
    """Bandpass filter for QRS energy extraction."""
    nyq = fs / 2.0
    lo_n = max(1e-6, lo / nyq)
    hi_n = min(0.999999, hi / nyq)
    if hi_n <= lo_n:
        hi_n = min(0.999999, lo_n + 0.05)
    b, a = butter(int(order), [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x, axis=0)


def _moving_integral(sig: np.ndarray, fs: float, w_sec: float = 0.150) -> np.ndarray:
    """Moving-window integration used in the R-peak feature signal."""
    w = max(1, int(round(float(w_sec) * float(fs))))
    ker = np.ones(w, dtype=np.float32) / float(w)
    out = np.empty_like(sig)
    for j in range(sig.shape[1]):
        out[:, j] = np.convolve(sig[:, j], ker, mode="same")
    return out
