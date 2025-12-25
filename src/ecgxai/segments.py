import numpy as np
from typing import Optional, Sequence
from scipy.signal import butter, filtfilt, find_peaks

def make_uniform_segments(x_tf, fs, window_sec):
    """
    Split the ECG into non-overlapping windows of length window_sec.
    Returns (segments, win_samples) where segments is list[(s, t)].
    """
    T, F = x_tf.shape
    win = max(1, int(round(window_sec * fs)))
    starts = np.arange(0, T, win)
    segs = [(int(s), int(min(T, s + win))) for s in starts]
    return segs, win

def make_event_segments(x_tf, fs, params, lead_names=None):
    """
    Return list of (start_sample, end_sample) for event-level explanation.
    """
    kind = params.get("event_kind", "uniform")
    T, F = x_tf.shape

    if kind == "uniform":
        win_samp = max(1, int(round(params["window_sec"] * fs)))
        segments = _segments(T, win_samp)
        return segments, win_samp

    # original beat_qrs mode (keep if you want)
    if kind == "beat_qrs":
        pre  = float(params.get("pre_qrs_sec", 0.08))
        post = float(params.get("post_qrs_sec", 0.20))
        rpeaks = detect_rpeaks(
            x_tf, fs,
            prefer_leads=("II","V1","V2"),
            lead_names=lead_names
        )
        segments = beat_segments(rpeaks, fs, pre=pre, post=post, T=T)
        return segments, None

    # NEW: qrs + qrs_term aligned segments
    if kind == "qrs_family":
        rpeaks = detect_rpeaks(
            x_tf, fs,
            prefer_leads=("II","V1","V2"),
            lead_names=lead_names
        )
        segments = qrs_family_segments(
            rpeaks, fs, T,
            qrs_pre=float(params.get("qrs_pre_sec", 0.05)),
            qrs_post=float(params.get("qrs_post_sec", 0.08)),
            term_pre=float(params.get("term_pre_sec", 0.08)),
            term_post=float(params.get("term_post_sec", 0.16)),
        )
        return segments, None

    raise ValueError(f"Unknown event_kind: {kind}")

def _segments(T, win_samp):
    """
    Split a time axis of length T into non-overlapping segments
    of size win_samp. Returns a list of (start_idx, end_idx).
    """
    starts = np.arange(0, T, win_samp)
    return [(int(s), int(min(T, s + win_samp))) for s in starts]

def detect_rpeaks(x: np.ndarray, fs: float, prefer: Sequence[str] = ("II","V2","V3"),
                  lead_names: Optional[Sequence[str]] = None, refractory_ms: int = 250,
                  min_prom: Optional[float] = None) -> np.ndarray:
    T, F = x.shape
    if lead_names is not None:
        chosen = [lead_names.index(L) for L in prefer if L in lead_names]
        if not chosen:
            chosen = [int(np.argmax(np.std(x, axis=0)))]
    else:
        chosen = [0]
    xb = _bandpass(x[:, chosen], fs)
    diff = np.diff(xb, axis=0, prepend=xb[:1])
    sq = diff**2
    feat = _moving_integral(sq, fs, w_sec=0.15).sum(axis=1)
    thr = np.median(feat) + 0.5*np.median(np.abs(feat - np.median(feat)))
    distance = int(round(refractory_ms/1000 * fs))
    if min_prom is None:
        min_prom = 0.5*np.std(feat)
    peaks, _ = find_peaks(feat, height=thr, distance=distance, prominence=min_prom)
    # refine
    ref, win = [], int(0.08*fs)
    xb_sum = xb.sum(axis=1)
    for p in peaks:
        s = max(0, p-win); e = min(T, p+win+1)
        ref.append(s + int(np.argmax(xb_sum[s:e])))
    return np.array(sorted(set(ref)), dtype=int)

def beat_segments(rpeaks, fs, pre=0.20, post=0.50, T=None):
    # windows around each beat, seconds -> samples
    segs = []
    pre_s = int(pre*fs); post_s = int(post*fs)
    for r in rpeaks:
        s = max(0, r - pre_s)
        e = r + post_s
        if T is not None: e = min(T, e)
        segs.append((s, e))
    return segs

def qrs_family_segments(rpeaks, fs, T,
                        qrs_pre=0.05, qrs_post=0.08,
                        term_pre=0.08, term_post=0.16):
    """
    Build segments that approximate GT window_keys:
      - qrs:     [r - qrs_pre,  r + qrs_post]
      - qrs_term:[r + term_pre, r + term_post]
    Returns list of (start_sample, end_sample).
    """
    segs = []
    for r in rpeaks:
        # QRS window
        s1 = max(0, int(round(r - qrs_pre * fs)))
        e1 = min(T, int(round(r + qrs_post * fs)))
        if e1 > s1:
            segs.append((s1, e1))

        # Terminal QRS window
        s2 = max(0, int(round(r + term_pre * fs)))
        e2 = min(T, int(round(r + term_post * fs)))
        if e2 > s2:
            segs.append((s2, e2))
    return segs

def _bandpass(x: np.ndarray, fs: float, lo: float = 5.0, hi: float = 18.0, order: int = 3) -> np.ndarray:
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x, axis=0)

def _moving_integral(sig: np.ndarray, fs: float, w_sec: float = 0.150) -> np.ndarray:
    w = max(1, int(round(w_sec*fs)))
    ker = np.ones(w, dtype=np.float32) / w
    out = np.empty_like(sig)
    for j in range(sig.shape[1]):
        out[:, j] = np.convolve(sig[:, j], ker, mode="same")
    return out