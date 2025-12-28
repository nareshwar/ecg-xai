"""
ecgxai.ecg_timeshap

TimeSHAP-like explanations for ECG classification models using a KernelSHAP-style
weighted linear surrogate on binary coalition masks.

Outputs a per-ECG DataFrame broadly aligned with the LIME driver:
- event-level attributions over time segments ("events")
- top-K events by |attribution|
- per-lead attributions inside each top event (stored as spans per lead)

Shapes / conventions:
- x_tf: np.ndarray of shape (T, F) where F is number of leads (typically 12).
- segments: list of (start_sample, end_sample) indices into T.
- Event coalitions: M_time shape (m, E) where E=len(segments)
- Feature coalitions: M_feat shape (m, F)

Link function:
- link="identity": uses probabilities directly
- link="logit": converts p -> log(p/(1-p)) before fitting the surrogate
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import json

import numpy as np
import pandas as pd

from .utils import weighted_ridge_fit, class_index
from .preprocessing import ensure_paths, parse_fs_and_leads, load_mat_TF
from .ecg_lime import make_event_segments, prior_probs_from_names

# -----------------------------
# Type aliases
# -----------------------------
Segment = Tuple[int, int]                # (start_sample, end_sample)
SpanSec = Tuple[float, float, float]     # (start_sec, end_sec, weight)
RngLike = Union[int, np.random.Generator]


def _as_rng(rng: RngLike) -> np.random.Generator:
    """Return a NumPy Generator from either a seed int or an existing Generator."""
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(int(rng))


# ---------------------------------------------------------------------
# KernelSHAP utilities
# ---------------------------------------------------------------------
def _sample_masks_stratified(
    rng: np.random.Generator,
    D: int,
    m_total: int,
    feature_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample binary coalition masks stratified by coalition size.

    We sample across sizes k=0..D. This reduces variance vs sampling sizes freely.

    Note: returns approximately m_total masks (exact count depends on D).

    Args:
        rng: NumPy Generator.
        D: Number of interpretable features.
        m_total: Approx desired number of masks.
        feature_weights: Optional (D,) weights to bias which features are included.

    Returns:
        masks: (m, D) float32 array in {0,1}.
    """
    sizes = np.arange(0, D + 1)
    per = max(1, m_total // len(sizes))  # ~equal per coalition size
    masks: List[np.ndarray] = []

    probs = None
    if feature_weights is not None:
        fw = np.asarray(feature_weights, dtype=np.float64)
        fw = np.clip(fw, 1e-9, None)
        probs = fw / fw.sum()

    for k in sizes:
        if k == 0:
            masks.append(np.zeros((1, D), dtype=np.float32))
            continue
        if k == D:
            masks.append(np.ones((1, D), dtype=np.float32))
            continue

        M = np.zeros((per, D), dtype=np.float32)
        for i in range(per):
            idx = rng.choice(D, size=int(k), replace=False, p=probs)
            M[i, idx] = 1.0
        masks.append(M)

    return np.vstack(masks).astype(np.float32, copy=False)


def _shap_kernel_weights(M: np.ndarray) -> np.ndarray:
    """KernelSHAP weights for coalition masks.

    For coalition size C:
        w(z) = (D-1) / (C * (D - C))   for 0 < C < D
    We assign large weights to C in {0, D} to anchor the surrogate.

    Args:
        M: (m, D) binary mask matrix.

    Returns:
        w: (m,) float64 weights.
    """
    D = M.shape[1]
    C = M.sum(axis=1).astype(int)

    w = np.empty(len(C), dtype=np.float64)
    w[(C == 0) | (C == D)] = 1000.0
    mid = (C > 0) & (C < D)
    w[mid] = (D - 1.0) / (C[mid] * (D - C[mid]))
    return w


def _apply_time_masks(x_tf: np.ndarray, M_time: np.ndarray, segments: Sequence[Segment]) -> np.ndarray:
    """Apply event masks over time segments.

    Args:
        x_tf: (T, F)
        M_time: (m, E) mask over events.
        segments: list[(s, t)] per event.

    Returns:
        Xp: (m, T, F) masked inputs.
    """
    x_tf = np.asarray(x_tf, dtype=np.float32)
    m = M_time.shape[0]
    Xp = np.repeat(x_tf[None, ...], m, axis=0)
    for e, (s, t) in enumerate(segments):
        Xp[:, s:t, :] *= M_time[:, e][:, None, None]
    return Xp


def _apply_feature_masks_in_window(
    x_tf: np.ndarray, M_feat: np.ndarray, seg: Segment, *, mode: str = "context"
) -> np.ndarray:
    """Apply per-lead masks inside a single time window.

    Args:
        x_tf: (T, F)
        M_feat: (m, F) mask over leads.
        seg: (s, t)
        mode:
            - "context": keep rest of signal intact, perturb only inside [s:t]
            - "isolated": zero outside [s:t], only keep masked region

    Returns:
        Xp: (m, T, F)
    """
    x_tf = np.asarray(x_tf, dtype=np.float32)
    T, F = x_tf.shape
    s, t = seg
    m = M_feat.shape[0]

    if mode == "isolated":
        Xp = np.zeros((m, T, F), dtype=np.float32)
        for i in range(m):
            Xp[i, s:t, :] = x_tf[s:t, :] * M_feat[i][None, :]
    elif mode == "context":
        Xp = np.repeat(x_tf[None, ...], m, axis=0)
        for i in range(m):
            Xp[i, s:t, :] = x_tf[s:t, :] * M_feat[i][None, :]
    else:
        raise ValueError("mode must be 'context' or 'isolated'")

    return Xp


def _apply_link(f: np.ndarray, link: str) -> np.ndarray:
    """Apply link function to model outputs."""
    f = np.asarray(f, dtype=np.float64)
    if link == "identity":
        return f
    if link == "logit":
        eps = 1e-6
        return np.log((f + eps) / (1.0 - f + eps))
    raise ValueError("link must be 'identity' or 'logit'")


# ---------------------------------------------------------------------
# TimeSHAP inside a single window (feature level)
# ---------------------------------------------------------------------
def timeshap_features_in_event(
    model: Any,
    x_tf: np.ndarray,
    class_idx: int,
    seg: Segment,
    *,
    m_coalitions: int = 200,
    rng: RngLike = 42,
    mode: str = "context",
    link: str = "identity",
    pred_batch: int = 128,
    lead_prior_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Feature-level KernelSHAP within one time window (lead attributions).

    Returns:
        shap_f: (F,) float32 per-lead attributions for this window.
    """
    rngg = _as_rng(rng)
    T, F = x_tf.shape

    M_feat = _sample_masks_stratified(rngg, F, m_coalitions, feature_weights=lead_prior_weights)
    Xp = _apply_feature_masks_in_window(x_tf, M_feat, seg, mode=mode)

    # batched prediction for speed
    preds: List[np.ndarray] = []
    for i in range(0, len(Xp), pred_batch):
        preds.append(model.predict(Xp[i : i + pred_batch], verbose=0))
    f = np.vstack(preds)[:, class_idx]
    f = _apply_link(f, link)

    w = _shap_kernel_weights(M_feat)
    coef, _ = weighted_ridge_fit(M_feat, f, sample_weight=w, alpha=1e-3, fit_intercept=True)
    return coef.astype(np.float32)


# ---------------------------------------------------------------------
# TimeSHAP over events (time axis)
# ---------------------------------------------------------------------
def timeshap_events(
    model: Any,
    x_tf: np.ndarray,
    class_idx: int,
    segments: Sequence[Segment],
    *,
    m_coalitions: int = 200,
    rng: RngLike = 42,
    link: str = "identity",
    pred_batch: int = 128,
) -> np.ndarray:
    """Event-level KernelSHAP over provided time segments.

    Returns:
        shap_e: (E,) float32 per-event attributions aligned with `segments`.
    """
    rngg = _as_rng(rng)
    E = len(segments)

    M_time = _sample_masks_stratified(rngg, E, m_coalitions)
    Xp = _apply_time_masks(x_tf, M_time, segments)

    preds: List[np.ndarray] = []
    for i in range(0, len(Xp), pred_batch):
        preds.append(model.predict(Xp[i : i + pred_batch], verbose=0))
    f = np.vstack(preds)[:, class_idx]
    f = _apply_link(f, link)

    w = _shap_kernel_weights(M_time)
    coef, _ = weighted_ridge_fit(M_time, f, sample_weight=w, alpha=1e-3, fit_intercept=True)
    return coef.astype(np.float32)


# ---------------------------------------------------------------------
# Driver: TimeSHAP for all ECGs of one class (using sel_df)
# ---------------------------------------------------------------------
def run_timeshap_for_one_class_from_sel(
    sel_df: pd.DataFrame,
    class_name: str,
    *,
    model: Any,
    class_names: Sequence[str],
    window_sec: float = 0.5,
    m_event: int = 200,
    m_feat: int = 200,
    topk_events: int = 6,
    explain_class: str = "force",  # 'force' or 'pred'
    mode: str = "context",
    rng: RngLike = 42,
    link: str = "identity",
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run TimeSHAP explanations for rows in sel_df where group_class == class_name.

    Required sel_df columns:
        - "group_class"
        - "filename"
        - optional "sel_idx" (used as stable ordering key)

    Returns:
        DataFrame with one row per ECG containing:
            - timeshap_event_values_json
            - segments_json
            - top_events_idx_json
            - top5_lead_idx_json
            - perlead_timeshap_top5_json
        plus metadata.
    """
    required = {"group_class", "filename"}
    missing = required - set(sel_df.columns)
    if missing:
        raise ValueError(f"sel_df missing required columns: {sorted(missing)}")

    rows = sel_df[sel_df["group_class"] == class_name]
    if rows.empty:
        raise ValueError(f"No rows in sel_df with group_class == {class_name!r}")

    # normalise params
    if params is None:
        params = {"event_kind": "uniform", "window_sec": float(window_sec), "lead_prior": None}
    else:
        params = dict(params)
        params.setdefault("event_kind", "uniform")
        params.setdefault("window_sec", float(window_sec))
        params.setdefault("lead_prior", None)

    window_sec_eff = float(params.get("window_sec", window_sec))

    c_force = class_index(class_names, class_name) if explain_class == "force" else None
    if explain_class == "force" and c_force is None:
        raise ValueError(f"class_name {class_name!r} not found in class_names")

    out: List[Dict[str, Any]] = []
    base_rng = _as_rng(rng)

    for _, r in rows.iterrows():
        val_idx = int(r.get("sel_idx", r.name))
        hea_path, mat_path = ensure_paths(r["filename"])
        fs, lead_names = parse_fs_and_leads(hea_path, default_fs=500.0)

        # per-lead priors (optional) -> probabilities/weights
        lead_prior_probs = prior_probs_from_names(lead_names, params.get("lead_prior"))
        lead_prior_weights = lead_prior_probs  # already (F,) if not None

        x_tf = load_mat_TF(mat_path)  # (T, F)
        T, F = x_tf.shape

        # choose class index to explain
        if explain_class == "force":
            c = int(c_force)  # safe due to check above
        elif explain_class == "pred":
            probs = model.predict(x_tf[None, ...], verbose=0)[0]
            c = int(np.argmax(probs))
        else:
            raise ValueError("explain_class must be 'force' or 'pred'")

        # build segments
        segments, win_samp = make_event_segments(x_tf, fs, params=params, lead_names=lead_names)

        # RNG: derive per-record seeds so outputs are stable but not identical everywhere
        rec_seed = int(base_rng.integers(0, 2**31 - 1))
        rng_events = np.random.default_rng(rec_seed)

        shap_e = timeshap_events(
            model,
            x_tf,
            c,
            segments=segments,
            m_coalitions=m_event,
            rng=rng_events,
            link=link,
        )

        topE = np.argsort(np.abs(shap_e))[::-1][: min(topk_events, len(segments))]

        perlead_spans: Dict[int, List[SpanSec]] = {j: [] for j in range(F)}

        for e_idx in topE:
            s, t = segments[int(e_idx)]

            # derive a different RNG for each event so coalitions differ across windows
            ev_seed = int(rng_events.integers(0, 2**31 - 1))
            rng_feat = np.random.default_rng(ev_seed)

            shap_f = timeshap_features_in_event(
                model,
                x_tf,
                c,
                seg=(s, t),
                m_coalitions=m_feat,
                rng=rng_feat,
                mode=mode,
                link=link,
                lead_prior_weights=lead_prior_weights,
            )

            for j in range(F):
                perlead_spans[j].append((float(s / fs), float(t / fs), float(shap_f[j])))

        lead_scores = {j: float(sum(abs(w) for (_, _, w) in spans)) for j, spans in perlead_spans.items()}
        top5 = sorted(lead_scores, key=lead_scores.get, reverse=True)[:5]
        spans_top5 = {int(j): perlead_spans[j] for j in top5}

        out.append(
            {
                "group_class": class_name,
                "val_idx": int(val_idx),
                "filename": r["filename"],
                "hea_path": hea_path,
                "mat_path": mat_path,
                "fs": float(fs),
                "window_sec": float(window_sec_eff),
                "target_class_explained": int(c),
                "timeshap_event_values_json": json.dumps([float(v) for v in shap_e]),
                "segments_json": json.dumps([[int(s), int(t)] for (s, t) in segments]),
                "window_size_samples": int(win_samp) if win_samp is not None else None,
                "num_segments": int(len(segments)),
                "top_events_idx_json": json.dumps([int(x) for x in list(topE)]),
                "top5_lead_idx_json": json.dumps([int(j) for j in top5]),
                "perlead_timeshap_top5_json": json.dumps(
                    {int(j): [(float(s), float(t), float(w)) for (s, t, w) in spans_top5[j]] for j in spans_top5}
                ),
                "lead_names": ",".join(lead_names) if lead_names is not None else None,
                "link": link,
                "mode": mode,
            }
        )

    return pd.DataFrame(out).sort_values(["group_class", "val_idx"]).reset_index(drop=True)
