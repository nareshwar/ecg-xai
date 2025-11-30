# ecg_timeshap.py
"""
TimeSHAP-like explanations over ECG segments.

Produces a per-ECG DataFrame similar to LIME:
- event-level SHAP values
- top-K events
- top-5 leads + spans per lead
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd

from utils import weighted_ridge_fit, class_index
from preprocessing import ensure_paths, parse_fs_and_leads, load_mat_TF
from ecg_lime import make_event_segments, prior_probs_from_names

# ---------------------------------------------------------------------
# KernelSHAP utilities
# ---------------------------------------------------------------------

def _sample_masks_stratified(
    rng: np.random.Generator,
    D: int,
    m_total: int,
    feature_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Stratified masks by coalition size (0..D).
    If feature_weights is provided (D,), sampling within each size is weighted.
    """
    sizes = np.arange(0, D + 1)
    per = max(1, m_total // len(sizes))
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
            idx = rng.choice(D, size=k, replace=False, p=probs)
            M[i, idx] = 1.0
        masks.append(M)

    return np.vstack(masks)


def _shap_kernel_weights(M: np.ndarray) -> np.ndarray:
    """
    SHAP kernel: w(z) = (D-1)/(C*(D-C)), large for C in {0, D}.
    """
    D = M.shape[1]
    C = M.sum(axis=1).astype(int)

    w = np.empty(len(C), dtype=np.float64)
    w[(C == 0) | (C == D)] = 1000.0
    mid = (C > 0) & (C < D)
    w[mid] = (D - 1.0) / (C[mid] * (D - C[mid]))
    return w


def _apply_time_masks(
    x_tf: np.ndarray,
    M_time: np.ndarray,
    segments: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """
    Apply time coalitions to x_tf.

    Args:
        x_tf   : (T, F)
        M_time : (m, E)  binary mask over events
        segments: list[(s,t)]
    Returns:
        Xp     : (m, T, F) masked inputs
    """
    T, F = x_tf.shape
    m, E = M_time.shape
    Xp = np.repeat(x_tf[None, ...], m, axis=0)
    for e, (s, t) in enumerate(segments):
        Xp[:, s:t, :] *= M_time[:, e][:, None, None]
    return Xp


def _apply_feature_masks_in_window(
    x_tf: np.ndarray,
    M_feat: np.ndarray,
    seg: Tuple[int, int],
    mode: str = "context",
) -> np.ndarray:
    """
    Apply feature coalitions inside one time window.

    Args:
        x_tf   : (T, F)
        M_feat : (m, F) binary mask over features
        seg    : (s, t) window
        mode   : 'isolated' or 'context'
    Returns:
        Xp     : (m, T, F)
    """
    T, F = x_tf.shape
    s, t = seg
    m = M_feat.shape[0]

    if mode == "isolated":
        Xp = np.zeros((m, T, F), dtype=np.float32)
        for i in range(m):
            Xp[i, s:t, :] = x_tf[s:t, :] * M_feat[i][None, :]
    else:
        Xp = np.repeat(x_tf[None, ...], m, axis=0)
        for i in range(m):
            Xp[i, s:t, :] = x_tf[s:t, :] * M_feat[i][None, :]

    return Xp


# ---------------------------------------------------------------------
# TimeSHAP inside a single window (feature level)
# ---------------------------------------------------------------------

def timeshap_features_in_event(
    model,
    x_tf: np.ndarray,
    class_idx: int,
    seg: Tuple[int, int],
    *,
    m_coalitions: int = 200,
    rng: int = 42,
    mode: str = "context",
    link: str = "identity",
    pred_batch: int = 128,
    lead_prior_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Feature-level TimeSHAP inside a single window.

    Parameters
    ----------
    model : keras.Model
    x_tf  : (T, F) ECG in model space
    class_idx : output index to explain
    seg   : (start_sample, end_sample)
    m_coalitions : number of feature coalitions
    lead_prior_weights : optional (F,) weights to bias coalitions

    Returns
    -------
    shap_f : (F,) feature attributions for this window
    """
    T, F = x_tf.shape
    rng = np.random.default_rng(rng)

    M_feat = _sample_masks_stratified(
        rng, F, m_coalitions,
        feature_weights=lead_prior_weights,
    )

    Xp = _apply_feature_masks_in_window(x_tf, M_feat, seg, mode=mode)

    f_all: List[np.ndarray] = []
    for i in range(0, len(Xp), pred_batch):
        f_all.append(model.predict(Xp[i:i + pred_batch], verbose=0))
    f = np.vstack(f_all)[:, class_idx]

    if link == "logit":
        eps = 1e-6
        f = np.log((f + eps) / (1.0 - f + eps))

    w = _shap_kernel_weights(M_feat)
    coef, _ = weighted_ridge_fit(
        M_feat,
        f,
        sample_weight=w,
        alpha=1e-3,
        fit_intercept=True,
    )
    return coef.astype(np.float32)


# ---------------------------------------------------------------------
# TimeSHAP over events (time axis)
# ---------------------------------------------------------------------

def timeshap_events(
    model,
    x_tf: np.ndarray,
    class_idx: int,
    segments: Sequence[Tuple[int, int]],
    *,
    m_coalitions: int = 200,
    rng: int = 42,
    link: str = "identity",
    pred_batch: int = 128,
) -> np.ndarray:
    """
    Event-level KernelSHAP over provided segments.

    Returns
    -------
    shap_e : (E,) attribution per segment in `segments`
    """
    T, F = x_tf.shape
    E = len(segments)

    rng = np.random.default_rng(rng)
    M_time = _sample_masks_stratified(rng, E, m_coalitions)

    Xp = _apply_time_masks(x_tf, M_time, segments)

    f_all: List[np.ndarray] = []
    for i in range(0, len(Xp), pred_batch):
        f_all.append(model.predict(Xp[i:i + pred_batch], verbose=0))
    f = np.vstack(f_all)[:, class_idx]

    if link == "logit":
        eps = 1e-6
        f = np.log((f + eps) / (1.0 - f + eps))

    w = _shap_kernel_weights(M_time)
    coef, _ = weighted_ridge_fit(
        M_time,
        f,
        sample_weight=w,
        alpha=1e-3,
        fit_intercept=True,
    )
    return coef.astype(np.float32)


# ---------------------------------------------------------------------
# Driver: TimeSHAP for all ECGs of one class (using sel_df)
# ---------------------------------------------------------------------

def run_timeshap_for_one_class_from_sel(
    sel_df: pd.DataFrame,
    class_name: str,
    *,
    model,
    class_names: Sequence[str],
    window_sec: float = 0.5,
    m_event: int = 200,
    m_feat: int = 200,
    topk_events: int = 6,
    explain_class: str = "force",   # 'force' or 'pred'
    mode: str = "context",
    rng: int = 42,
    link: str = "identity",
    params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Run TimeSHAP explanations for all rows in sel_df with group_class == class_name.

    Returns
    -------
    df_ts : DataFrame with one row per ECG:
      - timeshap_event_values_json
      - segments_json
      - top_events_idx_json
      - top5_lead_idx_json
      - perlead_timeshap_top5_json
      - plus metadata
    """
    rows = sel_df[sel_df["group_class"] == class_name]
    if rows.empty:
        raise ValueError(f"No rows in sel_df with group_class == {class_name!r}")

    # default params if none given
    if params is None:
        params = {
            "event_kind": "uniform",
            "window_sec": float(window_sec),
            "lead_prior": None,
        }

    c_force = class_index(class_names, class_name) if explain_class == "force" else None

    out: List[Dict] = []

    for _, r in rows.iterrows():
        val_idx = int(r.get("sel_idx", r.name))

        hea_path, mat_path = ensure_paths(r["filename"])
        fs, lead_names = parse_fs_and_leads(hea_path, default_fs=500.0)

        lead_prior_probs = prior_probs_from_names(
            lead_names,
            params.get("lead_prior") if params else None,
        )
        lead_prior_weights = lead_prior_probs

        x_tf = load_mat_TF(mat_path)  # (T, F)
        T, F = x_tf.shape

        # which class index to explain?
        if explain_class == "force":
            c = c_force
        elif explain_class == "pred":
            probs = model.predict(x_tf[None, ...], verbose=0)[0]
            c = int(np.argmax(probs))
        else:
            raise ValueError("explain_class must be 'force' or 'pred'")

        # build segments (time windows)
        segments, win_samp = make_event_segments(
            x_tf,
            fs,
            params=params,
            lead_names=lead_names,
        )

        shap_e = timeshap_events(
            model,
            x_tf,
            c,
            segments=segments,
            m_coalitions=m_event,
            rng=rng,
            link=link,
        )

        # choose top-K events by |SHAP|
        topE = np.argsort(np.abs(shap_e))[::-1][:min(topk_events, len(segments))]

        # refine inside each event to get per-lead contributions
        perlead_spans: Dict[int, List[Tuple[float, float, float]]] = {
            j: [] for j in range(F)
        }

        for e_idx in topE:
            s, t = segments[e_idx]
            shap_f = timeshap_features_in_event(
                model,
                x_tf,
                c,
                seg=(s, t),
                m_coalitions=m_feat,
                rng=rng,
                mode=mode,
                link=link,
                lead_prior_weights=lead_prior_weights,
            )
            for j in range(F):
                s_sec = s / fs
                t_sec = t / fs
                perlead_spans[j].append(
                    (float(s_sec), float(t_sec), float(shap_f[j]))
                )

        # aggregate lead-level scores
        lead_scores = {
            j: float(sum(abs(w) for (_, _, w) in spans))
            for j, spans in perlead_spans.items()
        }

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
                "window_sec": float(window_sec),
                "target_class_explained": int(c),
                # event-level
                "timeshap_event_values_json": json.dumps(
                    [float(v) for v in shap_e]
                ),
                "segments_json": json.dumps(
                    [[int(s), int(t)] for (s, t) in segments]
                ),
                "window_size_samples": int(win_samp)
                if win_samp is not None
                else None,
                "num_segments": int(len(segments)),
                "top_events_idx_json": json.dumps(
                    [int(x) for x in list(topE)]
                ),
                # lead-level (top-5)
                "top5_lead_idx_json": json.dumps([int(j) for j in top5]),
                "perlead_timeshap_top5_json": json.dumps(
                    {
                        int(j): [
                            (float(s), float(t), float(w))
                            for (s, t, w) in spans_top5[j]
                        ]
                        for j in spans_top5
                    }
                ),
                "lead_names": ",".join(lead_names)
                if lead_names is not None
                else None,
                "link": link,
                "mode": mode,
            }
        )

    df_ts = (
        pd.DataFrame(out)
        .sort_values(["group_class", "val_idx"])
        .reset_index(drop=True)
    )
    return df_ts