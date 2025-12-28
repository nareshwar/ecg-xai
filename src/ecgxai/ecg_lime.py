"""
ecgxai.ecg_lime

LIME-style explanations for ECG classifier predictions.

This module implements a two-level explanation strategy:

1) Event-level LIME:
   - Split a record into time windows ("events"/segments).
   - Randomly mask/unmask events and fit a local linear surrogate model.
   - Output: one importance weight per event.

2) Feature-level LIME within top events:
   - For the most important time windows, mask/unmask leads (features) inside
     that window and fit a local surrogate.
   - Output: one weight per lead for each top event, stored as per-lead spans.

Terminology & shapes:
- ECG input is expected as x_tf with shape (T, F):
    T = number of time samples, F = number of leads/features (typically 12).
- segments is a list of (start_idx, end_idx) in *samples* (not seconds).
- A "span" stored in outputs is (start_sec, end_sec, weight).

Lead priors:
- Optional per-lead probabilities can bias mask sampling (e.g., for AF prioritize II, V1).
- If no prior is provided, each lead is masked independently with P(on)=0.5.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances

from .preprocessing import ensure_paths, parse_fs_and_leads, load_mat_TF
from .utils import cosine_distance_to_ones, weighted_ridge_fit, class_index


# -----------------------------
# Type aliases
# -----------------------------
Segment = Tuple[int, int]               # (start_sample, end_sample)
SpanSec = Tuple[float, float, float]    # (start_sec, end_sec, weight)


def _as_rng(rng: Union[int, np.random.Generator]) -> np.random.Generator:
    """Return a NumPy Generator from either a seed (int) or a Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


def _kernel_weights_binary_masks(M: np.ndarray, kernel_width: float) -> np.ndarray:
    """Compute LIME kernel weights for binary masks relative to the all-ones mask.

    Args:
        M: Binary mask matrix of shape (N, D) where D is number of interpretable features.
        kernel_width: LIME kernel width hyperparameter.

    Returns:
        weights of shape (N,).
    """
    # cosine distance to the all-ones mask (classic LIME choice)
    d = cosine_distance_to_ones(M.astype(np.float32))
    w = np.sqrt(np.exp(-(d ** 2) / (kernel_width ** 2)))
    return w.astype(np.float32)


def lime_events(
    model: Any,
    x_tf: np.ndarray,
    class_idx: int,
    segments: Sequence[Segment],
    *,
    m_masks: int = 200,
    kernel_width: float = 0.25,
    rng: Union[int, np.random.Generator] = 42,
) -> np.ndarray:
    """Event-level LIME over time segments.

    For each segment, we sample binary masks indicating whether that segment is
    "kept" or "zeroed" in the perturbed input. We then fit a weighted ridge
    regression surrogate to approximate the model's local behaviour.

    Args:
        model: Any object with `predict(X, verbose=0)` returning shape (N, C).
        x_tf: ECG array of shape (T, F).
        class_idx: Index of the class probability to explain.
        segments: List of (start_sample, end_sample) segments.
        m_masks: Number of perturbation samples (including baseline).
        kernel_width: LIME kernel width.
        rng: Random seed or Generator.

    Returns:
        Coefficient vector of shape (E,), where E = len(segments).
        Larger magnitude => more influential segment.
    """
    T, F = x_tf.shape
    E = len(segments)

    rngg = _as_rng(rng)

    # Binary masks over interpretable features (segments)
    M = rngg.binomial(1, 0.5, size=(m_masks, E)).astype(np.float32)
    M[0, :] = 1.0  # baseline = all-on

    # Build perturbed inputs: Xp has shape (m_masks, T, F)
    Xp = np.repeat(x_tf[None, ...], m_masks, axis=0)
    for e, (s, t) in enumerate(segments):
        Xp[:, s:t, :] *= M[:, e][:, None, None]

    y = model.predict(Xp, verbose=0)[:, class_idx].astype(np.float32)

    w = _kernel_weights_binary_masks(M, kernel_width)

    coef, _ = weighted_ridge_fit(M, y, sample_weight=w, alpha=1e-2, fit_intercept=True)
    return coef.astype(np.float32)


def lime_features_in_event(
    model: Any,
    x_tf: np.ndarray,
    class_idx: int,
    seg: Segment,
    *,
    m_masks: int = 200,
    kernel_width: float = 0.25,
    mode: str = "context",
    rng: Union[int, np.random.Generator] = 42,
    lead_prior_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Feature-level (lead-level) LIME inside a single time window.

    We mask/unmask leads (features) inside the specified time window and fit a
    local weighted linear surrogate.

    Args:
        model: Any object with `predict(X, verbose=0)` returning shape (N, C).
        x_tf: ECG array of shape (T, F).
        class_idx: Index of the class probability to explain.
        seg: (start_sample, end_sample) window in samples.
        m_masks: Number of perturbation samples.
        kernel_width: LIME kernel width.
        mode:
            - "context": keep the rest of the signal intact; only perturb seg region.
            - "isolated": zero the entire signal except seg region.
        rng: Random seed or Generator.
        lead_prior_weights: Optional array of shape (F,) with probabilities in [0,1].
            If provided, each lead j is kept with P(mask_j = lead_prior_weights[j]).
            If None, each lead is kept with probability 0.5.

    Returns:
        Coefficient vector of shape (F,), one weight per lead.
    """
    T, F = x_tf.shape
    s, t = seg
    rngg = _as_rng(rng)

    # --- sample feature masks (N, F) -----------------------------------
    if lead_prior_weights is not None:
        p = np.asarray(lead_prior_weights, dtype=np.float64)
        if p.shape != (F,):
            raise ValueError(f"lead_prior_weights shape {p.shape} != ({F},)")
        p = np.clip(p, 1e-3, 1.0 - 1e-3)
        M = rngg.binomial(1, p, size=(m_masks, F)).astype(np.float32)
    else:
        M = rngg.binomial(1, 0.5, size=(m_masks, F)).astype(np.float32)

    M[0, :] = 1.0  # baseline all-on

    # --- build perturbed inputs ----------------------------------------
    if mode == "isolated":
        Xp = np.zeros((m_masks, T, F), dtype=np.float32)
        for i in range(m_masks):
            Xp[i, s:t, :] = x_tf[s:t, :] * M[i][None, :]
    elif mode == "context":
        Xp = np.repeat(x_tf[None, ...], m_masks, axis=0)
        for i in range(m_masks):
            Xp[i, s:t, :] = x_tf[s:t, :] * M[i][None, :]
    else:
        raise ValueError("mode must be 'context' or 'isolated'")

    y = model.predict(Xp, verbose=0)[:, class_idx].astype(np.float32)

    # kernel weights relative to all-ones mask (classic LIME)
    d = pairwise_distances(M, np.ones((1, F), dtype=np.float32), metric="cosine").ravel()
    w = np.sqrt(np.exp(-(d ** 2) / (kernel_width ** 2))).astype(np.float32)

    reg = Ridge(alpha=1e-2, solver="svd")
    reg.fit(M, y, sample_weight=w)
    return reg.coef_.astype(np.float32)


def run_lime_for_one_class_from_sel(
    sel_df: pd.DataFrame,
    class_name: str,
    *,
    model: Any,
    class_names: Sequence[str],
    window_sec: float = 0.5,
    m_event: int = 200,
    m_feat: int = 200,
    topk_events: int = 5,
    explain_class: str = "force",
    mode: str = "context",
    rng: Union[int, np.random.Generator] = 42,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run LIME explanations for rows in sel_df where group_class == class_name.

    Required sel_df columns:
        - "group_class": label/grouping used to filter rows (e.g., SNOMED code)
        - "filename": path to .mat or .hea/.mat pair
        - optional "sel_idx": stable index (used for sorting)

    The output DataFrame contains JSON-serialized arrays for portability.

    Args:
        sel_df: Selection DataFrame of records to explain.
        class_name: Value of sel_df["group_class"] to filter on.
        model: Trained classifier with `predict`.
        class_names: Class label list aligned to model outputs.
        window_sec: Default segment length for uniform segmentation.
        m_event: # perturbations for event-level LIME.
        m_feat: # perturbations for feature-level LIME.
        topk_events: Number of top events to drill down into (by |importance|).
        explain_class:
            - "force": always explain the class corresponding to class_name
            - "pred": explain the model's argmax class for each record
        mode: Passed to lime_features_in_event ("context" or "isolated").
        rng: Random seed or Generator.
        params: Optional dict for shared configuration with other explainers.
            Recognised keys:
                - event_kind: currently only "uniform"
                - window_sec: overrides window_sec
                - lead_prior: dict mapping lead_name -> weight in [0,1]

    Returns:
        DataFrame with per-record explanation artifacts.
    """
    # Validate required columns early
    required = {"group_class", "filename"}
    missing = required - set(sel_df.columns)
    if missing:
        raise ValueError(f"sel_df missing required columns: {sorted(missing)}")

    # Normalise params so TimeSHAP and LIME can share it
    if params is None:
        params = {"event_kind": "uniform", "window_sec": float(window_sec), "lead_prior": None}
    else:
        params = dict(params)
        params.setdefault("event_kind", "uniform")
        params.setdefault("window_sec", float(window_sec))
        params.setdefault("lead_prior", None)

    window_sec_eff = float(params.get("window_sec", window_sec))

    rows = sel_df[sel_df["group_class"] == class_name]
    if rows.empty:
        raise ValueError(f"No rows in sel_df with group_class == {class_name!r}")

    c_force = class_index(class_names, class_name) if explain_class == "force" else None
    out: List[Dict[str, Any]] = []

    for _, r in rows.iterrows():
        val_idx = int(r.get("sel_idx", r.name))

        hea_path, mat_path = ensure_paths(r["filename"])
        fs, lead_names = parse_fs_and_leads(hea_path, default_fs=500.0)

        # Optional per-lead sampling priors for feature-level masking
        lead_prior_probs = prior_probs_from_names(
            lead_names,
            params.get("lead_prior") if params else None,
        )

        x_tf = load_mat_TF(mat_path)
        T, F = x_tf.shape

        if explain_class == "force":
            c = int(c_force)  # type: ignore[arg-type]
        elif explain_class == "pred":
            probs = model.predict(x_tf[None, ...], verbose=0)[0]
            c = int(np.argmax(probs))
        else:
            raise ValueError("explain_class must be 'force' or 'pred'")

        segments, win_samp = make_event_segments(
            x_tf,
            fs,
            params=params,
            lead_names=lead_names,
        )

        event_scores = lime_events(
            model,
            x_tf,
            c,
            segments=segments,
            m_masks=m_event,
            rng=rng,
        )

        topE = np.argsort(np.abs(event_scores))[::-1][: min(topk_events, len(segments))]

        # Collect spans per lead, but only keep top-5 leads at the end
        perlead_spans: Dict[int, List[SpanSec]] = {j: [] for j in range(F)}

        for e_idx in topE:
            s, t = segments[int(e_idx)]
            coef_f = lime_features_in_event(
                model,
                x_tf,
                c,
                seg=(s, t),
                m_masks=m_feat,
                mode=mode,
                rng=rng,
                lead_prior_weights=lead_prior_probs,
            )
            for j in range(F):
                perlead_spans[j].append((float(s / fs), float(t / fs), float(coef_f[j])))

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
                "event_importances_json": json.dumps([float(v) for v in event_scores]),
                "segments_json": json.dumps([[int(s), int(t)] for (s, t) in segments]),
                "window_size_samples": int(win_samp),
                "num_segments": int(len(segments)),
                "top_events_idx_json": json.dumps([int(x) for x in list(topE)]),
                "top5_lead_idx_json": json.dumps([int(j) for j in top5]),
                "perlead_spans_top5_json": json.dumps(
                    {int(j): [(float(s), float(t), float(w)) for (s, t, w) in spans_top5[j]] for j in spans_top5}
                ),
                "lead_names": ",".join(lead_names) if lead_names is not None else None,
            }
        )

    df_lime = pd.DataFrame(out).sort_values(["group_class", "val_idx"]).reset_index(drop=True)
    return df_lime


def prior_probs_from_names(
    lead_names: Optional[Sequence[str]],
    lead_prior_dict: Optional[Mapping[str, float]],
    *,
    p_default: float = 0.02,
    p_lo: float = 0.1,
    p_hi: float = 0.99,
) -> Optional[np.ndarray]:
    """Map a dict of lead priors into per-lead probabilities.

    Args:
        lead_names: List of lead names (e.g. ["I","II","V1",...]).
        lead_prior_dict: Mapping lead_name -> weight in [0,1].
            If None, returns None (caller should ignore priors).
        p_default: Probability used for leads NOT present in lead_prior_dict.
        p_lo, p_hi: Map weight [0,1] onto probability [p_lo, p_hi].

    Returns:
        Array of shape (F,) of probabilities in [0,1], or None.
    """
    if lead_prior_dict is None or lead_names is None:
        return None

    p_default = float(np.clip(p_default, 1e-3, 1.0 - 1e-3))

    probs: List[float] = []
    for L in lead_names:
        w = lead_prior_dict.get(L, None)
        if w is None:
            probs.append(p_default)
        else:
            w = float(np.clip(w, 0.0, 1.0))
            probs.append(float(p_lo + (p_hi - p_lo) * w))

    return np.asarray(probs, dtype=np.float32)


def make_event_segments(x_tf: np.ndarray, fs: float, params: Mapping[str, Any], lead_names: Optional[Sequence[str]] = None) -> Tuple[List[Segment], int]:
    """Build event segments (time windows) for event-level explanation.

    Currently supports only uniform, non-overlapping fixed-length windows.

    Args:
        x_tf: ECG array of shape (T, F).
        fs: Sampling frequency in Hz.
        params: Must contain:
            - "event_kind": currently only "uniform"
            - "window_sec": segment length in seconds
        lead_names: Unused here (kept for API compatibility).

    Returns:
        segments: list of (start_idx, end_idx) in samples.
        win_samp: window size in samples.
    """
    kind = params.get("event_kind", "uniform")
    T, _ = x_tf.shape

    if kind != "uniform":
        raise ValueError(
            "make_event_segments currently only supports event_kind='uniform'. "
            f"Got {kind!r}."
        )

    window_sec = float(params.get("window_sec", 0.5))
    win_samp = max(1, int(round(window_sec * fs)))

    starts = np.arange(0, T, win_samp)
    segments: List[Segment] = [(int(s), int(min(T, s + win_samp))) for s in starts]
    return segments, win_samp


# ---------------------------------------------------------------------
# Lead-prior registry: class_name -> {lead_name: weight in [0,1]}
# ---------------------------------------------------------------------
DEFAULT_LEAD_PRIOR_REGISTRY: Dict[str, Dict[str, float]] = {
    "AF": {"II": 1.0, "V1": 0.9, "III": 0.7, "aVF": 0.6},
    "AFL": {"II": 1.0, "III": 0.9, "aVF": 0.9, "V1": 0.7},
    "I-AVB": {"II": 1.0, "V1": 0.8},
    "RBBB": {"V1": 1.0, "V2": 0.9, "V5": 0.6, "V6": 0.6, "I": 0.4},
    "LBBB": {"V5": 1.0, "V6": 0.9, "I": 0.8, "V1": 0.4},
    "PAC": {"II": 0.9, "V1": 0.8},
    "PVC": {"V1": 0.9, "V2": 0.9, "V3": 0.8, "V4": 0.8},
    "STEMI": {"V1": 0.5, "V2": 0.6, "V3": 0.8, "V4": 0.9, "V5": 0.9, "V6": 0.8, "II": 0.7, "III": 0.7, "aVF": 0.7},
    "NSTEMI": {"V1": 0.4, "V2": 0.5, "V3": 0.7, "V4": 0.8, "V5": 0.8, "V6": 0.7, "II": 0.6, "III": 0.6, "aVF": 0.6},
    "TINV": {"V2": 0.9, "V3": 1.0, "V4": 0.9, "V5": 0.8, "V6": 0.7},
    "BRADY": {"II": 1.0, "V1": 0.7},
    "TACHY": {"II": 1.0, "V1": 0.7},
}

_LEAD_PRIOR_ALIASES: Dict[str, str] = {
    "ATRIAL FIBRILLATION": "AF",
    "AFIB": "AF",
    "ATRIAL FLUTTER": "AFL",
    "FIRST DEGREE AV BLOCK": "I-AVB",
    "1ST DEGREE AV BLOCK": "I-AVB",
    "BRADYCARDIA": "BRADY",
    "TACHYCARDIA": "TACHY",
}


def lead_prior_from_class_name(class_name: Optional[str], registry: Optional[Mapping[str, Mapping[str, float]]] = None) -> Optional[Dict[str, float]]:
    """Map a class name (e.g., 'AF') to a per-lead weight dict.

    Args:
        class_name: Label like 'AF', 'AFIB', etc.
        registry: Optional custom registry. Defaults to DEFAULT_LEAD_PRIOR_REGISTRY.

    Returns:
        Dict of lead_name -> weight in [0,1], or None if not found.
    """
    if not class_name:
        return None
    key = class_name.strip().upper()
    key = _LEAD_PRIOR_ALIASES.get(key, key)
    reg = registry or DEFAULT_LEAD_PRIOR_REGISTRY
    hit = reg.get(key)
    return dict(hit) if hit is not None else None
