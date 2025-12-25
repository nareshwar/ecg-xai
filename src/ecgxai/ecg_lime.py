import json
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge

from preprocessing import ensure_paths, parse_fs_and_leads, load_mat_TF
from utils import cosine_distance_to_ones, weighted_ridge_fit, class_index

def lime_events(model, x_tf, class_idx, segments, m_masks=200, kernel_width=0.25, rng=42):
    """
    Event-level LIME over given time segments.
    """
    T, F = x_tf.shape
    E = len(segments)

    rng = np.random.default_rng(rng)
    M = rng.binomial(1, 0.5, size=(m_masks, E)).astype(np.float32)
    M[0, :] = 1.0  # baseline all-on

    Xp = np.repeat(x_tf[None, ...], m_masks, axis=0)
    for e, (s, t) in enumerate(segments):
        Xp[:, s:t, :] *= M[:, e][:, None, None]

    y = model.predict(Xp, verbose=0)[:, class_idx]

    d = cosine_distance_to_ones(M)
    w = np.sqrt(np.exp(-(d ** 2) / (kernel_width ** 2)))

    coef, _ = weighted_ridge_fit(M, y, sample_weight=w, alpha=1e-2, fit_intercept=True)
    return coef.astype(np.float32)

def lime_features_in_event(
    model,
    x_tf,
    class_idx,
    seg,
    m_masks: int = 200,
    kernel_width: float = 0.25,
    mode: str = "context",
    rng: int = 42,
    lead_prior_weights=None,
):
    """
    Feature-level LIME inside a single time window.

    Parameters
    ----------
    lead_prior_weights : optional (F,) array of per-lead probabilities.
        If provided, feature masks are sampled with P(mask_j = 1) =
        lead_prior_weights[j]. If None, each feature has probability 0.5
        (original behaviour).
    """
    T, F = x_tf.shape
    s, t = seg
    rng = np.random.default_rng(rng)

    # --- sample feature masks ------------------------------------------
    if lead_prior_weights is not None:
        p = np.asarray(lead_prior_weights, dtype=np.float64)
        if p.shape[0] != F:
            raise ValueError(
                f"lead_prior_weights length {p.shape[0]} != num features {F}"
            )
        # avoid exact 0 or 1 for stability
        p = np.clip(p, 1e-3, 1.0 - 1e-3)
        M = rng.binomial(1, p, size=(m_masks, F)).astype(np.float32)
    else:
        # original behaviour: each feature independently on with prob 0.5
        M = rng.binomial(1, 0.5, size=(m_masks, F)).astype(np.float32)

    # first mask: all features on
    M[0, :] = 1.0

    # --- build perturbed inputs ----------------------------------------
    if mode == "isolated":
        Xp = np.zeros((m_masks, T, F), dtype=np.float32)
        for i in range(m_masks):
            Xp[i, s:t, :] = x_tf[s:t, :] * M[i][None, :]
    else:
        Xp = np.repeat(x_tf[None, ...], m_masks, axis=0)
        for i in range(m_masks):
            Xp[i, s:t, :] = x_tf[s:t, :] * M[i][None, :]

    y = model.predict(Xp, verbose=0)[:, class_idx]

    d = pairwise_distances(M, np.ones((1, F)), metric="cosine").ravel()
    w = np.sqrt(np.exp(-(d ** 2) / (kernel_width ** 2)))

    reg = Ridge(alpha=1e-2, solver="svd")
    reg.fit(M, y, sample_weight=w)
    return reg.coef_.astype(np.float32)


def run_lime_for_one_class_from_sel(
    sel_df,
    class_name,
    *,
    model,
    class_names,
    window_sec=0.5,
    m_event=200,
    m_feat=200,
    topk_events=5,
    explain_class="force",
    mode="context",
    rng=42,
    params=None,
):
    """
    Run LIME explanations for all rows in sel_df with group_class == class_name.

    sel_df columns needed:
        - group_class
        - filename (path to .mat or .hea)
        - optional: sel_idx (index within validation split)
    """
    # normalise / default params so we can share with TimeSHAP
    if params is None:
        params = {
            "event_kind": "uniform",
            "window_sec": float(window_sec),
            "lead_prior": None,
        }
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

    out = []

    for _, r in rows.iterrows():
        val_idx = int(r.get("sel_idx", r.name))

        hea_path, mat_path = ensure_paths(r["filename"])
        fs, lead_names = parse_fs_and_leads(hea_path, default_fs=500.0)

        # build per-lead prior probabilities
        lead_prior_probs = prior_probs_from_names(
            lead_names,
            params.get("lead_prior") if params else None,
        )

        x_tf = load_mat_TF(mat_path)
        T, F = x_tf.shape

        if explain_class == "force":
            c = c_force
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

        perlead_spans = {j: [] for j in range(F)}
        for e_idx in topE:
            s, t = segments[e_idx]
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
                s_sec = s / fs
                t_sec = t / fs
                perlead_spans[j].append((float(s_sec), float(t_sec), float(coef_f[j])))

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
                "window_sec": float(window_sec_eff),
                "target_class_explained": int(c),
                "event_importances_json": json.dumps([float(v) for v in event_scores]),
                "segments_json": json.dumps([[int(s), int(t)] for (s, t) in segments]),
                "window_size_samples": int(win_samp),
                "num_segments": int(len(segments)),
                "top_events_idx_json": json.dumps([int(x) for x in list(topE)]),
                "top5_lead_idx_json": json.dumps([int(j) for j in top5]),
                "perlead_spans_top5_json": json.dumps(
                    {
                        int(j): [
                            (float(s), float(t), float(w))
                            for (s, t, w) in spans_top5[j]
                        ]
                        for j in spans_top5
                    }
                ),
                "lead_names": ",".join(lead_names) if lead_names is not None else None,
            }
        )

    df_lime = (
        pd.DataFrame(out)
        .sort_values(["group_class", "val_idx"])
        .reset_index(drop=True)
    )
    return df_lime

def prior_probs_from_names(
    lead_names,
    lead_prior_dict,
    p_default: float = 0.02,
    p_lo: float = 0.1,
    p_hi: float = 0.99,
):
    """
    Map a dict of lead priors into per-lead probabilities.

    Parameters
    ----------
    lead_names : list of lead names (e.g. ["I","II","V1",...])
    lead_prior_dict : dict[lead_name -> weight in [0,1]]
        Typically comes from a registry of "important" leads per diagnosis.
        If None, returns None and caller will ignore priors.
    p_default : baseline probability for leads not in dict
    p_lo, p_hi : map weight [0,1] onto [p_lo, p_hi]

    Returns
    -------
    probs : np.ndarray of shape (F,) or None
    """
    if lead_prior_dict is None or lead_names is None:
        return None

    probs = []
    for L in lead_names:
        w = lead_prior_dict.get(L, None)
        if w is None:
            probs.append(0.0)
        else:
            w = float(w)
            w = min(max(w, 0.0), 1.0)
            p = p_lo + (p_hi - p_lo) * w
            probs.append(p)
    return np.asarray(probs, dtype=np.float32)


def make_event_segments(x_tf, fs: float, params: dict, lead_names=None):
    """
    Return list of (start_sample, end_sample) for event-level explanation.

    For now we implement only `event_kind == "uniform"`, which splits the
    record into non-overlapping fixed-length windows of `window_sec`.

    Parameters
    ----------
    x_tf : np.ndarray, shape (T, F)
        ECG in model space (time x features).
    fs : float
        Sampling frequency in Hz.
    params : dict
        Must contain at least:
            - "event_kind": "uniform"
            - "window_sec": segment length in seconds
        Other values are ignored in this simple version.
    lead_names : unused here, kept for API compatibility.

    Returns
    -------
    segments : list of (start_idx, end_idx)
    win_samp : window size in samples
    """
    kind = params.get("event_kind", "uniform")
    T, F = x_tf.shape

    if kind != "uniform":
        raise ValueError(
            f"make_event_segments currently only supports event_kind='uniform', "
            f"got {kind!r}. You can extend this later for beat_qrs / qrs_family."
        )

    window_sec = float(params.get("window_sec", 0.5))
    win_samp = max(1, int(round(window_sec * fs)))

    starts = np.arange(0, T, win_samp)
    segments = [
        (int(s), int(min(T, s + win_samp)))
        for s in starts
    ]

    return segments, win_samp


# ---------------------------------------------------------------------
# Lead-prior registry: class_name -> {lead_name: weight in [0,1]}
# ---------------------------------------------------------------------

DEFAULT_LEAD_PRIOR_REGISTRY = {
    # Atrial fibrillation: irregular baseline + fibrillatory waves
    # best seen in II and V1
    "AF": {
        "II": 1.0,
        "V1": 0.9,
        "III": 0.7,
        "aVF": 0.6,
    },

    # Atrial flutter: sawtooth waves, inferior + V1
    "AFL": {
        "II": 1.0,
        "III": 0.9,
        "aVF": 0.9,
        "V1": 0.7,
    },

    # 1st-degree AV block: long PR → II & V1 are good
    "I-AVB": {
        "II": 1.0,
        "V1": 0.8,
    },

    # Right bundle-branch block: V1–V2 (RSR') and lateral leads
    "RBBB": {
        "V1": 1.0,
        "V2": 0.9,
        "V5": 0.6,
        "V6": 0.6,
        "I": 0.4,
    },

    # Left bundle-branch block: I, V5, V6 mostly
    "LBBB": {
        "V5": 1.0,
        "V6": 0.9,
        "I": 0.8,
        "V1": 0.4,
    },

    # Premature atrial complexes
    "PAC": {
        "II": 0.9,
        "V1": 0.8,
    },

    # Premature ventricular complexes
    "PVC": {
        "V1": 0.9,
        "V2": 0.9,
        "V3": 0.8,
        "V4": 0.8,
    },

    # Generic ST elevation / MI (you can split by territory if needed)
    "STEMI": {
        "V1": 0.5,
        "V2": 0.6,
        "V3": 0.8,
        "V4": 0.9,
        "V5": 0.9,
        "V6": 0.8,
        "II": 0.7,
        "III": 0.7,
        "aVF": 0.7,
    },

    "NSTEMI": {
        "V1": 0.4,
        "V2": 0.5,
        "V3": 0.7,
        "V4": 0.8,
        "V5": 0.8,
        "V6": 0.7,
        "II": 0.6,
        "III": 0.6,
        "aVF": 0.6,
    },

    # T-wave inversion / repolarisation abnormalities
    "TINV": {
        "V2": 0.9,
        "V3": 1.0,
        "V4": 0.9,
        "V5": 0.8,
        "V6": 0.7,
    },

    # Rate problems – II + V1 are usually good rhythm strips
    "BRADY": {
        "II": 1.0,
        "V1": 0.7,
    },
    "TACHY": {
        "II": 1.0,
        "V1": 0.7,
    },
}

# A few textual aliases (optional – extend as you meet your labels)
_LEAD_PRIOR_ALIASES = {
    "ATRIAL FIBRILLATION": "AF",
    "AFIB": "AF",
    "ATRIAL FLUTTER": "AFL",
    "FIRST DEGREE AV BLOCK": "I-AVB",
    "1ST DEGREE AV BLOCK": "I-AVB",
    "BRADYCARDIA": "BRADY",
    "TACHYCARDIA": "TACHY",
}

def lead_prior_from_class_name(
    class_name: str,
    registry=None,
):
    """
    Map a class_name (e.g. 'AF') to a per-lead weight dict.

    Returns
    -------
    dict[str, float] or None
    """
    if class_name is None:
        return None
    key = class_name.strip().upper()

    # normalise aliases like "AFIB" -> "AF"
    key = _LEAD_PRIOR_ALIASES.get(key, key)

    reg = registry or DEFAULT_LEAD_PRIOR_REGISTRY
    return reg.get(key)

