import pandas as pd

from typing import Sequence, Dict, Optional, Tuple

from ecg_lime import run_lime_for_one_class_from_sel
from ecg_timeshap import run_timeshap_for_one_class_from_sel
from fusion import fuse_lime_timeshap_payload
from payload import payload_from_lime_row, payload_from_timeshap_row
from plot import plot_from_payload

# ---------------------------------------------------------------------
# Lead-prior registry keyed by SNOMED CT codes for your target classes.
# You can extend this as you add more diagnoses.
# ---------------------------------------------------------------------

LEAD_PRIOR_BY_SNOMED: Dict[str, Dict[str, float]] = {
    # 426783006: sinus rhythm
    "426783006": {
        # core: highest
        "II": 1.0,
        "V1": 1.0,
        # extended: still strong
        "I": 0.8,
        "aVF": 0.8,
        "V2": 0.8,
    },

    # 164889003: AF (example)
    "164889003": {
        "II": 1.0,
        "V1": 0.9,
        "III": 0.7,
        "aVF": 0.6,
        "V2": 0.5,
    },

    # 17338001: PVC (example)
    "17338001": {
        "V1": 1.0,
        "V2": 0.9,
        "V3": 0.8,
        "V4": 0.8,
        "II": 0.6,
    },
}

WINDOW_SEC_BY_SNOMED: Dict[str, float] = {
    "426783006": 0.25,   # e.g. shorter windows around P/QRS
    "164889003": 0.5,
    "17338001": 0.4,
}

def lead_prior_for_class_name(class_name: str) -> Optional[Dict[str, float]]:
    """Return a dict[lead_name -> weight in 0..1] for a SNOMED code.

    If the class is not in the registry, returns None and the explainer
    will fall back to uniform feature sampling.
    """
    if class_name is None:
        return None
    return LEAD_PRIOR_BY_SNOMED.get(str(class_name))

def default_explainer_config(class_name: str) -> Dict[str, object]:
    """
    Return explainer hyperparameters for a given class.

    This is the central place where you can specialise settings per class:
        - window_sec, m_event, m_feat, topk_events, mode
        - lead_prior used by both LIME and TimeSHAP
    """
    base_window = 0.5  # default if class not in WINDOW_SEC_BY_SNOMED
    window_sec = float(WINDOW_SEC_BY_SNOMED.get(str(class_name), base_window))

    # Look up a lead-prior dict for this SNOMED code, if available.
    lead_prior = lead_prior_for_class_name(class_name)

    params = {
        "event_kind": "uniform",   # currently only uniform windows implemented
        "window_sec": window_sec,
        "lead_prior": lead_prior,
    }

    return {
        "window_sec": window_sec,   # segment length in seconds
        "m_event": 300,             # number of event-level masks
        "m_feat": 400,              # number of feature-level masks
        "topk_events": 5,           # how many top events to refine per-lead
        "mode": "context",          # 'context' or 'isolated'
        "params": params,           # shared params for LIME + TimeSHAP
    }

def run_explain_for_one_class(
    class_name: str,
    sel_df: pd.DataFrame,
    model,
    class_names: Sequence[str],
    max_examples: int | None = None,
    plot: bool = False,
) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    """
    Run LIME explanations for a single diagnosis label.

    Parameters
    ----------
    class_name : target diagnosis name (e.g. "ventricular premature beats")
    sel_df     : selection dataframe with at least:
                 - 'group_class' (string)
                 - 'filename'    (path to .mat)
                 - 'sel_idx'     (validation index within fold)
    model      : Keras model (T,F) -> (num_classes,)
    class_names: list/array of SNOMED class names (len C)
    max_examples : if not None, limit number of ECGs for this class
    plot       : if True, plot each payload via plot_from_payload

    Returns
    -------
    df_lime_cls : per-ECG LIME rows for this class
    payloads    : dict[val_idx -> plotting payload]
    """
    cfg = default_explainer_config(class_name)

    # Run LIME for this class
    df_lime_cls = run_lime_for_one_class_from_sel(
        sel_df,
        class_name,
        model=model,
        class_names=class_names,
        window_sec=cfg["window_sec"],
        m_event=cfg["m_event"],
        m_feat=cfg["m_feat"],
        topk_events=cfg["topk_events"],
        explain_class="force",
        mode=cfg["mode"],
        rng=42,
        params=cfg.get("params"),
    )

    # Optionally limit number of examples per class
    if max_examples is not None and len(df_lime_cls) > max_examples:
        df_lime_cls = df_lime_cls.iloc[:max_examples].reset_index(drop=True)

    # Build plotting payloads
    payloads: Dict[int, Dict] = {}
    for _, row in df_lime_cls.iterrows():
        # val_idx is the index within your validation fold;
        # fall back to row index if it's not present
        val_idx = int(row.get("val_idx", row.name))

        payload = payload_from_lime_row(
            row,
            label_for_title=row.get("group_class", class_name),
        )
        payloads[val_idx] = payload

        if plot:
            plot_from_payload(payload)

    return df_lime_cls, payloads


def run_pipeline_for_classes(
    target_classes: Sequence[str],
    sel_df: pd.DataFrame,
    model,
    class_names: Sequence[str],
    max_examples_per_class: int | None = None,
    plot: bool = False,
) -> Tuple[Dict[str, Dict[int, Dict]], pd.DataFrame]:
    """
    High-level pipeline:
      - loop over target classes
      - run explanations for each
      - collect payloads and a single combined dataframe

    Parameters
    ----------
    target_classes : list of diagnosis names
    sel_df         : selection dataframe
    model          : Keras model
    class_names    : SNOMED class names
    max_examples_per_class : cap number of ECGs per class
    plot           : if True, plot explanations as you go

    Returns
    -------
    all_payloads : dict[class_name -> dict[val_idx -> payload]]
    df_all       : concatenated LIME rows for all classes
    """
    all_payloads: Dict[str, Dict[int, Dict]] = {}
    all_dfs: list[pd.DataFrame] = []

    for cls in target_classes:
        df_cls, payloads_cls = run_explain_for_one_class(
            class_name=cls,
            sel_df=sel_df,
            model=model,
            class_names=class_names,
            max_examples=max_examples_per_class,
            plot=plot,
        )
        all_payloads[cls] = payloads_cls
        all_dfs.append(df_cls)

    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return all_payloads, df_all

def run_fused_for_one_class(
    class_name: str,
    sel_df: pd.DataFrame,
    model,
    class_names: Sequence[str],
    max_examples: Optional[int] = None,
    plot: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict]]:
    """
    Run LIME + TimeSHAP + fusion for a single class.

    Returns
    -------
    df_lime_cls  : LIME rows for this class
    df_ts_cls    : TimeSHAP rows for this class
    fused_payloads : dict[val_idx -> fused payload]
    """
    cfg = default_explainer_config(class_name)

    # LIME
    df_lime_cls = run_lime_for_one_class_from_sel(
        sel_df,
        class_name,
        model=model,
        class_names=class_names,
        window_sec=cfg["window_sec"],
        m_event=cfg["m_event"],
        m_feat=cfg["m_feat"],
        topk_events=cfg["topk_events"],
        explain_class="force",
        mode=cfg["mode"],
        rng=42,
        params=cfg.get("params"), 
    )

    # TimeSHAP
    df_ts_cls = run_timeshap_for_one_class_from_sel(
        sel_df,
        class_name,
        model=model,
        class_names=class_names,
        window_sec=cfg["window_sec"],
        m_event=cfg["m_event"],
        m_feat=cfg["m_feat"],
        topk_events=cfg["topk_events"],
        explain_class="force",
        mode=cfg["mode"],
        rng=42,
        link="logit",
        params=cfg.get("params"), 
    )

    # Optionally limit number of examples
    if max_examples is not None:
        df_lime_cls = df_lime_cls.iloc[:max_examples].reset_index(drop=True)
        df_ts_cls = df_ts_cls.iloc[:max_examples].reset_index(drop=True)

    # Fuse LIME + TimeSHAP per val_idx
    fused_payloads: Dict[int, Dict] = {}

    common_val_idx = sorted(
        set(df_lime_cls["val_idx"].astype(int))
        & set(df_ts_cls["val_idx"].astype(int))
    )

    for val_i in common_val_idx:
        row_L = df_lime_cls.loc[df_lime_cls["val_idx"] == val_i].iloc[0]
        row_T = df_ts_cls.loc[df_ts_cls["val_idx"] == val_i].iloc[0]

        payload_L = payload_from_lime_row(
            row_L,
            label_for_title=row_L.get("group_class", class_name),
        )
        payload_T = payload_from_timeshap_row(
            row_T,
            label_for_title=row_T.get("group_class", class_name),
        )

        # Simple fuse settings (you can specialize per class later)
        payload_F = fuse_lime_timeshap_payload(
            payload_L,
            payload_T,
            agg="geomean",
            beta=1.0,
            tau=0.02,
            topk=5,
        )

        fused_payloads[int(val_i)] = payload_F

        if plot:
            plot_from_payload(payload_F)

    return df_lime_cls, df_ts_cls, fused_payloads

def run_fused_pipeline_for_classes(
    target_classes: Sequence[str],
    sel_df: pd.DataFrame,
    model,
    class_names: Sequence[str],
    max_examples_per_class: Optional[int] = None,
    plot: bool = False,
) -> Tuple[Dict[str, Dict[int, Dict]], pd.DataFrame, pd.DataFrame]:
    """
    High-level fused pipeline:
      - loop over target classes
      - run LIME + TimeSHAP + fusion for each
      - collect fused payloads + LIME/TimeSHAP dataframes

    Returns
    -------
    all_fused_payloads : dict[class_name -> dict[val_idx -> fused payload]]
    df_lime_all        : concatenated LIME rows for all classes
    df_ts_all          : concatenated TimeSHAP rows for all classes
    """
    all_fused_payloads: Dict[str, Dict[int, Dict]] = {}
    all_lime: list[pd.DataFrame] = []
    all_ts: list[pd.DataFrame] = []

    for cls in target_classes:
        df_lime_cls, df_ts_cls, fused_payloads_cls = run_fused_for_one_class(
            class_name=cls,
            sel_df=sel_df,
            model=model,
            class_names=class_names,
            max_examples=max_examples_per_class,
            plot=plot,
        )

        all_fused_payloads[cls] = fused_payloads_cls
        all_lime.append(df_lime_cls)
        all_ts.append(df_ts_cls)

    df_lime_all = pd.concat(all_lime, ignore_index=True) if all_lime else pd.DataFrame()
    df_ts_all = pd.concat(all_ts, ignore_index=True) if all_ts else pd.DataFrame()

    return all_fused_payloads, df_lime_all, df_ts_all