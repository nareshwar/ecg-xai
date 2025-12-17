# explainer.py
from __future__ import annotations

import os
import time
from typing import Sequence, Dict, Optional, Tuple, Any

import pandas as pd
from tqdm import tqdm

from ecg_lime import run_lime_for_one_class_from_sel
from ecg_timeshap import run_timeshap_for_one_class_from_sel
from fusion import fuse_lime_timeshap_payload
from payload import payload_from_lime_row, payload_from_timeshap_row
from plot import plot_from_payload

# ---------------------------------------------------------------------
# Lead-prior registry keyed by SNOMED CT codes for your target classes.
# ---------------------------------------------------------------------
LEAD_PRIOR_BY_SNOMED: Dict[str, Dict[str, float]] = {
    # 426783006: sinus rhythm
    "426783006": {"II": 1.0, "V1": 1.0, "I": 0.8, "aVF": 0.8, "V2": 0.8},
    # 164889003: atrial fibrillation
    "164889003": {"II": 1.0, "V1": 0.9, "III": 0.7, "aVF": 0.6, "V2": 0.5},
    # 17338001: ventricular premature beats
    "17338001": {"V1": 1.0, "V2": 0.9, "V3": 0.8, "V4": 0.8, "II": 0.6},
}

# Optional: per-class window size (seconds)
WINDOW_SEC_BY_SNOMED: Dict[str, float] = {
    "426783006": 0.25,  # sinus rhythm: shorter window
    "164889003": 0.5,
    "17338001": 0.4,
}

BASE_WINDOW_SEC: float = 0.5


def _fmt_s(sec: float) -> str:
    sec = int(max(0, sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def _safe_basename(p: Any) -> str:
    if p is None:
        return ""
    try:
        return os.path.basename(str(p))
    except Exception:
        return ""


def lead_prior_for_class_name(class_name: str) -> Optional[Dict[str, float]]:
    """Return a dict[lead_name -> weight in 0..1] for a SNOMED code."""
    if class_name is None:
        return None
    return LEAD_PRIOR_BY_SNOMED.get(str(class_name))


def default_explainer_config(class_name: str) -> Dict[str, object]:
    """
    Central place to specialise settings per class:
      - window_sec, m_event, m_feat, topk_events, mode
      - lead_prior used by both LIME and TimeSHAP
    """
    code = str(class_name)

    window_sec = float(WINDOW_SEC_BY_SNOMED.get(code, BASE_WINDOW_SEC))
    m_event = 100
    m_feat = 100
    topk_events = 5
    mode = "context"

    lead_prior = lead_prior_for_class_name(code)

    params = {
        "event_kind": "uniform",
        "window_sec": window_sec,
        "lead_prior": lead_prior,
    }

    return {
        "window_sec": window_sec,
        "m_event": m_event,
        "m_feat": m_feat,
        "topk_events": topk_events,
        "mode": mode,
        "params": params,
    }

def run_explain_for_one_class(
    class_name: str,
    sel_df: pd.DataFrame,
    model,
    class_names: Sequence[str],
    max_examples: int | None = None,
    plot: bool = False,
    progress: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    """
    Run LIME explanations for a single class and return:
      - df_lime_cls
      - payloads: dict[val_idx -> plotting payload]
    """
    cfg = default_explainer_config(class_name)
    t0 = time.perf_counter()

    if verbose:
        tqdm.write(
            f"\n LIME class={class_name} | window={cfg['window_sec']}s "
            f"| m_event={cfg['m_event']} | m_feat={cfg['m_feat']} | topk_events={cfg['topk_events']}"
        )

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

    if max_examples is not None and len(df_lime_cls) > max_examples:
        df_lime_cls = df_lime_cls.iloc[:max_examples].reset_index(drop=True)

    payloads: Dict[int, Dict] = {}

    it = df_lime_cls.iterrows()
    if progress:
        it = tqdm(list(it), desc=f"Payloads {class_name}", leave=False)

    for _, row in it:
        val_idx = int(row.get("val_idx", row.name))
        payload = payload_from_lime_row(row, label_for_title=row.get("group_class", class_name))
        payloads[val_idx] = payload
        if plot:
            plot_from_payload(payload)

    if verbose:
        tqdm.write(f"  LIME done: {len(df_lime_cls)} rows in {_fmt_s(time.perf_counter()-t0)}")

    return df_lime_cls, payloads


def run_pipeline_for_classes(
    target_classes: Sequence[str],
    sel_df: pd.DataFrame,
    model,
    class_names: Sequence[str],
    max_examples_per_class: int | None = None,
    plot: bool = False,
    progress: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[int, Dict]], pd.DataFrame]:
    
    """Run LIME for multiple classes and return all_payloads + concatenated df."""
    
    all_payloads: Dict[str, Dict[int, Dict]] = {}
    all_dfs: list[pd.DataFrame] = []

    cls_iter = list(target_classes)
    if progress:
        cls_iter = tqdm(cls_iter, desc="Classes (LIME)")

    t_start = time.perf_counter()
    times: list[float] = []

    for i, cls in enumerate(cls_iter, start=1):
        t_cls = time.perf_counter()

        df_cls, payloads_cls = run_explain_for_one_class(
            class_name=cls,
            sel_df=sel_df,
            model=model,
            class_names=class_names,
            max_examples=max_examples_per_class,
            plot=plot,
            progress=progress,
            verbose=verbose,
        )

        all_payloads[cls] = payloads_cls
        all_dfs.append(df_cls)

        dt = time.perf_counter() - t_cls
        times.append(dt)

        if verbose:
            avg = sum(times) / len(times)
            remaining = (len(target_classes) - i) * avg
            tqdm.write(f" — Progress: {i}/{len(target_classes)} classes | ETA ~ {_fmt_s(remaining)}")

    if verbose:
        tqdm.write(f"\n LIME pipeline complete in {_fmt_s(time.perf_counter()-t_start)}")

    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return all_payloads, df_all


def run_fused_for_one_class(
    class_name: str,
    sel_df: pd.DataFrame,
    model,
    class_names: Sequence[str],
    max_examples: Optional[int] = None,
    plot: bool = False,
    progress: bool = True,
    verbose: bool = True,
    show_record: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict]]:
    """
    Run LIME + TimeSHAP + fusion for a single class.

    Returns: (df_lime_cls, df_ts_cls, fused_payloads[val_idx] = payload_F)
    """
    cfg = default_explainer_config(class_name)
    t0 = time.perf_counter()

    if verbose:
        tqdm.write(
            f"\n FUSED class={class_name} | window={cfg['window_sec']}s "
            f"| m_event={cfg['m_event']} | m_feat={cfg['m_feat']} | topk_events={cfg['topk_events']}"
        )

    sel_df_cls = _subset_sel_df_for_class(sel_df, class_name, max_examples)

    # ---- LIME ----
    t_l = time.perf_counter()
    df_lime_cls = run_lime_for_one_class_from_sel(
        sel_df_cls,
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
    if verbose:
        tqdm.write(f" LIME done: {len(df_lime_cls)} rows in {_fmt_s(time.perf_counter()-t_l)}")

    # ---- TimeSHAP ----
    t_t = time.perf_counter()
    df_ts_cls = run_timeshap_for_one_class_from_sel(
        sel_df_cls,
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
    if verbose:
        tqdm.write(f" TimeSHAP done: {len(df_ts_cls)} rows in {_fmt_s(time.perf_counter()-t_t)}")

    # ---- fuse common val_idx ----
    fused_payloads: Dict[int, Dict] = {}

    if "val_idx" not in df_lime_cls.columns or "val_idx" not in df_ts_cls.columns:
        raise KeyError("Expected 'val_idx' column in both LIME and TimeSHAP dataframes.")

    common_val_idx = sorted(
        set(df_lime_cls["val_idx"].astype(int)) & set(df_ts_cls["val_idx"].astype(int))
    )

    if verbose:
        tqdm.write(f"  Fusing: {len(common_val_idx)} common records")

    it = common_val_idx
    if progress:
        it = tqdm(common_val_idx, desc=f"Fuse {class_name}", leave=False)

    for val_i in it:
        row_L = df_lime_cls.loc[df_lime_cls["val_idx"] == val_i].iloc[0]
        row_T = df_ts_cls.loc[df_ts_cls["val_idx"] == val_i].iloc[0]

        payload_L = payload_from_lime_row(row_L, label_for_title=row_L.get("group_class", class_name))
        payload_T = payload_from_timeshap_row(row_T, label_for_title=row_T.get("group_class", class_name))

        # class-specific method weights (optional)
        code = str(class_name)
        method_weights = (0.5, 0.5)
        if code == "426783006":  # sinus rhythm
            method_weights = (0.7, 0.3)  # (LIME, TimeSHAP)

        payload_F = fuse_lime_timeshap_payload(
            payload_L,
            payload_T,
            agg="geomean",
            beta=1.0,
            tau=0.02,
            topk=5,
            method_weights=method_weights,
        )

        fused_payloads[int(val_i)] = payload_F

        # progress postfix: show which record (best-effort)
        if progress and show_record:
            fname = _safe_basename(row_L.get("filename", row_L.get("mat_path", "")))
            sel_idx = row_L.get("sel_idx", "")
            it.set_postfix_str(f"val={int(val_i)} sel={sel_idx} file={fname}")

        if plot:
            plot_from_payload(payload_F)

    if verbose:
        tqdm.write(f"  Class {class_name} total: {_fmt_s(time.perf_counter()-t0)}")

    return df_lime_cls, df_ts_cls, fused_payloads


def run_fused_pipeline_for_classes(
    target_classes: Sequence[str],
    sel_df: pd.DataFrame,
    model,
    class_names: Sequence[str],
    max_examples_per_class: Optional[int] = None,
    plot: bool = False,
    progress: bool = True,
    verbose: bool = True,
    show_record: bool = True,
) -> Tuple[Dict[str, Dict[int, Dict]], pd.DataFrame, pd.DataFrame]:
    """
    High-level fused pipeline:
      - loop over target classes
      - run LIME + TimeSHAP + fusion for each
      - collect fused payloads + LIME/TimeSHAP dataframes
    """
    all_fused_payloads: Dict[str, Dict[int, Dict]] = {}
    all_lime: list[pd.DataFrame] = []
    all_ts: list[pd.DataFrame] = []

    classes = list(target_classes)
    cls_iter = classes
    if progress:
        cls_iter = classes

    t_start = time.perf_counter()
    times: list[float] = []

    for i, cls in enumerate(cls_iter, start=1):
        t_cls = time.perf_counter()

        if verbose:
            tqdm.write(f"\n=== [{i}/{len(classes)}] Processing class: {cls} ===")

        df_lime_cls, df_ts_cls, fused_payloads_cls = run_fused_for_one_class(
            class_name=cls,
            sel_df=sel_df,
            model=model,
            class_names=class_names,
            max_examples=max_examples_per_class,
            plot=plot,
            progress=progress,
            verbose=verbose,
            show_record=show_record,
        )

        all_fused_payloads[cls] = fused_payloads_cls
        all_lime.append(df_lime_cls)
        all_ts.append(df_ts_cls)

        dt = time.perf_counter() - t_cls
        times.append(dt)

        if verbose:
            avg = sum(times) / len(times)
            remaining = (len(classes) - i) * avg
            tqdm.write(f"— Progress: {i}/{len(classes)} classes | ETA ~ {_fmt_s(remaining)}")

    if verbose:
        tqdm.write(f"\n All classes complete in {_fmt_s(time.perf_counter()-t_start)}")

    df_lime_all = pd.concat(all_lime, ignore_index=True) if all_lime else pd.DataFrame()
    df_ts_all = pd.concat(all_ts, ignore_index=True) if all_ts else pd.DataFrame()
    return all_fused_payloads, df_lime_all, df_ts_all

def _subset_sel_df_for_class(sel_df: pd.DataFrame, class_name: str, max_examples: int | None):
    df = sel_df[sel_df["group_class"].astype(str) == str(class_name)].copy()

    # If you have a probability column, prefer highest confidence examples
    for col in ["prob", "p", "pred_prob", "max_prob", "score"]:
        if col in df.columns:
            df = df.sort_values(col, ascending=False)
            break

    if max_examples is not None:
        df = df.head(int(max_examples)).reset_index(drop=True)

    return df