from __future__ import annotations

"""
ecgxai.explainer

High-level orchestration for generating ECG explanations.

This module provides:
- Per-class configuration (window size, lead priors, sampling sizes).
- LIME-only pipeline: run_explain_for_one_class(), run_pipeline_for_classes()
- Fused pipeline: run_fused_for_one_class(), run_fused_pipeline_for_classes()

Conventions:
- `class_name` is treated as a SNOMED CT code string (e.g., "426783006").
- `sel_df` must contain:
    - "group_class": SNOMED code (string/int OK)
    - "filename": record path (or mat/hea base path)
    - optional: "sel_idx" (stable id), and optionally a confidence column
      ("prob", "p", "pred_prob", "max_prob", "score") for best-example selection.

Logging:
- Environment default: ECGXAI_LOG_STYLE in {"showcase","debug","quiet"} (default "showcase")
- `verbose=True` forces "debug" style.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from .ecg_lime import run_lime_for_one_class_from_sel
from .ecg_timeshap import run_timeshap_for_one_class_from_sel
from .fusion import fuse_lime_timeshap_payload
from .payload import payload_from_lime_row, payload_from_timeshap_row
from .plot import plot_from_payload


# ---------------------------------------------------------------------
# Logging (showcase-friendly)
#   ECGXAI_LOG_STYLE = "showcase" | "debug" | "quiet"
#   - showcase: one line per class + one final line (default)
#   - debug: detailed logs
#   - quiet: no logs
# ---------------------------------------------------------------------
DEFAULT_LOG_STYLE = os.environ.get("ECGXAI_LOG_STYLE", "showcase").lower()


def _resolve_log_style(log_style: Optional[str], verbose: bool) -> str:
    """Resolve logging style from (log_style, verbose, env default)."""
    if verbose:
        return "debug"
    s = (str(log_style).lower() if log_style else DEFAULT_LOG_STYLE)
    if s not in ("quiet", "showcase", "debug"):
        s = "showcase"
    return s


def _log(style: str, msg: str) -> None:
    """Log message using tqdm.write so it plays nicely with progress bars."""
    if style == "quiet":
        return
    tqdm.write(msg)


def _fmt_s(sec: float) -> str:
    """Format seconds as XmYYs or HhMMmSSs."""
    sec = int(max(0, sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def _safe_basename(p: Any) -> str:
    """Best-effort basename for progress messages."""
    if p is None:
        return ""
    try:
        return os.path.basename(str(p))
    except Exception:
        return ""


# ---------------------------------------------------------------------
# Per-class configuration
# ---------------------------------------------------------------------
LEAD_PRIOR_BY_SNOMED: Dict[str, Dict[str, float]] = {
    # 426783006: sinus rhythm
    "426783006": {"II": 1.0, "V1": 1.0, "I": 0.8, "aVF": 0.8, "V2": 0.8},
    # 164889003: atrial fibrillation
    "164889003": {"II": 1.0, "V1": 0.9, "III": 0.7, "aVF": 0.6, "V2": 0.5},
    # 17338001: ventricular premature beats
    "17338001": {"V1": 1.0, "V2": 0.9, "V3": 0.8, "V4": 0.8, "II": 0.6},
}

WINDOW_SEC_BY_SNOMED: Dict[str, float] = {
    "426783006": 0.25,  # sinus rhythm: shorter window
    "164889003": 0.5,
    "17338001": 0.4,
}

# Optional: class-specific fusion weights (LIME, TimeSHAP)
FUSION_WEIGHTS_BY_SNOMED: Dict[str, Tuple[float, float]] = {
    "426783006": (0.7, 0.3),  # sinus rhythm: LIME weighted higher
    # default otherwise: (0.5, 0.5)
}

BASE_WINDOW_SEC: float = 0.5


@dataclass(frozen=True)
class ExplainerConfig:
    """Centralized explainer configuration for a single class."""
    window_sec: float = BASE_WINDOW_SEC
    m_event: int = 100
    m_feat: int = 100
    topk_events: int = 5
    mode: str = "context"          # "context" or "isolated"
    rng: int = 42                  # seed used by explainers
    timeshap_link: str = "logit"   # identity/logit; logit tends to work better for SHAP
    params: Dict[str, object] = None  # shared param dict passed to LIME/TimeSHAP

    def __post_init__(self):
        # dataclass(frozen=True) + mutable default: enforce dict init
        if self.params is None:
            object.__setattr__(self, "params", {})


def lead_prior_for_class_name(class_name: str) -> Optional[Dict[str, float]]:
    """Return a dict[lead_name -> weight in 0..1] for a SNOMED code."""
    if class_name is None:
        return None
    return LEAD_PRIOR_BY_SNOMED.get(str(class_name))


def get_explainer_config(class_name: str) -> ExplainerConfig:
    """Return class-specific configuration for both LIME and TimeSHAP."""
    code = str(class_name)
    window_sec = float(WINDOW_SEC_BY_SNOMED.get(code, BASE_WINDOW_SEC))
    lead_prior = lead_prior_for_class_name(code)

    params = {
        "event_kind": "uniform",
        "window_sec": window_sec,
        "lead_prior": lead_prior,  # consumed by both LIME & TimeSHAP
    }

    return ExplainerConfig(window_sec=window_sec, params=params)


# Backwards compatibility: keep the old name returning a dict-like config.
def default_explainer_config(class_name: str) -> Dict[str, object]:
    """Back-compat wrapper returning a dict config (older interface)."""
    cfg = get_explainer_config(class_name)
    return {
        "window_sec": cfg.window_sec,
        "m_event": cfg.m_event,
        "m_feat": cfg.m_feat,
        "topk_events": cfg.topk_events,
        "mode": cfg.mode,
        "params": cfg.params,
    }


def _subset_sel_df_for_class(sel_df: pd.DataFrame, class_name: str, max_examples: int | None) -> pd.DataFrame:
    """Filter sel_df for one class and optionally pick top-N by confidence."""
    df = sel_df[sel_df["group_class"].astype(str) == str(class_name)].copy()

    # If have a probability column, prefer highest confidence examples
    for col in ["prob", "p", "pred_prob", "max_prob", "score"]:
        if col in df.columns:
            df = df.sort_values(col, ascending=False)
            break

    if max_examples is not None:
        df = df.head(int(max_examples)).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# LIME-only pipeline
# ---------------------------------------------------------------------
def run_explain_for_one_class(
    class_name: str,
    sel_df: pd.DataFrame,
    model: Any,
    class_names: Sequence[str],
    max_examples: int | None = None,
    plot: bool = False,
    progress: bool = True,
    verbose: bool = True,
    log_style: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    """Run LIME for a single class and return (df_lime_cls, payloads_by_val_idx).

    Args:
        class_name: SNOMED code string (e.g. "426783006").
        sel_df: Selection dataframe containing at least "group_class" and "filename".
        model: Keras/TF model with predict().
        class_names: Array-like of model output class identifiers.
        max_examples: If set, restrict to top-N examples for this class.
        plot: If True, render plots as payloads are produced.
        progress: If True, show tqdm for payload creation.
        verbose: If True, forces debug logging.
        log_style: Optional override for logging style.

    Returns:
        df_lime_cls: LIME results dataframe (one row per ECG).
        payloads: dict[val_idx -> plotting payload]
    """
    style = _resolve_log_style(log_style, verbose)
    debug = style == "debug"

    cfg = get_explainer_config(class_name)
    t0 = time.perf_counter()

    sel_df_cls = _subset_sel_df_for_class(sel_df, class_name, max_examples)

    if debug:
        _log(style, (
            f"\nLIME class={class_name} | n={len(sel_df_cls)} "
            f"| win={cfg.window_sec}s | mE={cfg.m_event} | mF={cfg.m_feat} | topkE={cfg.topk_events}"
        ))

    df_lime_cls = run_lime_for_one_class_from_sel(
        sel_df_cls,
        class_name,
        model=model,
        class_names=class_names,
        window_sec=cfg.window_sec,
        m_event=cfg.m_event,
        m_feat=cfg.m_feat,
        topk_events=cfg.topk_events,
        explain_class="force",
        mode=cfg.mode,
        rng=cfg.rng,
        params=cfg.params,
    )

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

    if style == "showcase":
        _log(style, f"LIME {class_name} | n={len(df_lime_cls)} | total={_fmt_s(time.perf_counter()-t0)}")
    elif debug:
        _log(style, f"  LIME done: {len(df_lime_cls)} rows in {_fmt_s(time.perf_counter()-t0)}")

    return df_lime_cls, payloads


def run_pipeline_for_classes(
    target_classes: Sequence[str],
    sel_df: pd.DataFrame,
    model: Any,
    class_names: Sequence[str],
    max_examples_per_class: int | None = None,
    plot: bool = False,
    progress: bool = True,
    verbose: bool = True,
    log_style: Optional[str] = None,
) -> Tuple[Dict[str, Dict[int, Dict]], pd.DataFrame]:
    """Run LIME for multiple classes. Returns (all_payloads, df_all)."""
    style = _resolve_log_style(log_style, verbose)
    debug = style == "debug"

    all_payloads: Dict[str, Dict[int, Dict]] = {}
    all_dfs: list[pd.DataFrame] = []

    classes = list(target_classes)
    cls_iter = tqdm(classes, desc="Classes (LIME)") if progress else classes

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
            log_style=log_style,
        )

        all_payloads[cls] = payloads_cls
        all_dfs.append(df_cls)

        dt = time.perf_counter() - t_cls
        times.append(dt)

        if debug:
            avg = sum(times) / len(times)
            remaining = (len(classes) - i) * avg
            _log(style, f"— Progress: {i}/{len(classes)} classes | ETA ~ {_fmt_s(remaining)}")

    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if style == "showcase":
        _log(style, f"LIME pipeline complete: {len(classes)} classes in {_fmt_s(time.perf_counter()-t_start)}")
    elif debug:
        _log(style, f"\n LIME pipeline complete in {_fmt_s(time.perf_counter()-t_start)}")

    return all_payloads, df_all


# ---------------------------------------------------------------------
# Fused (LIME + TimeSHAP + fusion) pipeline
# ---------------------------------------------------------------------
def run_fused_for_one_class(
    class_name: str,
    sel_df: pd.DataFrame,
    model: Any,
    class_names: Sequence[str],
    max_examples: Optional[int] = None,
    plot: bool = False,
    progress: bool = False,
    verbose: bool = False,
    show_record: bool = False,
    log_style: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict]]:
    """Run LIME + TimeSHAP + fusion for a single class.

    Returns:
        df_lime_cls: LIME outputs (rows = ECGs)
        df_ts_cls: TimeSHAP outputs (rows = ECGs)
        fused_payloads: dict[val_idx -> fused payload]
    """
    cfg = get_explainer_config(class_name)
    t0 = time.perf_counter()

    style = _resolve_log_style(log_style, verbose)
    debug = style == "debug"

    sel_df_cls = _subset_sel_df_for_class(sel_df, class_name, max_examples)

    if debug:
        _log(style, (
            f"\nFUSED class={class_name} | n={len(sel_df_cls)} | win={cfg.window_sec}s "
            f"| mE={cfg.m_event} | mF={cfg.m_feat} | topkE={cfg.topk_events}"
        ))

    # ---- LIME ----
    t_l = time.perf_counter()
    df_lime_cls = run_lime_for_one_class_from_sel(
        sel_df_cls,
        class_name,
        model=model,
        class_names=class_names,
        window_sec=cfg.window_sec,
        m_event=cfg.m_event,
        m_feat=cfg.m_feat,
        topk_events=cfg.topk_events,
        explain_class="force",
        mode=cfg.mode,
        rng=cfg.rng,
        params=cfg.params,
    )
    dt_lime = time.perf_counter() - t_l
    if debug:
        _log(style, f"  LIME done: {len(df_lime_cls)} rows in {_fmt_s(dt_lime)}")

    # ---- TimeSHAP ----
    t_t = time.perf_counter()
    df_ts_cls = run_timeshap_for_one_class_from_sel(
        sel_df_cls,
        class_name,
        model=model,
        class_names=class_names,
        window_sec=cfg.window_sec,
        m_event=cfg.m_event,
        m_feat=cfg.m_feat,
        topk_events=cfg.topk_events,
        explain_class="force",
        mode=cfg.mode,
        rng=cfg.rng,
        link=cfg.timeshap_link,
        params=cfg.params,
    )
    dt_ts = time.perf_counter() - t_t
    if debug:
        _log(style, f"  TimeSHAP done: {len(df_ts_cls)} rows in {_fmt_s(dt_ts)}")

    # ---- fuse common val_idx ----
    if "val_idx" not in df_lime_cls.columns or "val_idx" not in df_ts_cls.columns:
        raise KeyError("Expected 'val_idx' column in both LIME and TimeSHAP dataframes.")

    common_val_idx = sorted(
        set(df_lime_cls["val_idx"].astype(int)) & set(df_ts_cls["val_idx"].astype(int))
    )

    if debug:
        _log(style, f"  Fusing: {len(common_val_idx)} common records")

    fused_payloads: Dict[int, Dict] = {}

    it = common_val_idx
    if progress:
        it = tqdm(common_val_idx, desc=f"Fuse {class_name}", leave=False)

    # class-specific method weights (optional)
    method_weights = FUSION_WEIGHTS_BY_SNOMED.get(str(class_name), (0.5, 0.5))

    for val_i in it:
        row_L = df_lime_cls.loc[df_lime_cls["val_idx"] == val_i].iloc[0]
        row_T = df_ts_cls.loc[df_ts_cls["val_idx"] == val_i].iloc[0]

        payload_L = payload_from_lime_row(row_L, label_for_title=row_L.get("group_class", class_name))
        payload_T = payload_from_timeshap_row(row_T, label_for_title=row_T.get("group_class", class_name))

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

    dt_total = time.perf_counter() - t0

    if style == "showcase":
        _log(style, (
            f"FUSED {class_name} | n={len(sel_df_cls)} | "
            f"win={cfg.window_sec}s mE={cfg.m_event} mF={cfg.m_feat} topkE={cfg.topk_events} | "
            f"LIME={_fmt_s(dt_lime)} TS={_fmt_s(dt_ts)} | "
            f"fused={len(common_val_idx)} | total={_fmt_s(dt_total)}"
        ))
    elif debug:
        _log(style, f"  Class {class_name} total: {_fmt_s(dt_total)}")

    return df_lime_cls, df_ts_cls, fused_payloads


def run_fused_pipeline_for_classes(
    target_classes: Sequence[str],
    sel_df: pd.DataFrame,
    model: Any,
    class_names: Sequence[str],
    max_examples_per_class: Optional[int] = None,
    plot: bool = False,
    progress: bool = False,
    verbose: bool = False,
    show_record: bool = False,
    log_style: Optional[str] = None,
) -> Tuple[Dict[str, Dict[int, Dict]], pd.DataFrame, pd.DataFrame]:
    """High-level fused pipeline for multiple classes.

    Returns:
        all_fused_payloads: dict[class_code -> dict[val_idx -> fused payload]]
        df_lime_all: concatenated LIME dataframe
        df_ts_all: concatenated TimeSHAP dataframe
    """
    style = _resolve_log_style(log_style, verbose)
    debug = style == "debug"

    all_fused_payloads: Dict[str, Dict[int, Dict]] = {}
    all_lime: list[pd.DataFrame] = []
    all_ts: list[pd.DataFrame] = []

    classes = list(target_classes)
    cls_iter = tqdm(classes, desc="Classes (Fused)") if progress else classes

    t_start = time.perf_counter()
    times: list[float] = []

    for i, cls in enumerate(cls_iter, start=1):
        t_cls = time.perf_counter()

        if debug:
            _log(style, f"\n=== [{i}/{len(classes)}] Processing class: {cls} ===")

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
            log_style=log_style,
        )

        all_fused_payloads[cls] = fused_payloads_cls
        all_lime.append(df_lime_cls)
        all_ts.append(df_ts_cls)

        dt = time.perf_counter() - t_cls
        times.append(dt)

        if debug:
            avg = sum(times) / len(times)
            remaining = (len(classes) - i) * avg
            _log(style, f"— Progress: {i}/{len(classes)} classes | ETA ~ {_fmt_s(remaining)}")

    df_lime_all = pd.concat(all_lime, ignore_index=True) if all_lime else pd.DataFrame()
    df_ts_all = pd.concat(all_ts, ignore_index=True) if all_ts else pd.DataFrame()

    if style == "showcase":
        _log(style, f"Fused pipeline complete: {len(classes)} classes in {_fmt_s(time.perf_counter()-t_start)}")
    elif debug:
        _log(style, f"\n All classes complete in {_fmt_s(time.perf_counter()-t_start)}")

    return all_fused_payloads, df_lime_all, df_ts_all
