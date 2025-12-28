"""
ecgxai package

Top-level public API for the ECG explainability toolkit.

This file re-exports the most commonly used functions/classes so callers can do:

    import ecgxai as ex

    x = ex.load_mat_TF("record.mat")
    payloads, df = ex.run_pipeline_for_classes(...)

Design notes:
- Keep imports here reasonably lightweight. If importing this package ever feels
    slow, can move heavier bits (plotting / training-time utilities) behind
    lazy imports.
- Avoid wildcard imports: explicit exports make the public API clear.
"""

from __future__ import annotations

# Semantic version for packaging / reproducibility
__version__ = "0.1.0"

# ---- Core config / constants ----
from .config import MAXLEN

# ---- Data loading / preprocessing ----
from .preprocessing import load_mat_TF, preprocess_for_model

# ---- Explainability methods ----
from .ecg_lime import run_lime_for_one_class_from_sel
from .ecg_timeshap import run_timeshap_for_one_class_from_sel
from .explainer import (
    run_explain_for_one_class,
    run_pipeline_for_classes,
    run_fused_for_one_class,
    run_fused_pipeline_for_classes,
)

# ---- Payload / plotting ----
from .payload import payload_from_lime_row, payload_from_timeshap_row
from .plot import plot_from_payload

# ---- Evaluation ----
from .eval import REGISTRY, evaluate_explanation, evaluate_all_payloads

# ---- Persistence / utilities ----
from .utils import load_run, save_run

# ---- Optional modules ----
# Expose module namespace without wildcard exporting everything
from . import segments

__all__ = [
    "__version__",

    # config
    "MAXLEN",

    # preprocessing
    "load_mat_TF",
    "preprocess_for_model",

    # explainers
    "run_lime_for_one_class_from_sel",
    "run_timeshap_for_one_class_from_sel",
    "run_explain_for_one_class",
    "run_pipeline_for_classes",
    "run_fused_for_one_class",
    "run_fused_pipeline_for_classes",

    # payload/plotting
    "payload_from_lime_row",
    "payload_from_timeshap_row",
    "plot_from_payload",

    # evaluation
    "evaluate_explanation",
    "evaluate_all_payloads",
    "REGISTRY",

    # utils
    "save_run",
    "load_run",

    # segments module
    "segments",
]
