"""
ecgxai.plot

Plot ECGs with shaded explanation spans from a payload produced by:
- payload_from_lime_row()
- payload_from_timeshap_row()
- fuse_lime_timeshap_payload()

Payload contract:
    payload = {
        "mat_path": str,
        "page_seconds": float,
        "perlead_spans": {lead: [(start_sec, end_sec, weight), ...], ...},
        "lead_scores": {lead: float, ...} (optional),
        "top5_leads": [lead,...] (optional),
        "method_label": str (optional),
        "target_label": str (optional),
    }

Plot convention:
- 2×6 layout produced by ecg_plot:
    left column: limb leads I, II, III, aVR, aVL, aVF
    right column: precordial leads V1..V6
- Spans for V leads are shifted by +column_width on the x-axis.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import ecg_plot

from .preprocessing import infer_fs_from_header
from .utils import load_physionet_data


# ---------------------------------------------------------------------
# Lead layout + colors
# ---------------------------------------------------------------------
LIMB: List[str] = ["I", "II", "III", "aVR", "aVL", "aVF"]
VLEADS: List[str] = ["V1", "V2", "V3", "V4", "V5", "V6"]
DEFAULT_LEADS: List[str] = LIMB + VLEADS

LEAD_COLOR: Dict[str, str] = {
    "I": "#2F5597",   "II": "#E69138", "III": "#38761D",
    "aVR": "#CC0000", "aVL": "#674EA7", "aVF": "#B45F06",
    "V1": "#A64D79",  "V2": "#6AA84F",  "V3": "#BF9000",
    "V4": "#3D85C6",  "V5": "#45818E",  "V6": "#741B47",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _read_header_lines(hea_path: str) -> List[str]:
    """Read header lines with robust encoding (used only for debugging)."""
    if not os.path.exists(hea_path):
        return []
    try:
        with open(hea_path, "r", encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    except UnicodeDecodeError:
        with open(hea_path, "r", encoding="latin-1") as f:
            return [ln.rstrip("\n") for ln in f]


def _bands_from_label_centers(ax) -> Dict[str, Tuple[float, float]]:
    """Infer y-bands for each lead from the lead label positions.

    ecg_plot writes lead names as text labels. We parse their y-positions and
    infer top/bottom edges for each band. Fallback to evenly spaced bands if
    labels aren't found.
    """
    centers: Dict[str, float] = {}
    for t in ax.texts:
        name = t.get_text().strip()
        if name in DEFAULT_LEADS:
            _, y = t.get_position()
            centers[name] = float(y)

    def _from(order: Sequence[str]) -> Dict[str, Tuple[float, float]]:
        ys = [centers[n] for n in order]
        mids = [(ys[i] + ys[i + 1]) / 2 for i in range(len(ys) - 1)]
        top = ys[0] + (ys[0] - ys[1]) / 2
        bottom = ys[-1] - (ys[-2] - ys[-1]) / 2
        edges = [top] + mids + [bottom]
        return {order[i]: (edges[i], edges[i + 1]) for i in range(len(order))}

    bands: Dict[str, Tuple[float, float]] = {}
    if all(n in centers for n in LIMB):
        bands.update(_from(LIMB))
    if all(n in centers for n in VLEADS):
        bands.update(_from(VLEADS))

    # fallback: evenly spaced bands (still works)
    if len(bands) < 12:
        ymin, ymax = ax.get_ylim()
        h = (ymax - ymin) / 12.0
        for i, n in enumerate(DEFAULT_LEADS):
            ymid = ymax - (i + 0.5) * h
            pad = 0.48 * h
            bands[n] = (ymid + pad, ymid - pad)

    return bands


def _choose_topk_leads(lead_scores, perlead_spans, topk: int) -> List[str]:
    """Choose top-k leads using provided lead_scores; fallback to number of spans."""
    if isinstance(lead_scores, dict) and lead_scores:
        ordered = sorted(lead_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in ordered[:topk]]

    ordered = sorted(perlead_spans.items(), key=lambda kv: len(kv[1]), reverse=True)
    return [k for k, _ in ordered[:topk]]


# ---------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------
def plot_from_payload(
    payload: dict,
    *,
    topk: int = 5,
    show_all_leads: bool = False,
    alpha_fill: float = 0.40,
    alpha_min: float = 0.18,
    edge_alpha: float = 0.80,
    tick_major_s: float = 1.0,
    tick_minor_s: float = 0.2,
    figsize=(20, 6),
) -> None:
    """Plot ECG with shaded explanation rectangles from a payload.

    Args:
        payload: Payload dict (see module docstring).
        topk: Number of leads to show when payload doesn't specify top5_leads.
        show_all_leads: If True, show spans for all leads present in payload.
        alpha_fill: Max fill alpha for the strongest span within each lead.
        alpha_min: Min fill alpha for weaker spans.
        edge_alpha: Alpha for the vertical edge lines.
        tick_major_s: Major x-axis tick spacing in seconds.
        tick_minor_s: Minor x-axis tick spacing in seconds.
        figsize: Matplotlib figure size.

    Notes:
        The plot is generated by ecg_plot which renders two columns on the same x-axis.
        We shift precordial (V*) spans by +column_width to align with the right column.
    """
    mat_path = payload["mat_path"]
    hea_path = os.path.splitext(mat_path)[0] + ".hea"

    # ---- load ECG ----
    data_12xT, _ = load_physionet_data(mat_path)  # shape (12, T_raw)
    fs = float(infer_fs_from_header(hea_path, default=500.0))

    page_sec = float(payload.get("page_seconds", data_12xT.shape[1] / fs))
    K = int(round(page_sec * fs))
    if K > 0 and data_12xT.shape[1] > K:
        data_12xT = data_12xT[:, :K]

    # ---- base ECG plot ----
    plt.figure(figsize=figsize)
    ecg_plot.plot(data_12xT / 1000.0, sample_rate=fs, title="")
    ax = plt.gca()
    fig = plt.gcf()

    # avoid MAXTICKS warning with long ECGs
    mticker.Locator.MAXTICKS = max(mticker.Locator.MAXTICKS, 10000)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_major_s))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(tick_minor_s))

    bands = _bands_from_label_centers(ax)
    perlead_spans = payload.get("perlead_spans", {}) or {}
    lead_scores = payload.get("lead_scores", {}) or {}

    # ---- choose leads ----
    if show_all_leads:
        leads_to_show = [L for L in DEFAULT_LEADS if L in perlead_spans]
    else:
        if payload.get("top5_leads"):
            leads_to_show = list(payload["top5_leads"])
        else:
            leads_to_show = _choose_topk_leads(lead_scores, perlead_spans, topk=topk)
        leads_to_show = leads_to_show[:topk]

    x0, x1 = ax.get_xlim()
    col_width = (x1 - x0) / 2.0  # ecg_plot uses two equal-width columns

    handles: List[Patch] = []

    for L in leads_to_show:
        if L not in perlead_spans or L not in bands:
            continue

        spans = perlead_spans[L]
        if not spans:
            continue

        y_top, y_bot = bands[L]
        color = LEAD_COLOR.get(L, "#888888")

        # limb leads in left column, V leads in right column
        x_shift = col_width if L in VLEADS else 0.0

        # per-lead alpha scaling: strongest |w| => alpha_fill
        weights = np.array([abs(float(s[2])) for s in spans], dtype=float)
        wmax = float(weights.max()) if weights.size else 0.0

        for (s, e, w) in spans:
            s = float(s) + x_shift
            e = float(e) + x_shift
            if e <= x0 or s >= x1:
                continue

            s_clip = max(s, x0)
            e_clip = min(e, x1)
            if e_clip <= s_clip:
                continue

            ww = abs(float(w))
            if wmax > 0:
                a = alpha_min + (alpha_fill - alpha_min) * min(ww / wmax, 1.0)
            else:
                a = alpha_fill

            ax.fill_between(
                [s_clip, e_clip],
                y_top,
                y_bot,
                facecolor=color,
                alpha=a,
                edgecolor="none",
                zorder=0.6,
            )
            ax.vlines(
                [s_clip, e_clip],
                y_bot,
                y_top,
                colors=color,
                alpha=edge_alpha,
                linewidth=0.9,
                zorder=0.8,
            )

        handles.append(Patch(facecolor=color, alpha=alpha_fill, edgecolor="none", label=L))

    # ---- title & legend ----
    base = os.path.basename(mat_path)
    method_label = payload.get("method_label", "LIME")
    target_label = payload.get("target_label", "")
    title = f"ECG + {method_label} — {base}"
    if target_label:
        title += f" [{target_label}]"
    fig.suptitle(title, y=0.98, fontsize=14, fontweight="bold")

    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=min(6, len(handles)),
            framealpha=0.95,
            bbox_to_anchor=(0.5, 0.965),
        )

    fig.tight_layout()
    plt.show()
