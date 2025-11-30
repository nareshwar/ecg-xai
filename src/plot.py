import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import ecg_plot
import numpy as np

from preprocessing import infer_fs_from_header
from payload import _infer_fs_from_header_lines
from utils import load_physionet_data


LIMB   = ["I","II","III","aVR","aVL","aVF"]
VLEADS = ["V1","V2","V3","V4","V5","V6"]
DEFAULT_LEADS = LIMB + VLEADS
LEAD_COLOR = {
    "I":"#2F5597","II":"#E69138","III":"#38761D",
    "aVR":"#CC0000","aVL":"#674EA7","aVF":"#B45F06",
    "V1":"#A64D79","V2":"#6AA84F","V3":"#BF9000",
    "V4":"#3D85C6","V5":"#45818E","V6":"#741B47"
}


def _read_header_lines(hea_path):
    if not os.path.exists(hea_path):
        return []
    try:
        with open(hea_path, "r", encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    except UnicodeDecodeError:
        with open(hea_path, "r", encoding="latin-1") as f:
            return [ln.rstrip("\n") for ln in f]


def _bands_from_label_centers(ax):
    centers = {}
    for t in ax.texts:
        name = t.get_text().strip()
        if name in DEFAULT_LEADS:
            _, y = t.get_position()
            centers[name] = float(y)

    def _from(order):
        ys = [centers[n] for n in order]
        mids = [(ys[i] + ys[i+1]) / 2 for i in range(len(ys)-1)]
        top = ys[0] + (ys[0] - ys[1]) / 2
        bottom = ys[-1] - (ys[-2] - ys[-1]) / 2
        edges = [top] + mids + [bottom]
        return {order[i]: (edges[i], edges[i+1]) for i in range(len(order))}

    bands = {}
    if all(n in centers for n in LIMB):
        bands.update(_from(LIMB))
    if all(n in centers for n in VLEADS):
        bands.update(_from(VLEADS))

    if len(bands) < 12:
        ymin, ymax = ax.get_ylim()
        h = (ymax - ymin) / 12.0
        for i, n in enumerate(DEFAULT_LEADS):
            ymid = ymax - (i + 0.5) * h
            pad = 0.48 * h
            bands[n] = (ymid + pad, ymid - pad)
    return bands


def _clip(span, dom):
    s, e = span
    a, b = dom
    s, e = max(s, a), min(e, b)
    return None if e <= s else (s, e)


def _choose_top5_from_scores(lead_scores, perlead_spans):
    if isinstance(lead_scores, dict) and lead_scores:
        return [
            k for k, _ in sorted(lead_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ]
    return [k for k, _ in sorted(perlead_spans.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]]

import ecg_plot

def _choose_topk_from_scores(lead_scores, perlead_spans, topk: int):
    """Fallback: choose top-k leads using scores or span counts."""
    if isinstance(lead_scores, dict) and lead_scores:
        ordered = sorted(lead_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in ordered[:topk]]
    # if no scores, just use number of spans
    ordered = sorted(
        perlead_spans.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    return [k for k, _ in ordered[:topk]]

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import ecg_plot

# column layout
LIMB   = ["I", "II", "III", "aVR", "aVL", "aVF"]
VLEADS = ["V1", "V2", "V3", "V4", "V5", "V6"]
DEFAULT_LEADS = LIMB + VLEADS


def _choose_topk_from_scores(lead_scores, perlead_spans, topk: int):
    if isinstance(lead_scores, dict) and lead_scores:
        ordered = sorted(lead_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in ordered[:topk]]
    ordered = sorted(perlead_spans.items(), key=lambda kv: len(kv[1]), reverse=True)
    return [k for k, _ in ordered[:topk]]


def plot_from_payload(
    payload: dict,
    *,
    topk: int = 5,
    show_all_leads: bool = False,
    alpha_fill: float = 0.40,
    alpha_min: float = 0.18,
    edge_alpha: float = 0.8,
    tick_major_s: float = 1.0,
    tick_minor_s: float = 0.2,
    figsize=(20, 6),
):
    """
    Plot ECG + shaded explanation rectangles from a fused/LIME/TimeSHAP payload.

    Assumes standard 2×6 layout:
      - I, II, III, aVR, aVL, aVF in left column
      - V1–V6 in right column

    `perlead_spans` times are in seconds 0..page_seconds for *each* lead.
    We therefore shift V-lead spans by +page_seconds when plotting.
    """
    mat_path = payload["mat_path"]
    hea_path = os.path.splitext(mat_path)[0] + ".hea"

    # ---- load ECG ----
    data_12xT, _ = load_physionet_data(mat_path)
    header_lines = _read_header_lines(hea_path)
    fs = _infer_fs_from_header_lines(header_lines, default=500.0)

    page_sec = float(payload.get("page_seconds", data_12xT.shape[1] / fs))
    K = int(round(page_sec * fs))
    if K > 0 and data_12xT.shape[1] > K:
        data_12xT = data_12xT[:, :K]

    # ---- base ECG plot ----
    plt.figure(figsize=figsize)
    ecg_plot.plot(data_12xT / 1000.0, sample_rate=fs, title="")
    ax = plt.gca()
    fig = plt.gcf()

    mticker.Locator.MAXTICKS = max(mticker.Locator.MAXTICKS, 10000)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_major_s))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(tick_minor_s))

    bands = _bands_from_label_centers(ax)  # {lead: (y_top, y_bot)}
    perlead_spans = payload["perlead_spans"]
    lead_scores = payload.get("lead_scores", {})

    # which leads?
    if show_all_leads:
        leads_to_show = [L for L in DEFAULT_LEADS if L in perlead_spans]
    else:
        if payload.get("top5_leads"):
            leads_to_show = list(payload["top5_leads"])
        else:
            leads_to_show = _choose_topk_from_scores(
                lead_scores, perlead_spans, topk=topk
            )
        if topk is not None:
            leads_to_show = leads_to_show[:topk]

    x0, x1 = ax.get_xlim()
    # ecg_plot uses two equal-width columns across the x-axis
    col_width = (x1 - x0) / 2.0

    handles = []

    for L in leads_to_show:
        if L not in perlead_spans or L not in bands:
            continue

        spans = perlead_spans[L]
        if not spans:
            continue

        y_top, y_bot = bands[L]
        color = LEAD_COLOR.get(L, "#888888")

        # limb leads left column, V-leads right column
        if L in LIMB:
            x_shift = 0.0
        elif L in VLEADS:
            # shift whole window to second column
            x_shift = col_width
        else:
            x_shift = 0.0  # fallback

        # scale alpha inside this lead by max |weight|
        weights = np.array(
            [abs(s[2]) if len(s) >= 3 else 1.0 for s in spans],
            dtype=float,
        )
        wmax = weights.max() if weights.size else None

        for (s, e, *rest) in spans:
            s = float(s) + x_shift
            e = float(e) + x_shift

            if e <= x0 or s >= x1:
                continue

            s_clip = max(s, x0)
            e_clip = min(e, x1)
            if e_clip <= s_clip:
                continue

            w = abs(rest[0]) if rest else 1.0
            if wmax and wmax > 0:
                a = alpha_min + (alpha_fill - alpha_min) * min(w / wmax, 1.0)
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

        handles.append(
            Patch(facecolor=color, alpha=alpha_fill, edgecolor="none", label=L)
        )

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
