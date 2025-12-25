import os
import json
import pandas as pd

DEFAULT_LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def _choose_top5_from_scores(lead_scores, perlead_spans):
    if isinstance(lead_scores, dict) and lead_scores:
        return [
            k for k, _ in sorted(lead_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ]
    return [k for k, _ in sorted(perlead_spans.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]]


def _lead_names_from_row(row):
    ln = row.get("lead_names", None)
    if isinstance(ln, str) and "," in ln:
        return [x.strip() for x in ln.split(",")]
    if isinstance(ln, str) and ln:
        return [x.strip() for x in ln.split()]
    return DEFAULT_LEADS[:]

def _read_header_lines(hea_path: str):
    if not os.path.exists(hea_path): return []
    try:
        with open(hea_path, "r", encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    except UnicodeDecodeError:
        with open(hea_path, "r", encoding="latin-1") as f:
            return [ln.rstrip("\n") for ln in f]

def _infer_fs_from_header_lines(header_lines, default=500.0):
    if not header_lines: return float(default)
    toks = header_lines[0].split()
    for tok in toks:
        try:
            x = float(tok)
        except Exception:
            continue
        if 50 <= x <= 2000:
            return float(x)
    return float(default)

def payload_from_lime_row(row, *, label_for_title=""):
    """
    Convert a LIME row into a plotting payload.
    """
    mat_path = row["mat_path"] if "mat_path" in row else os.path.splitext(str(row["filename"]))[0] + ".mat"

    fs = float(row["fs"]) if "fs" in row and not pd.isna(row["fs"]) else 500.0

    segs = json.loads(row["segments_json"])
    T = max(int(t) for (_, t) in segs) if segs else int(round(10 * fs))
    page_seconds = T / fs

    spans_top5 = json.loads(row["perlead_spans_top5_json"])
    lead_names = _lead_names_from_row(row)

    perlead_spans_sec = {}
    lead_scores = {}
    for k, lst in spans_top5.items():
        j = int(k)
        L = lead_names[j] if j < len(lead_names) else f"ch{j}"
        spans_sec = [(float(s), float(t), float(w)) for (s, t, w) in lst]
        perlead_spans_sec[L] = spans_sec
        lead_scores[L] = sum(abs(w) for (_, _, w) in spans_sec)

    if "top5_lead_idx_json" in row:
        top5 = [lead_names[j] for j in json.loads(row["top5_lead_idx_json"])]
    else:
        top5 = _choose_top5_from_scores(lead_scores, perlead_spans_sec)

    return {
        "mat_path": mat_path,
        "target_label": label_for_title,
        "method_label": "LIME",
        "page_seconds": page_seconds,
        "perlead_spans": perlead_spans_sec,
        "lead_scores": lead_scores,
        "top5_leads": top5,
    }

def payload_from_timeshap_row(row, *, label_for_title: str = "") -> dict:
    """
    Build a plotting payload from a TimeSHAP row.

    Assumes row has:
      - 'mat_path' (or 'filename')
      - 'fs'
      - 'segments_json'
      - 'perlead_timeshap_top5_json'
      - 'top5_lead_idx_json'
      - 'lead_names'
    """
    mat_path = (
        row["mat_path"]
        if "mat_path" in row
        else os.path.splitext(str(row["filename"]))[0] + ".mat"
    )
    hea_path = os.path.splitext(mat_path)[0] + ".hea"

    if "fs" in row and not pd.isna(row["fs"]):
        fs = float(row["fs"])
    else:
        header_lines = _read_header_lines(hea_path)
        fs = _infer_fs_from_header_lines(header_lines, default=500.0)

    segs = json.loads(row["segments_json"])
    T = max(int(t) for (_, t) in segs) if segs else int(round(10 * fs))
    page_seconds = T / fs

    spans_top5 = json.loads(row["perlead_timeshap_top5_json"])
    lead_names = _lead_names_from_row(row)

    perlead_spans_sec = {}
    lead_scores = {}
    for k, lst in spans_top5.items():
        j = int(k)
        L = lead_names[j] if j < len(lead_names) else f"ch{j}"
        spans_sec = [(float(s), float(t), float(w)) for (s, t, w) in lst]
        perlead_spans_sec[L] = spans_sec
        lead_scores[L] = sum(abs(w) for (_, _, w) in lst)

    if "top5_lead_idx_json" in row:
        top5 = [
            lead_names[j]
            for j in json.loads(row["top5_lead_idx_json"])
        ]
    else:
        top5 = _choose_top5_from_scores(lead_scores, perlead_spans_sec)

    return {
        "mat_path": mat_path,
        "target_label": label_for_title,
        "method_label": "TimeSHAP",
        "page_seconds": page_seconds,
        "perlead_spans": perlead_spans_sec,
        "lead_scores": lead_scores,
        "top5_leads": top5,
    }
