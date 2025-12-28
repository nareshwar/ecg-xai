"""
ecgxai.payload

Conversion helpers to turn a single LIME/TimeSHAP dataframe row into a
"plotting payload" consumed by plot_from_payload().

Payload schema (returned dict):
    {
      "mat_path": str,
      "target_label": str,
      "method_label": str,          # "LIME" | "TimeSHAP" | (others)
      "page_seconds": float,        # duration represented in the plot
      "perlead_spans": {
          "II": [(start_sec, end_sec, weight), ...],
          "V1": [...],
          ...
      },
      "lead_scores": {"II": float, "V1": float, ...},   # per-lead importance
      "top5_leads": ["II","V1",...],                    # leads to display
    }

Row expectations:
- LIME row should have:
    - filename or mat_path
    - fs (optional)
    - segments_json (list of [start_sample, end_sample])
    - perlead_spans_top5_json (dict[str lead_idx] -> list[(s_sec,e_sec,w)])
    - top5_lead_idx_json (optional)
    - lead_names (optional)
- TimeSHAP row should have:
    - filename or mat_path
    - fs (optional)
    - segments_json
    - perlead_timeshap_top5_json
    - top5_lead_idx_json (optional)
    - lead_names (optional)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, Union

import pandas as pd

DEFAULT_LEADS: List[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

Span = Tuple[float, float, float]             # (start_sec, end_sec, weight)
PerLeadSpans = Dict[str, List[Span]]
LeadScores = Dict[str, float]
Payload = Dict[str, Any]


# Prefer your shared preprocessing inference if present; fallback is local parsing.
try:
    from .preprocessing import infer_fs_from_header as _infer_fs_from_header  # type: ignore
except Exception:  # pragma: no cover
    _infer_fs_from_header = None  # type: ignore


def _json_loads(value: Any, *, label: str) -> Any:
    """Load JSON from a cell value with a helpful error."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception as e:
        raise ValueError(f"Failed to json.loads({label}). Value starts with: {str(value)[:120]!r}") from e


def _lead_names_from_row(row: Mapping[str, Any]) -> List[str]:
    """Read lead names from row['lead_names'] or return DEFAULT_LEADS."""
    ln = row.get("lead_names", None)
    if isinstance(ln, str) and "," in ln:
        return [x.strip() for x in ln.split(",") if x.strip()]
    if isinstance(ln, str) and ln.strip():
        return [x.strip() for x in ln.split() if x.strip()]
    return DEFAULT_LEADS[:]  # copy


def _mat_path_from_row(row: Mapping[str, Any]) -> str:
    """Resolve .mat path from row, accepting either mat_path or filename."""
    if "mat_path" in row and row["mat_path"]:
        return str(row["mat_path"])
    if "filename" in row and row["filename"]:
        base, _ = os.path.splitext(str(row["filename"]))
        return base + ".mat"
    raise KeyError("Row must contain either 'mat_path' or 'filename'.")


def _infer_fs(mat_path: str, row: Mapping[str, Any], default: float = 500.0) -> float:
    """Infer sampling frequency, preferring row['fs'], then header, then default."""
    if "fs" in row and row["fs"] is not None and not pd.isna(row["fs"]):
        try:
            return float(row["fs"])
        except Exception:
            pass

    hea_path = os.path.splitext(mat_path)[0] + ".hea"
    if _infer_fs_from_header is not None and os.path.exists(hea_path):
        try:
            return float(_infer_fs_from_header(hea_path, default=default))
        except Exception:
            pass

    # fallback: best-effort scan first header line
    if os.path.exists(hea_path):
        try:
            with open(hea_path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
        except UnicodeDecodeError:
            with open(hea_path, "r", encoding="latin-1") as f:
                first = f.readline().strip()

        for tok in first.split():
            try:
                x = float(tok)
            except Exception:
                continue
            if 50 <= x <= 2000:
                return float(x)

    return float(default)


def _page_seconds_from_segments(segments_json: Any, fs: float, *, default_seconds: float = 10.0) -> float:
    """Compute plot duration from sample-index segments_json."""
    segs = _json_loads(segments_json, label="segments_json")
    if not segs:
        return float(default_seconds)

    # segments are expected as [[s0,e0],[s1,e1],...]
    try:
        T = max(int(t) for (_, t) in segs)
        return float(T) / float(fs)
    except Exception:
        return float(default_seconds)


def _choose_top_leads(
    lead_scores: LeadScores,
    perlead_spans: PerLeadSpans,
    k: int = 5,
) -> List[str]:
    """Pick top-k leads by lead_scores; fallback to most spans."""
    if isinstance(lead_scores, dict) and lead_scores:
        return [L for (L, _) in sorted(lead_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
    return [L for (L, _) in sorted(perlead_spans.items(), key=lambda kv: len(kv[1]), reverse=True)[:k]]


def _build_perlead_from_indexed_spans(
    spans_by_idx: Mapping[str, Iterable[Iterable[Any]]],
    lead_names: Sequence[str],
) -> Tuple[PerLeadSpans, LeadScores]:
    """Convert dict[idx -> spans] into dict[lead_name -> spans] + lead_scores."""
    perlead: PerLeadSpans = {}
    scores: LeadScores = {}

    for k, lst in spans_by_idx.items():
        j = int(k)
        L = lead_names[j] if j < len(lead_names) else f"ch{j}"

        spans_sec: List[Span] = []
        for s, t, w in lst:
            spans_sec.append((float(s), float(t), float(w)))

        perlead[L] = spans_sec
        scores[L] = float(sum(abs(w) for (_, _, w) in spans_sec))

    return perlead, scores


def payload_from_lime_row(row: Mapping[str, Any], *, label_for_title: str = "") -> Payload:
    """Convert one LIME dataframe row into a plotting payload."""
    mat_path = _mat_path_from_row(row)
    fs = _infer_fs(mat_path, row, default=500.0)

    page_seconds = _page_seconds_from_segments(row.get("segments_json"), fs, default_seconds=10.0)

    spans_top5 = _json_loads(row.get("perlead_spans_top5_json"), label="perlead_spans_top5_json") or {}
    lead_names = _lead_names_from_row(row)
    lead_names = [L for L in DEFAULT_LEADS if L in lead_names] + [L for L in lead_names if L not in DEFAULT_LEADS]

    perlead_spans_sec, lead_scores = _build_perlead_from_indexed_spans(spans_top5, lead_names)

    if "top5_lead_idx_json" in row and row.get("top5_lead_idx_json") is not None and not pd.isna(row.get("top5_lead_idx_json")):
        idxs = _json_loads(row.get("top5_lead_idx_json"), label="top5_lead_idx_json") or []
        top5 = [lead_names[int(j)] for j in idxs if int(j) < len(lead_names)]
    else:
        top5 = _choose_top_leads(lead_scores, perlead_spans_sec, k=5)

    return {
        "mat_path": mat_path,
        "target_label": label_for_title,
        "method_label": "LIME",
        "page_seconds": float(page_seconds),
        "perlead_spans": perlead_spans_sec,
        "lead_scores": lead_scores,
        "top5_leads": top5,
    }


def payload_from_timeshap_row(row: Mapping[str, Any], *, label_for_title: str = "") -> Payload:
    """Convert one TimeSHAP dataframe row into a plotting payload."""
    mat_path = _mat_path_from_row(row)
    fs = _infer_fs(mat_path, row, default=500.0)

    page_seconds = _page_seconds_from_segments(row.get("segments_json"), fs, default_seconds=10.0)

    spans_top5 = _json_loads(row.get("perlead_timeshap_top5_json"), label="perlead_timeshap_top5_json") or {}
    lead_names = _lead_names_from_row(row)

    perlead_spans_sec, lead_scores = _build_perlead_from_indexed_spans(spans_top5, lead_names)

    if "top5_lead_idx_json" in row and row.get("top5_lead_idx_json") is not None and not pd.isna(row.get("top5_lead_idx_json")):
        idxs = _json_loads(row.get("top5_lead_idx_json"), label="top5_lead_idx_json") or []
        top5 = [lead_names[int(j)] for j in idxs if int(j) < len(lead_names)]
    else:
        top5 = _choose_top_leads(lead_scores, perlead_spans_sec, k=5)

    return {
        "mat_path": mat_path,
        "target_label": label_for_title,
        "method_label": "TimeSHAP",
        "page_seconds": float(page_seconds),
        "perlead_spans": perlead_spans_sec,
        "lead_scores": lead_scores,
        "top5_leads": top5,
    }
