"""
ecgxai.fusion

Fuse LIME + TimeSHAP per-lead spans into a single payload suitable for plotting.

Both input payloads are expected to include:
- "mat_path": str
- "perlead_spans": dict[str, list[(start_sec, end_sec, weight)]]
- "lead_scores": dict[str, float]   (optional but recommended)

Fusion outputs a payload with:
- "perlead_spans": fused spans (per lead) with weights in a normalized scale
- "lead_scores": fused lead-level scores (weighted blend of input lead_scores)
- "top5_leads": leads ranked by fused *mass* (duration × |weight|) by default
- "meta": diagnostic metadata about agreement vs solo evidence and settings

Key concepts:
- "agree": overlapping regions from both methods (same-sign by default)
- "conflict": overlapping regions but opposite sign (handled by sign_policy)
- "solo_L"/"solo_T": regions present only in one method (kept with beta scaling)

Notes:
- This module normalizes per-lead weights by each method’s per-lead max|w|
  before fusing. That keeps fusion scale stable across ECGs.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# -----------------------------
# Type aliases
# -----------------------------
Span = Tuple[float, float, float]            # (start_sec, end_sec, weight)
SpanSrc = Tuple[float, float, float, str]    # (start_sec, end_sec, weight, src)
PerLeadSpans = Dict[str, List[Span]]
LeadScores = Dict[str, float]
Payload = Dict[str, object]


def _interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Return overlap interval between [a0,a1] and [b0,b1], else None."""
    s = max(float(a[0]), float(b[0]))
    e = min(float(a[1]), float(b[1]))
    return (s, e) if e > s else None


def _normalize_nonneg_scores(d: Mapping[str, float]) -> Dict[str, float]:
    """Normalize non-negative scores so they sum to 1.0 (or return all zeros)."""
    keys = list(d.keys())
    vals = np.array([max(0.0, float(d[k])) for k in keys], dtype=float)
    s = float(vals.sum())
    if s <= 0:
        return {k: 0.0 for k in keys}
    return {k: float(max(0.0, d[k])) / s for k in keys}


def _perlead_max_abs(perlead_spans: PerLeadSpans) -> Dict[str, float]:
    """Max absolute span weight per lead (used for per-lead normalization)."""
    out: Dict[str, float] = {}
    for L, spans in perlead_spans.items():
        ws = [abs(float(w)) for (_, _, w) in spans]
        out[L] = max(ws) if ws else 0.0
    return out


def _lead_mass_from_spans(perlead_spans: PerLeadSpans) -> Dict[str, float]:
    """Compute mass per lead: Σ duration × |weight|."""
    masses: Dict[str, float] = {}
    for L, spans in perlead_spans.items():
        masses[L] = float(sum((float(e) - float(s)) * abs(float(w)) for (s, e, w) in spans))
    return masses


def _lead_scores_fallback(perlead_spans: PerLeadSpans) -> Dict[str, float]:
    """Fallback lead_scores if payload doesn't provide them (normalized masses)."""
    masses = _lead_mass_from_spans(perlead_spans)
    return _normalize_nonneg_scores(masses)


def _merge_adjacent(
    spans: List[SpanSrc],
    *,
    gap_tol: float = 0.025,
    weight_tol: float = 0.15,
) -> List[SpanSrc]:
    """Merge spans if close in time and weights are similar.

    Two spans merge if:
      - gap between them <= gap_tol
      - |w1 - w2| <= weight_tol × mean(|w1|,|w2|)

    Source label rule:
      - stays "agree" only if both are "agree"
      - otherwise keep the source of the span with larger |w|
    """
    if not spans:
        return spans

    spans = sorted(spans, key=lambda t: (t[0], t[1]))
    merged: List[List[Union[float, str]]] = [list(spans[0])]

    for s, e, w, src in spans[1:]:
        S, E, W, SRC = merged[-1]  # type: ignore[misc]
        S = float(S); E = float(E); W = float(W)  # defensive
        s = float(s); e = float(e); w = float(w)

        close = (s - E) <= float(gap_tol)
        similar = abs(w - W) <= float(weight_tol) * max(1e-6, (abs(w) + abs(W)) / 2.0)

        if close and similar:
            # merge into [S, max(E,e)] and duration-weighted average weight
            newE = max(E, e)
            dur_old = max(1e-9, E - S)
            dur_new = max(1e-9, e - s)
            newW = (W * dur_old + w * dur_new) / max(1e-9, (dur_old + dur_new))

            merged[-1][1] = newE
            merged[-1][2] = newW
            merged[-1][3] = (
                "agree" if (str(SRC) == "agree" and src == "agree")
                else (str(SRC) if abs(W) >= abs(w) else src)
            )
        else:
            merged.append([s, e, w, src])

    return [tuple(m) for m in merged]  # type: ignore[misc]


def _normalize_method_weights(method_weights: Tuple[float, float]) -> Tuple[float, float]:
    wL, wT = float(method_weights[0]), float(method_weights[1])
    s = wL + wT
    if s <= 0:
        return (0.5, 0.5)
    return (wL / s, wT / s)


def fuse_lime_timeshap_payload(
    payload_L: Mapping[str, object],
    payload_T: Mapping[str, object],
    *,
    agg: str = "geomean",           # 'geomean'|'mean'|'min'|'max'
    beta: float = 0.35,             # keep solo spans at reduced weight
    tau: float = 0.02,              # prune tiny spans after fusion
    tau_mode: str = "abs",          # 'abs' or 'mass'
    topk: int = 5,
    method_weights: Tuple[float, float] = (0.5, 0.5),  # (w_LIME, w_TimeSHAP)
    sign_policy: str = "penalize",  # 'penalize'|'drop'|'abs'
    gap_merge: float = 0.02,
    wt_merge_tol: float = 0.15,
    force_leads: Sequence[str] = (),  # e.g., ('II','V1')
) -> Payload:
    """Fuse two payloads (LIME & TimeSHAP) into a single payload.

    Args:
        payload_L, payload_T: Input payloads.
        agg: How to combine magnitudes in overlap:
            - geomean: sqrt(|wL|*|wT|) (default)
            - mean/min/max on magnitudes
        beta: Scale for solo regions (present only in one method).
        tau: Pruning threshold (either abs weight or mass depending on tau_mode).
        tau_mode:
            - "abs": keep spans where |w| >= tau
            - "mass": keep spans where duration*|w| >= tau
        topk: How many leads to list in "top5_leads" (name kept for compatibility).
        method_weights: Lead-score blend weights (normalized internally).
        sign_policy:
            - "abs": ignore sign; overlap becomes positive magnitude
            - "drop": drop overlaps with opposite sign
            - "penalize": keep opposite-sign overlap but shrink (conservative)
        gap_merge, wt_merge_tol: Adjacent span merge params.
        force_leads: Leads to force into top list if present.

    Returns:
        payload_F: Fused payload dict.
    """
    if "mat_path" not in payload_L or "mat_path" not in payload_T:
        raise KeyError("Both payloads must include 'mat_path'.")
    if str(payload_L["mat_path"]) != str(payload_T["mat_path"]):
        raise ValueError("mat_path differs between LIME and TimeSHAP payloads.")

    if topk <= 0:
        raise ValueError(f"topk must be > 0, got {topk}")
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    if tau_mode not in ("abs", "mass"):
        raise ValueError("tau_mode must be 'abs' or 'mass'")
    if sign_policy not in ("penalize", "drop", "abs"):
        raise ValueError("sign_policy must be 'penalize'|'drop'|'abs'")

    mat_path = str(payload_L["mat_path"])
    page_seconds = max(
        float(payload_L.get("page_seconds", 10.0)),  # type: ignore[arg-type]
        float(payload_T.get("page_seconds", 10.0)),  # type: ignore[arg-type]
    )
    target_label = str(payload_L.get("target_label") or payload_T.get("target_label") or "")

    # -------- 1) fused lead-level scores (blend of normalized nonneg lead scores) --------
    perL: PerLeadSpans = dict(payload_L.get("perlead_spans", {}) or {})  # type: ignore[assignment]
    perT: PerLeadSpans = dict(payload_T.get("perlead_spans", {}) or {})  # type: ignore[assignment]

    sL_in: LeadScores = dict(payload_L.get("lead_scores", {}) or {})  # type: ignore[assignment]
    sT_in: LeadScores = dict(payload_T.get("lead_scores", {}) or {})  # type: ignore[assignment]

    # fallback if not provided (helps robustness)
    if not sL_in:
        sL_in = _lead_scores_fallback(perL)
    if not sT_in:
        sT_in = _lead_scores_fallback(perT)

    sL = _normalize_nonneg_scores(sL_in)
    sT = _normalize_nonneg_scores(sT_in)

    wL, wT = _normalize_method_weights(method_weights)

    all_leads = sorted(set(sL) | set(sT))
    lead_scores_fused: Dict[str, float] = {L: (wL * sL.get(L, 0.0) + wT * sT.get(L, 0.0)) for L in all_leads}

    # -------- 2) fuse per-lead spans (sign-aware) --------
    maxL = _perlead_max_abs(perL)
    maxT = _perlead_max_abs(perT)

    perlead_spans_F: PerLeadSpans = {}
    agreement_meta: Dict[str, Dict[str, float]] = {}

    leads_union = sorted(set(perL.keys()) | set(perT.keys()))

    for L in leads_union:
        spansL = perL.get(L, [])
        spansT = perT.get(L, [])

        def _norm_spans(spans: List[Span], maxabs: float) -> List[Span]:
            m = max(1e-9, float(maxabs))
            return [(float(s), float(e), float(w) / m) for (s, e, w) in spans]

        nL = _norm_spans(spansL, maxL.get(L, 0.0)) if spansL else []
        nT = _norm_spans(spansT, maxT.get(L, 0.0)) if spansT else []

        fused: List[SpanSrc] = []
        usedL = [False] * len(nL)
        usedT = [False] * len(nT)

        agree_mass = 0.0
        other_mass = 0.0

        # overlaps
        for i, (s1, e1, w1) in enumerate(nL):
            for j, (s2, e2, w2) in enumerate(nT):
                ov = _interval_overlap((s1, e1), (s2, e2))
                if not ov:
                    continue

                a1, a2 = abs(w1), abs(w2)
                if agg == "geomean":
                    mag = float(np.sqrt(a1 * a2))
                elif agg == "mean":
                    mag = float(0.5 * (a1 + a2))
                elif agg == "min":
                    mag = float(min(a1, a2))
                elif agg == "max":
                    mag = float(max(a1, a2))
                else:
                    mag = float(np.sqrt(a1 * a2))

                if sign_policy == "abs":
                    w = mag
                    src = "agree"
                else:
                    same_sign = (np.sign(w1) == np.sign(w2)) or (w1 == 0.0 or w2 == 0.0)
                    if same_sign:
                        w = float(np.sign(w1 if abs(w1) >= abs(w2) else w2) * mag)
                        src = "agree"
                    else:
                        if sign_policy == "drop":
                            usedL[i] = True
                            usedT[j] = True
                            continue
                        # conservative: keep but penalize
                        w = float(0.25 * (w1 + w2))
                        src = "conflict"

                fused.append((ov[0], ov[1], w, src))
                mass = (ov[1] - ov[0]) * abs(w)
                if src == "agree":
                    agree_mass += mass
                else:
                    other_mass += mass

                usedL[i] = True
                usedT[j] = True

        # solo spans
        if beta > 0:
            for i, (s1, e1, w1) in enumerate(nL):
                if not usedL[i]:
                    w = float(beta * w1)
                    fused.append((s1, e1, w, "solo_L"))
                    other_mass += (e1 - s1) * abs(w)

            for j, (s2, e2, w2) in enumerate(nT):
                if not usedT[j]:
                    w = float(beta * w2)
                    fused.append((s2, e2, w, "solo_T"))
                    other_mass += (e2 - s2) * abs(w)

        # prune tiny spans
        if tau_mode == "mass":
            fused = [(s, e, w, src) for (s, e, w, src) in fused if (e - s) * abs(w) >= float(tau)]
        else:
            fused = [(s, e, w, src) for (s, e, w, src) in fused if abs(w) >= float(tau)]

        # merge neighbors
        fused = _merge_adjacent(fused, gap_tol=gap_merge, weight_tol=wt_merge_tol)

        if fused:
            perlead_spans_F[L] = [(s, e, w) for (s, e, w, _) in fused]

        agreement_meta[L] = {
            "agreement_mass": float(agree_mass),
            "other_mass": float(other_mass),
        }

    # -------- 3) choose top-k leads by fused mass (duration × |w|) --------
    lead_masses = _lead_mass_from_spans(perlead_spans_F)

    ranked = [L for L, _ in sorted(lead_masses.items(), key=lambda kv: kv[1], reverse=True)]

    top = list(ranked)
    for fl in force_leads:
        if fl in lead_masses and fl not in top:
            top.append(fl)

    topk_leads = top[: int(topk)]

    payload_F: Payload = {
        "mat_path": mat_path,
        "target_label": target_label,
        "method_label": "LIME+TimeSHAP",
        "page_seconds": page_seconds,
        "perlead_spans": perlead_spans_F,
        "lead_scores": lead_scores_fused,
        "top5_leads": topk_leads,  # keep key name for compatibility
        "meta": {
            "lead_masses": lead_masses,
            "agreement_vs_other": agreement_meta,
            "settings": {
                "agg": agg,
                "beta": float(beta),
                "tau": float(tau),
                "tau_mode": tau_mode,
                "method_weights": (wL, wT),
                "sign_policy": sign_policy,
                "gap_merge": float(gap_merge),
                "wt_merge_tol": float(wt_merge_tol),
            },
        },
    }
    return payload_F
