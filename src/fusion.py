# fusion.py
"""
Fuse LIME + TimeSHAP per-lead spans into a single payload suitable for plotting.
"""

from __future__ import annotations
from typing import Dict, Tuple, List

import numpy as np


def _interval_overlap(a: Tuple[float, float], b: Tuple[float, float]):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return (s, e) if e > s else None


def _normalize_scores(d: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([max(0.0, float(v)) for v in d.values()], float)
    s = float(vals.sum())
    if s <= 0:
        return {k: 0.0 for k in d}
    return {k: float(max(0.0, d[k])) / s for k in d}


def _perlead_max_abs(perlead_spans: Dict[str, List[Tuple[float, float, float]]]):
    out = {}
    for L, spans in perlead_spans.items():
        ws = [abs(float(t[2])) for t in spans if len(t) >= 3]
        out[L] = max(ws) if ws else 0.0
    return out


def _merge_adjacent(
    spans: List[Tuple[float, float, float, str]],
    gap_tol: float = 0.025,
    weight_tol: float = 0.15,
):
    """
    Merge spans [(s,e,w,src)] if they are close and weights similar.
    Keeps src as 'agree' if both were 'agree', else dominant source.
    """
    if not spans:
        return spans
    spans = sorted(spans, key=lambda t: (t[0], t[1]))
    merged = [list(spans[0])]
    for s, e, w, src in spans[1:]:
        S, E, W, SRC = merged[-1]
        if s - E <= gap_tol and (
            abs(w - W) <= weight_tol * max(1e-6, (abs(w) + abs(W)) / 2)
        ):
            # merge
            merged[-1][1] = max(E, e)
            dur_old, dur_new = (E - S), (e - s)
            merged[-1][2] = (W * dur_old + w * dur_new) / max(
                1e-9, (dur_old + dur_new)
            )
            merged[-1][3] = (
                "agree"
                if (SRC == "agree" and src == "agree")
                else (SRC if abs(W) >= abs(w) else src)
            )
        else:
            merged.append([s, e, w, src])
    return [tuple(m) for m in merged]


def fuse_lime_timeshap_payload(
    payload_L: Dict,
    payload_T: Dict,
    *,
    agg: str = "geomean",           # 'geomean'|'mean'|'min'|'max'
    beta: float = 0.35,             # keep solo spans at reduced weight
    tau: float = 0.02,              # prune tiny spans after fusion
    tau_mode: str = "abs",          # 'abs' or 'mass'
    topk: int = 5,
    method_weights=(0.5, 0.5),      # (w_LIME, w_TimeSHAP)
    sign_policy: str = "penalize",  # 'penalize'|'drop'|'abs'
    gap_merge: float = 0.02,
    wt_merge_tol: float = 0.15,
    force_leads=(),                 # e.g., ('II','V1')
) -> Dict:
    """
    Fuse two payloads (LIME & TimeSHAP) into a single one.

    Both payloads must have:
      - 'mat_path'
      - 'perlead_spans': dict[lead -> [(s,e,w), ...]]
      - 'lead_scores':   dict[lead -> score]
    """
    if payload_L["mat_path"] != payload_T["mat_path"]:
        raise ValueError("mat_path differs between LIME and TimeSHAP payloads.")

    mat_path = payload_L["mat_path"]
    page_seconds = max(
        float(payload_L.get("page_seconds", 10.0)),
        float(payload_T.get("page_seconds", 10.0)),
    )
    target_label = (
        payload_L.get("target_label")
        or payload_T.get("target_label")
        or ""
    )

    # 1) lead-level fused scores
    sL = _normalize_scores(payload_L.get("lead_scores", {}))
    sT = _normalize_scores(payload_T.get("lead_scores", {}))
    wL, wT = method_weights
    all_leads = sorted(set(sL) | set(sT))
    lead_scores_fused = {
        L: wL * sL.get(L, 0.0) + wT * sT.get(L, 0.0) for L in all_leads
    }

    # 2) per-lead spans, sign-aware
    perL = payload_L.get("perlead_spans", {})
    perT = payload_T.get("perlead_spans", {})
    maxL = _perlead_max_abs(perL)
    maxT = _perlead_max_abs(perT)

    perlead_spans_F: Dict[str, List[Tuple[float, float, float]]] = {}
    meta: Dict[str, Dict] = {}

    for L in sorted(set(perL.keys()) | set(perT.keys())):
        spansL = perL.get(L, [])
        spansT = perT.get(L, [])

        def _norm_spans(spans, maxabs):
            out = []
            m = max(1e-9, maxabs)
            for (s, e, w) in spans:
                out.append((float(s), float(e), float(w) / m))
            return out

        nL = _norm_spans(spansL, maxL.get(L, 0.0)) if spansL else []
        nT = _norm_spans(spansT, maxT.get(L, 0.0)) if spansT else []

        fused: List[Tuple[float, float, float, str]] = []
        usedL = [False] * len(nL)
        usedT = [False] * len(nT)
        agree_mass = 0.0
        solo_mass = 0.0

        # overlaps
        for i, (s1, e1, w1) in enumerate(nL):
            for j, (s2, e2, w2) in enumerate(nT):
                ov = _interval_overlap((s1, e1), (s2, e2))
                if not ov:
                    continue

                a1, a2 = abs(w1), abs(w2)
                if agg == "geomean":
                    mag = np.sqrt(a1 * a2)
                elif agg == "mean":
                    mag = 0.5 * (a1 + a2)
                elif agg == "min":
                    mag = min(a1, a2)
                elif agg == "max":
                    mag = max(a1, a2)
                else:
                    mag = np.sqrt(a1 * a2)

                if sign_policy == "abs":
                    w = mag
                    src = "agree"
                else:
                    same_sign = (
                        np.sign(w1) == np.sign(w2)
                        or (w1 == 0.0 or w2 == 0.0)
                    )
                    if same_sign:
                        w = np.sign(w1 if abs(w1) >= abs(w2) else w2) * mag
                        src = "agree"
                    else:
                        if sign_policy == "drop":
                            usedL[i] = True
                            usedT[j] = True
                            continue
                        elif sign_policy == "penalize":
                            w = 0.25 * (w1 + w2)
                            src = "conflict"

                fused.append((ov[0], ov[1], float(w), src))
                mass = (ov[1] - ov[0]) * abs(w)
                if src == "agree":
                    agree_mass += mass
                else:
                    solo_mass += mass
                usedL[i] = True
                usedT[j] = True

        # solo spans
        if beta > 0:
            for i, (s1, e1, w1) in enumerate(nL):
                if not usedL[i]:
                    w = beta * w1
                    fused.append((s1, e1, float(w), "solo_L"))
                    solo_mass += (e1 - s1) * abs(w)

            for j, (s2, e2, w2) in enumerate(nT):
                if not usedT[j]:
                    w = beta * w2
                    fused.append((s2, e2, float(w), "solo_T"))
                    solo_mass += (e2 - s2) * abs(w)

        # prune tiny
        if tau_mode == "mass":
            fused = [
                (s, e, w, src)
                for (s, e, w, src) in fused
                if (e - s) * abs(w) >= float(tau)
            ]
        else:
            fused = [
                (s, e, w, src)
                for (s, e, w, src) in fused
                if abs(w) >= float(tau)
            ]

        # merge neighbors
        fused = _merge_adjacent(
            fused,
            gap_tol=gap_merge,
            weight_tol=wt_merge_tol,
        )

        if fused:
            perlead_spans_F[L] = [(s, e, w) for (s, e, w, src) in fused]

        meta[L] = dict(
            agreement_mass=float(agree_mass),
            solo_mass=float(solo_mass),
        )

    # choose top-k leads by fused mass
    def _lead_mass(L):
        spans = perlead_spans_F.get(L, [])
        return sum((e - s) * abs(w) for (s, e, w) in spans)

    lead_masses = {L: _lead_mass(L) for L in perlead_spans_F}
    top_by_mass = [
        L
        for L, _ in sorted(
            lead_masses.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
    ]

    top = list(top_by_mass)
    for fl in force_leads:
        if fl in lead_masses and fl not in top:
            top.append(fl)
    top5 = top[:topk]

    payload_F = {
        "mat_path": mat_path,
        "target_label": target_label,
        "method_label": "LIME+TimeSHAP",
        "page_seconds": page_seconds,
        "perlead_spans": perlead_spans_F,
        "lead_scores": lead_scores_fused,
        "top5_leads": top5,
        "meta": {
            "lead_masses": lead_masses,
            "agreement_vs_solo": meta,
            "settings": {
                "agg": agg,
                "beta": beta,
                "tau": tau,
                "tau_mode": tau_mode,
                "method_weights": method_weights,
                "sign_policy": sign_policy,
                "gap_merge": gap_merge,
                "wt_merge_tol": wt_merge_tol,
            },
        },
    }
    return payload_F
