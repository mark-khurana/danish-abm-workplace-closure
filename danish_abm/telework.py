"""Occupation-weighted withdrawal from the workplace: the sensitivity the uniform
colleague multiplier cannot express.
"""
from __future__ import annotations

import json
import os

import numpy as np

TELEWORK_PUBLISHED = {
    "isco1_managers":              0.76,
    "isco2_professionals":         0.71,
    "isco3_associate":             0.54,
    "isco4_clerical":              0.84,
    "isco5_service_sales":         0.26,
    "isco6_agricultural":          0.00,
    "isco7_craft":                 0.02,
    "isco8_operators":             0.02,
    "isco9_elementary":            0.02,
    "self_employed":               0.32,
    "_all_dependent_employment":   0.37,
    "_top_job_wage_quintile":      0.74,
    "_bottom_job_wage_quintile":   0.03,
}

TELEWORK_DEFAULT = {
    "131": 0.76,
    "132": 0.71,
    "133": 0.54,
    "134": 0.25,
    "135": 0.02,
    "111": 0.32, "112": 0.32, "113": 0.32, "114": 0.32, "120": 0.32,
}

MODES = ("weighted", "literal")


def load_weights(path=None):
    path = path or os.environ.get("ABM_TELEWORK_WEIGHTS")
    if not path:
        return dict(TELEWORK_DEFAULT)
    with open(path) as fh:
        raw = json.load(fh)
    out = {}
    for k, v in raw.items():
        if str(k).startswith("_"):
            continue
        out[str(int(float(k)))] = float(v)
    return out


def agent_teleworkability(codes, exposure, weights=None):
    """tau per agent. `codes` are SOCIO13 values (NaN allowed); `exposure` is each
    agent's colleague exposure (sum of B_n over colleague links), used to weight the
    mean that fills uncoded agents. Returns (tau, filled_mask, mean_tau).
    """
    weights = weights or load_weights()
    codes = np.asarray(codes, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    tau = np.full(codes.shape, np.nan)
    ok = np.isfinite(codes)
    keys = np.where(ok, codes, -1).astype(np.int64).astype(str)
    for k, v in weights.items():
        tau[keys == k] = v
    coded = np.isfinite(tau)
    w = exposure[coded]
    mean_tau = float(np.average(tau[coded], weights=w)) if coded.any() and w.sum() > 0 \
        else float(np.nanmean(tau)) if coded.any() else 0.0
    filled = ~coded
    tau[filled] = mean_tau
    return tau, filled, mean_tau


def _iter_edges(edges, chunk):
    """Yield (src, dst, w) chunks from a (src, dst, w) triple or from a callable that
    returns an iterator of such chunks (the simulator passes a callable so the
    colleague edges are never materialised as one long-typed copy on the GPU).
    """
    if callable(edges):
        yield from edges()
        return
    src, dst, w = edges
    n = len(src)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        yield src[s:e], dst[s:e], w[s:e]


def weighted_mean_presence(p, edges, chunk=50_000_000):
    """sum(w p_i p_j) / sum(w) over the colleague edges. numpy or torch."""
    try:
        import torch
        is_t = isinstance(p, torch.Tensor)
    except ImportError:
        is_t = False
    num = 0.0
    den = 0.0
    for src, dst, w in _iter_edges(edges, chunk):
        if is_t:
            pp = p[src.long()].float() * p[dst.long()].float()
            ww = w.float()
            num += float((pp * ww).sum())
            den += float(ww.sum())
        else:
            pp = p[src] * p[dst]
            num += float((pp * w).sum())
            den += float(w.sum())
    return num / den if den > 0 else float("nan")


def presence_from_k(tau, k):
    p = 1.0 - k * tau
    return p.clip(0.0, 1.0) if hasattr(p, "clip") else p.clamp(0.0, 1.0)


def solve_presence(tau, edges, m_target, mode="weighted", tol=1e-4, max_iter=60):
    """Presence per agent such that the exposure-weighted mean of p_i p_j is m_target."""
    if not 0 < m_target <= 1:
        raise ValueError(f"m_target must be in (0, 1], got {m_target}")
    if mode == "literal":
        p = presence_from_k(tau, 1.0)
        return 1.0, p, weighted_mean_presence(p, edges)
    if mode != "weighted":
        raise ValueError(f"unknown mode {mode!r}; use one of {MODES}")
    tmax = float(tau.max())
    if tmax <= 0:
        raise ValueError("every tau is 0; nothing to distribute")
    k_lo, k_hi = 0.0, 1.0 / tmax
    p_hi = presence_from_k(tau, k_hi)
    m_hi = weighted_mean_presence(p_hi, edges)
    if m_hi > m_target + tol:
        raise ValueError(
            f"target colleague multiplier {m_target:.3f} is below what full withdrawal of "
            f"the teleworkable share can deliver ({m_hi:.3f}); the tau table cannot "
            f"express this reduction")
    for _ in range(max_iter):
        k = 0.5 * (k_lo + k_hi)
        p = presence_from_k(tau, k)
        m = weighted_mean_presence(p, edges)
        if abs(m - m_target) < tol:
            return k, p, m
        if m > m_target:
            k_lo = k
        else:
            k_hi = k
    return k, p, m


def report(tau, p, codes, k, m_target, achieved, mean_tau, filled, exposure, labels=None):
    """Presence by SOCIO13 code, exposure-weighted, for the log and the JSON report."""
    codes = np.asarray(codes, dtype=float)
    p = np.asarray(p, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    rows = []
    for c in sorted(set(codes[np.isfinite(codes)].astype(int))):
        m = codes == c
        we = exposure[m]
        rows.append({"socio13": int(c), "label": (labels or {}).get(str(c), ""),
                     "n": int(m.sum()),
                     "n_with_colleagues": int((we > 0).sum()),
                     "tau": float(tau[m].mean()),
                     "presence": float(p[m].mean()),
                     "presence_exposure_weighted": float(np.average(p[m], weights=we))
                     if we.sum() > 0 else float("nan")})
    return {"mode_note": "presence p_i = clip(1 - k*tau_i, 0, 1); a colleague link carries p_i*p_j",
            "k": float(k), "m_target": float(m_target), "achieved_mean": float(achieved),
            "mean_tau_fill": float(mean_tau), "n_filled_with_mean": int(np.asarray(filled).sum()),
            "by_socio13": rows}
