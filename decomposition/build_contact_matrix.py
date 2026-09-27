"""Write `contact_matrix.pt` for a register population directory."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PREM_DIR = os.path.join(REPO, "danish_abm", "data", "prem_et_al_2021")

N_AGE = 8
AGE_BANDS = ["0-9", "10-19", "20-29", "30-39", "40-49",
             "50-59", "60-69", "70+"]

DK_POP_5YR = np.array([
    308, 337,
    352, 342,
    376, 385,
    383, 368,
    361, 383,
    405, 396,
    369, 326,
    275, 270,
], dtype=np.float64)


def load_prem(setting="others"):
    """Read one Prem 16x16 matrix with the stdlib csv module."""
    path = os.path.join(PREM_DIR, f"Denmark_contact_{setting}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Expected the Prem et al. 2021 Denmark matrices in "
            f"{PREM_DIR} (they are committed to the repo).")
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0][1:]
    body = [[float(v) for v in r[1:]] for r in rows[1:]]
    C = np.asarray(body, dtype=np.float64)
    if C.shape != (16, 16):
        raise ValueError(f"{path}: expected 16x16, got {C.shape}")
    return C, header, [r[0] for r in rows[1:]]


def aggregate_to_decades(C16, pop16):
    """16 five-year bands -> 8 ten-year bands."""
    C8 = np.zeros((N_AGE, N_AGE), dtype=np.float64)
    for I in range(N_AGE):
        i0, i1 = 2 * I, 2 * I + 1
        w = pop16[[i0, i1]]
        w = w / w.sum()
        for J in range(N_AGE):
            j0, j1 = 2 * J, 2 * J + 1
            C8[I, J] = (w[0] * (C16[i0, j0] + C16[i0, j1])
                        + w[1] * (C16[i1, j0] + C16[i1, j1]))
    return C8


def reciprocity_residual(C8, counts):
    """Mean relative asymmetry of N_a*C[a,b] vs N_b*C[b,a]. Zero when consistent."""
    F = counts[:, None] * C8
    return float((np.abs(F - F.T) / np.maximum(np.abs(F + F.T), 1e-12)).mean())


def reciprocity_correct(C8, counts):
    """Make N_a*C[a,b] symmetric against THIS population."""
    N = np.maximum(counts.astype(np.float64), 1.0)
    F = N[:, None] * C8
    F_sym = 0.5 * (F + F.T)
    return F_sym / N[:, None]


def agent_age_counts(pop_dir):
    """Agent counts per ten-year band, or None if agents.parquet is unreadable."""
    try:
        import pandas as pd
    except ImportError:
        return None
    path = os.path.join(pop_dir, "agents.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path, columns=["age_group"])
    except Exception:
        return None
    codes = pd.to_numeric(df["age_group"], errors="coerce")
    codes = codes.dropna().astype(int).clip(0, N_AGE - 1)
    return np.bincount(codes.to_numpy(), minlength=N_AGE).astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population-dir", required=True)
    ap.add_argument("--setting", default="others",
                    choices=["others", "all", "home", "school", "work"],
                    help="Prem setting. Use 'others' — home/school/work are carried "
                         "by network edges and would be double-counted.")
    ap.add_argument("--out", default=None,
                    help="Default: <population-dir>/contact_matrix.pt")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing contact_matrix.pt")
    ap.add_argument("--no-reciprocity-correct", action="store_true",
                    help="Skip the reciprocity correction. DIAGNOSTIC ONLY — the "
                         "derived R_community is then wrong by a factor of order 2.")
    a = ap.parse_args()

    out = a.out or os.path.join(a.population_dir, "contact_matrix.pt")
    if os.path.exists(out) and not a.force:
        print(f"  {out} already exists — pass --force to overwrite. Nothing done.")
        return 0
    if not os.path.isdir(a.population_dir):
        raise FileNotFoundError(f"no such population dir: {a.population_dir}")

    if a.setting != "others":
        print(f"  WARNING: setting='{a.setting}'. Only 'others' avoids double-counting "
              f"the household/school/workplace network layers.")

    C16, cols, rows = load_prem(a.setting)
    print(f"\n=== Community contact matrix ===")
    print(f"  Prem et al. 2021 Denmark '{a.setting}': 16x16")
    print(f"  bands: {rows[0]} .. {rows[-1]}")
    C8 = aggregate_to_decades(C16, DK_POP_5YR)

    print(f"\n  Aggregated to {N_AGE} ten-year bands "
          f"(mean daily contacts per person, row sums):")
    for i, band in enumerate(AGE_BANDS):
        print(f"    {band:>6s}  {C8[i].sum():6.2f}")
    print(f"  population mean over bands: {C8.sum(axis=1).mean():.2f} contacts/day")

    if not np.isfinite(C8).all() or (C8 < 0).any():
        raise AssertionError("aggregated matrix has non-finite or negative entries")

    counts = agent_age_counts(a.population_dir)
    recip_before = recip = None
    if counts is None or counts.sum() <= 0:
        if not a.no_reciprocity_correct:
            raise SystemExit(
                f"\n  Could not read age counts from "
                f"{a.population_dir}/agents.parquet, so the reciprocity correction "
                f"cannot be applied.\n"
                f"  It is not optional by default: without it N_a*C[a,b] is "
                f"asymmetric, the row/column convention for 'which index is the "
                f"susceptible' stops matching the per-layer matrices, and the derived "
                f"R_community is wrong by a factor of order 2.\n"
                f"  Override with --no-reciprocity-correct for a diagnostic build only.")
        print("\n  (agents.parquet not readable — no reciprocity report or correction)")
    else:
        recip_before = reciprocity_residual(C8, counts)
        print(f"\n  Register age distribution: "
              f"{', '.join(f'{b}={int(c):,}' for b, c in zip(AGE_BANDS, counts))}")
        print(f"  Reciprocity residual before: {recip_before:.3f}   "
              f"(mean |N_a C_ab - N_b C_ba| / |N_a C_ab + N_b C_ba|)")
        if a.no_reciprocity_correct:
            recip = recip_before
            print(f"  *** NOT CORRECTED (--no-reciprocity-correct). The derived "
                  f"R_community will be wrong by a factor of order 2. Diagnostic "
                  f"use only. ***")
        else:
            total_before = float((counts[:, None] * C8).sum())
            C8 = reciprocity_correct(C8, counts)
            total_after = float((counts[:, None] * C8).sum())
            recip = reciprocity_residual(C8, counts)
            if abs(total_after / max(total_before, 1e-12) - 1.0) > 1e-9:
                raise AssertionError(
                    f"reciprocity correction changed the total contact count "
                    f"({total_before:,.0f} -> {total_after:,.0f}); it must only "
                    f"redistribute across cells")
            print(f"  Reciprocity residual after:  {recip:.2e}  (corrected)")
            print(f"  Total contacts preserved: {total_before:,.0f} "
                  f"(rel. change {abs(total_after / total_before - 1):.1e})")
            print(f"    Prem's matrix is built against their own demographic "
                  f"projection, so a residual against the register population is "
                  f"expected. Correcting it is required, not cosmetic: the estimator's "
                  f"mean-field mass and the simulator's row-indexed hazard agree on "
                  f"which index is the susceptible one only when N_a*C is symmetric. "
                  f"Uses only the register age distribution and the published matrix, "
                  f"so no observed transmission quantity enters and no circularity is "
                  f"introduced.")
            print(f"  Row sums after correction (mean daily contacts per person):")
            for i, band in enumerate(AGE_BANDS):
                print(f"    {band:>6s}  {C8[i].sum():6.2f}")
            print(f"  population mean: {C8.sum(axis=1).mean():.2f} contacts/day")

    payload = {
        "C_comix": torch.tensor(C8, dtype=torch.float32),
        "age_bands": AGE_BANDS,
        "provenance": {
            "source": "Prem K, van Zandvoort K, Klepac P, et al. PLoS Comput Biol "
                      "2021;17(7):e1009098. doi:10.1371/journal.pcbi.1009098",
            "setting": a.setting,
            "setting_rationale": "non-home/school/work contacts only; the other "
                                 "settings are carried by explicit network layers",
            "aggregation": "16 five-year bands -> 8 ten-year bands; contacts summed "
                           "over target sub-bands, population-weighted mean over "
                           "source sub-bands",
            "population_weights": "Statistics Denmark FOLK1A 2020, five-year bands "
                                  "(within-decade weighting only)",
            "units": "mean daily contacts per person",
            "reciprocity_corrected": (not a.no_reciprocity_correct
                                      and recip_before is not None),
            "reciprocity_residual_before": recip_before,
            "reciprocity_residual_after": recip,
            "reciprocity_note": "C_sym[a,b] = (N_a C[a,b] + N_b C[b,a]) / (2 N_a), "
                                "against this population's register age counts. "
                                "Preserves total contacts exactly; required so that "
                                "N_a*C is symmetric and the estimator's mean-field "
                                "mass matches the simulator's row-indexed hazard.",
        },
    }
    torch.save(payload, out)
    print(f"\n  wrote {out}")
    print(f"  The mean-field channel is now ACTIVE for this population. R_community "
          f"must be re-derived (fit_structural_q.py) — the config prior 0.30 was "
          f"never on a common scale with q_layers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
