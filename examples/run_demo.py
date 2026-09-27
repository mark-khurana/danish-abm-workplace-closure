#!/usr/bin/env python
"""Run the workplace-closure analysis end to end on a synthetic population.

    python examples/run_demo.py            # 20,000 people, 4 draws per scenario, CPU

Steps:
  1. build a synthetic population (synthetic/make_population.py) and its community
     contact matrix (decomposition/build_contact_matrix.py) if they do not exist;
  2. turn the fitted parameters in params/wave1_calibrated.json into a checkpoint;
  3. simulate the three scenarios used in the paper for every parameter draw:
       A  workplace contact reduced, with displacement of contact into homes
       B  no reduction in workplace contact
       C  workplace contact reduced, without displacement
  4. print infections prevented by workplace group and income quartile.

The population is invented, so the numbers are illustrative only. They show that the
code runs and what it produces; they are not estimates for Denmark.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def make_checkpoint(params_json: Path, out: Path) -> Path:
    p = json.loads(params_json.read_text())
    r2 = torch.tensor(p.pop("R2_weekly"), dtype=torch.float32).reshape(-1, 1)
    p["R2"] = r2.flatten().tolist()
    ckpt = {"R2": r2, "params": p,
            "anchored_r2_mean": float(r2.mean()), "r2_mean": float(r2.mean())}
    torch.save(ckpt, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000, help="synthetic population size")
    ap.add_argument("--runs", type=int, default=4, help="parameter draws per scenario")
    ap.add_argument("--out", default="demo_output")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    os.chdir(ROOT)
    pop = ROOT / "synthetic_population"
    if not (pop / "agents.parquet").exists():
        subprocess.run([sys.executable, "synthetic/make_population.py",
                        "--out", str(pop), "--n", str(a.n)], check=True)

    if not (pop / "contact_matrix.pt").exists():
        subprocess.run([sys.executable, "decomposition/build_contact_matrix.py",
                        "--population-dir", str(pop)], check=True)

    out = ROOT / a.out
    out.mkdir(exist_ok=True)
    ckpt = make_checkpoint(ROOT / "params" / "wave1_calibrated.json", out / "checkpoint.pt")

    subprocess.run([sys.executable, "decomposition/run_displacement_decomposition.py",
                    "--wave", "config", "--checkpoint", str(ckpt),
                    "--policies", "workplace_closure", "--arm-d", "none",
                    "--n-runs", str(a.runs), "--device", a.device,
                    "--strata", "workplace_band,income", "--out", str(out)], check=True)

    df = pd.read_csv(out / "stratified_arms.csv")
    print("\nInfections prevented by the reduction in workplace contact "
          "(scenario B minus C), share of each group's infections without it:")
    print(summarise(df).to_string(index=False))
    print(f"\nFull output in {out}/  (synthetic data: illustrative only)")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each group's expected infections (scenario B) that the reduction
    prevented (B minus C), averaged over runs."""
    names = {"workplace_band": {"0": "no workplace", "1": "small", "2": "medium",
                                "3": "large"}}
    rows = []
    for (kind, stratum), g in df.groupby(["stratum_kind", "stratum"], sort=True):
        b = g.loc[g["arm"] == "B_no_closure", "infections"].mean()
        c = g.loc[g["arm"] == "C_reduction_only", "infections"].mean()
        label = names.get(kind, {}).get(str(stratum).split(".")[0], str(stratum))
        rows.append({"grouping": kind, "group": label, "people": int(g["n_agents"].iloc[0]),
                     "infections_B": round(b, 1), "infections_C": round(c, 1),
                     "prevented_%": round(100 * (b - c) / b, 1) if b else float("nan")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
