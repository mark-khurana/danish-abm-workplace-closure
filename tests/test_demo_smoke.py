"""End-to-end check: synthetic population -> scenarios A/B/C -> summary table."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_and_writes_all_three_scenarios(tmp_path):
    subprocess.run([sys.executable, "examples/run_demo.py", "--runs", "1",
                    "--out", str(tmp_path)], cwd=ROOT, check=True)
    df = pd.read_csv(tmp_path / "stratified_arms.csv")
    assert set(df["arm"]) == {"A_observed", "B_no_closure", "C_reduction_only"}
    assert set(df["stratum_kind"]) == {"workplace_band", "income"}
    assert (df.groupby("arm")["infections"].sum() > 0).all()
