"""Registry data loaders: convert individual-level CSVs into model-ready formats."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .agent_properties import read_tabular


def _build_agent_id_map(
    agents_file: Union[str, Path, None],
    agent_col: str = "agent_id",
) -> Optional[dict]:
    """Build original-agent-ID → row-index mapping from agents.parquet."""
    if agents_file is None:
        return None
    agents_file = Path(agents_file)
    if not agents_file.exists():
        return None
    agents_df = read_tabular(agents_file)
    aid_col = None
    for c in [agent_col, "agent_id", "person_id", "id"]:
        if c in agents_df.columns:
            aid_col = c
            break
    if aid_col is None:
        return None
    sorted_ids = agents_df[aid_col].sort_values().values
    id_map = {int(old): new for new, old in enumerate(sorted_ids)}
    print(f"  Agent ID map: {len(id_map)} agents, "
          f"original range [{int(sorted_ids[0])}..{int(sorted_ids[-1])}] → [0..{len(id_map)-1}]")
    return id_map


def vaccination_csv_to_schedule(
    path_or_df: Union[str, Path, pd.DataFrame],
    sim_start: str,
    n_days: int,
    n_agents: int,
    agent_col: str = "agent_id",
    date_col: str = "date",
    type_col: str = "vaccine_type",
    dose_col: str = "dose",
    agents_file: Union[str, Path, None] = None,
) -> dict:
    """Convert a vaccination registry CSV to a schedule dict."""
    if isinstance(path_or_df, (str, Path)):
        df = read_tabular(path_or_df)
    else:
        df = path_or_df.copy()

    df[date_col] = pd.to_datetime(df[date_col])
    sim_start_dt = pd.Timestamp(sim_start)
    df["_day"] = (df[date_col] - sim_start_dt).dt.days

    if type_col not in df.columns:
        df[type_col] = "unknown"
    if dose_col not in df.columns:
        df = df.sort_values([agent_col, date_col])
        df[dose_col] = df.groupby(agent_col).cumcount() + 1

    df = df[df["_day"] < n_days]

    id_map = _build_agent_id_map(agents_file, agent_col)

    if id_map is None and isinstance(path_or_df, (str, Path)):
        vax_path = Path(path_or_df)
        for candidate in [
            vax_path.parent.parent / "agents.parquet",
            vax_path.parent / "agents.parquet",
        ]:
            id_map = _build_agent_id_map(candidate, agent_col)
            if id_map is not None:
                print(f"  Auto-discovered agents file: {candidate}")
                break

    if id_map is not None:
        valid_ids = set(id_map.keys())
        df = df[df[agent_col].isin(valid_ids)]
        df[agent_col] = df[agent_col].map(id_map)
    else:
        df = df[df[agent_col] < n_agents]

    df_pre = df[df["_day"] < 0].copy()
    df = df[df["_day"] >= 0].copy()

    schedule = {}
    for day, group in df.groupby("_day"):
        schedule[int(day)] = {
            "agent_ids": torch.tensor(group[agent_col].values, dtype=torch.long),
            "vaccine_types": group[type_col].tolist(),
            "doses": torch.tensor(group[dose_col].values, dtype=torch.long),
        }

    pre_sim_agents = None
    pre_sim_doses = None
    pre_sim_days_before = None
    if len(df_pre) > 0:
        agg = df_pre.groupby(agent_col).agg(
            max_dose=(dose_col, "max"),
            last_day=("_day", "max"),
        ).reset_index()
        pre_sim_agents = torch.tensor(agg[agent_col].values, dtype=torch.long)
        pre_sim_doses = torch.tensor(agg["max_dose"].values, dtype=torch.long)
        pre_sim_days_before = torch.tensor(agg["last_day"].values, dtype=torch.long)
        print(f"Pre-sim vaccinations: {len(agg):,} agents "
              f"(dose counts: {dict(agg['max_dose'].value_counts().sort_index())})")

    schedule["_pre_sim"] = {
        "agent_ids": pre_sim_agents,
        "doses": pre_sim_doses,
        "days_before": pre_sim_days_before,
    }

    total_doses = sum(len(v["agent_ids"]) for k, v in schedule.items()
                      if k != "_pre_sim" and v.get("agent_ids") is not None)
    unique_agents = df[agent_col].nunique()
    n_pre = len(pre_sim_agents) if pre_sim_agents is not None else 0
    print(f"Vaccination schedule: {total_doses:,} in-sim doses over "
          f"{len(schedule) - 1} days + {n_pre:,} pre-sim agents "
          f"({unique_agents + n_pre:,} total unique agents)")

    return schedule


def save_vaccination_schedule(schedule: dict, path: Union[str, Path]):
    """Save vaccination schedule as a .pt file readable by VaccinationRollout."""
    flat = {}
    for day, data in schedule.items():
        if day == "_pre_sim":
            flat["_pre_sim"] = data
        else:
            flat[day] = data["agent_ids"]
    torch.save(flat, str(path))
    print(f"Saved vaccination schedule: {path}")


def build_vaccination_schedule_from_csv(
    csv_path: Union[str, Path],
    output_path: Union[str, Path],
    sim_start: str,
    n_days: int,
    n_agents: int,
    **kwargs,
):
    """One-shot: CSV → vaccination_schedule.pt."""
    schedule = vaccination_csv_to_schedule(csv_path, sim_start, n_days, n_agents, **kwargs)
    save_vaccination_schedule(schedule, output_path)
    return schedule


def load_individual_test_data(
    path_or_df: Union[str, Path, pd.DataFrame],
    sim_start: str,
    n_days: int,
    n_agents: int,
    agent_col: str = "agent_id",
    date_col: str = "test_date",
    result_col: str = "result",
    agents_file: Union[str, Path, None] = None,
    infection_window_days: int = 60,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Load individual-level test records for calibration."""
    if isinstance(path_or_df, (str, Path)):
        df = read_tabular(path_or_df)
    else:
        df = path_or_df.copy()

    if date_col not in df.columns:
        for alt in ["date", "Date", "test_date", "TEST_DATE"]:
            if alt in df.columns:
                date_col = alt
                break
    if result_col not in df.columns:
        for alt in ["result", "Result", "test_result", "TEST_RESULT"]:
            if alt in df.columns:
                result_col = alt
                break

    df[date_col] = pd.to_datetime(df[date_col])

    if df[result_col].dtype.kind in ("O", "U") or str(df[result_col].dtype) == "string":
        df[result_col] = df[result_col].str.lower().map(
            {"positive": 1, "pos": 1, "1": 1, "negative": 0, "neg": 0, "0": 0}
        ).fillna(0).astype(int)
    else:
        _codes = df[result_col].value_counts(dropna=False)
        _unknown = _codes[~_codes.index.isin([0, 1])]
        if len(_unknown):
            _n_unknown = int(_unknown.sum())
            _frac = _n_unknown / max(len(df), 1)
            _desc = ", ".join(f"{k}={v:,}" for k, v in _unknown.items())
            if _frac > 0.05:
                raise ValueError(
                    f"result column '{result_col}' holds non-binary codes "
                    f"({_desc}) covering {_frac:.1%} of {len(df):,} records. "
                    f"Expected 0=negative, 1=positive. Too large a share to "
                    f"discard silently — confirm the column and its encoding "
                    f"before calibrating on it."
                )
            print(f"  Dropped {_n_unknown:,} test records with non-binary "
                  f"result codes ({_desc}) — {_frac:.2%} of the register. "
                  f"Assumed inconclusive: they carry no information about "
                  f"infection state, so they are excluded from BOTH the "
                  f"positivity numerator and its denominator.")
            df = df[df[result_col].isin([0, 1])]
        df[result_col] = df[result_col].astype(int)
    sim_start_dt = pd.Timestamp(sim_start)
    df["_day"] = (df[date_col] - sim_start_dt).dt.days

    df["_result_raw"] = df[result_col].astype(float)

    if infection_window_days > 0:
        n_pos_before = int(df[result_col].sum())
        df = df.sort_values([agent_col, date_col])

        _res = df[result_col].to_numpy().copy()
        _pos = np.flatnonzero(_res == 1)
        if _pos.size:
            _aid = df[agent_col].to_numpy()[_pos]
            _day = df["_day"].to_numpy()[_pos]
            _first_of_agent = np.empty(_aid.size, dtype=bool)
            _first_of_agent[0] = True
            np.not_equal(_aid[1:], _aid[:-1], out=_first_of_agent[1:])
            _drop = np.zeros(_aid.size, dtype=bool)
            _last = 0
            for _i in range(_aid.size):
                if _first_of_agent[_i]:
                    _last = _day[_i]
                elif _day[_i] - _last < infection_window_days:
                    _drop[_i] = True
                else:
                    _last = _day[_i]
            _res[_pos[_drop]] = 0
        df[result_col] = _res
        n_pos_after = int(df[result_col].sum())
        if n_pos_before != n_pos_after:
            print(f"  Dedup ({infection_window_days}d window): {n_pos_before:,} → {n_pos_after:,} positives "
                  f"({n_pos_before - n_pos_after:,} reclassified)")

    df = df[(df["_day"] >= 0) & (df["_day"] < n_days)]

    id_map = _build_agent_id_map(agents_file, agent_col)

    if id_map is None and isinstance(path_or_df, (str, Path)):
        test_path = Path(path_or_df)
        for candidate in [
            test_path.parent.parent / "agents.parquet",
            test_path.parent / "agents.parquet",
        ]:
            id_map = _build_agent_id_map(candidate, agent_col)
            if id_map is not None:
                print(f"  Auto-discovered agents file: {candidate}")
                break

    if id_map is not None:
        valid_ids = set(id_map.keys())
        n_before = len(df)
        df = df[df[agent_col].isin(valid_ids)]
        df[agent_col] = df[agent_col].map(id_map)
        max_id = df[agent_col].max() if len(df) > 0 else -1
        print(f"  Remapped agent IDs: {n_before:,} → {len(df):,}, max remapped ID={max_id}")
    else:
        df = df[df[agent_col] < n_agents]

    type_col = None
    for cand in ("test_type", "testtype", "test_kind"):
        if cand in df.columns:
            type_col = cand
            break

    test_data = {}
    for day, group in df.groupby("_day"):
        entry = [
            torch.tensor(group[agent_col].values, dtype=torch.long),
            torch.tensor(group[result_col].values, dtype=torch.float32),
        ]
        if type_col is not None:
            entry.append(np.asarray(group[type_col].astype(str).values, dtype=object))
        else:
            entry.append(None)
        entry.append(torch.tensor(group["_result_raw"].values, dtype=torch.float32))
        test_data[int(day)] = tuple(entry)

    n_positive = df[result_col].sum()
    print(f"Test data: {len(df):,} tests over {len(test_data)} days")
    print(f"  Positive: {n_positive:,.0f} ({df[result_col].mean()*100:.1f}%)")
    print(f"  Agents tested: {df[agent_col].nunique():,}")
    if type_col is None:
        print(f"  test_type: ABSENT — the positivity anchor cannot be restricted to a "
              f"screening arm, so it assumes testing was unselected (bias 1.0)")
    else:
        _by = df.groupby(type_col)[["_result_raw", result_col]].agg(
            ["sum", "count", "mean"])
        print(f"  test_type present ({type_col}); positivity by arm "
              f"(undeduped = prevalence, used by the anchor; "
              f"deduped = episodes, used by the cases term):")
        for _t in _by.index:
            _n = int(_by.loc[_t, ("_result_raw", "count")])
            _pr = float(_by.loc[_t, ("_result_raw", "mean")])
            _pd_ = float(_by.loc[_t, (result_col, "mean")])
            print(f"    {str(_t):10s} {_n:>12,} tests  "
                  f"undeduped {_pr * 100:6.4f}%  deduped {_pd_ * 100:6.4f}%")
        _raw_mean = _by[("_result_raw", "mean")]
        if len(_by) == 2 and _raw_mean.min() > 0:
            print(f"    undeduped positivity ratio between arms: "
                  f"{_raw_mean.max() / _raw_mean.min():.2f} "
                  f"(part assay sensitivity, part test-seeking selection — see "
                  f"--anchor-test-types)")

    return test_data


def load_municipality_ground_truth(
    path_or_df: Union[str, Path, pd.DataFrame],
    sim_start: str,
    n_days: int,
    scale_factor: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Load SSI municipality-level case time series."""
    if isinstance(path_or_df, (str, Path)):
        path_str = str(path_or_df)
        ext = Path(path_str).suffix.lower()
        if ext in (".parquet", ".pq"):
            df = pd.read_parquet(path_str)
        else:
            df = pd.read_csv(path_str, sep=";")
    else:
        df = path_or_df.copy()

    df["SampleDate"] = pd.to_datetime(df["SampleDate"])
    sim_start_dt = pd.Timestamp(sim_start)
    sim_end_dt = sim_start_dt + pd.Timedelta(days=n_days - 1)

    df = df[(df["SampleDate"] >= sim_start_dt) & (df["SampleDate"] <= sim_end_dt)]
    df = df.sort_values("SampleDate").reset_index(drop=True)

    skip_cols = {"SampleDate", "NA", "Christiansø"}
    muni_cols = [c for c in df.columns if c not in skip_cols]

    for col in muni_cols:
        df[col] = df[col].astype(str).str.strip().str.replace(".", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    n = min(len(df), n_days)
    values = df[muni_cols].values[:n].astype(np.float32)
    daily_cases = torch.tensor(values, dtype=torch.float32) * scale_factor

    print(f"Municipality GT: {n} days, {len(muni_cols)} municipalities")
    print(f"  Total scaled cases: {daily_cases.sum():.0f}")

    return {
        "municipality_names": muni_cols,
        "daily_cases": daily_cases,
        "dates": df["SampleDate"].values[:n],
    }
