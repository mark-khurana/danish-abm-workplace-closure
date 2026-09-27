"""Preprocess OxCGRT + Google Mobility Denmark data into intervention-schedule-ready
format.
"""

import os
import numpy as np
import pandas as pd
import torch


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(DATA_DIR)
RAW_OXCGRT = os.path.join(DATA_DIR, "OxCGRT_Denmark.csv")
RAW_MOBILITY = os.path.join(PARENT_DIR, "google_mobility", "google_mobility_denmark.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "oxcgrt_denmark_clean.csv")


def load_oxcgrt(raw_path=RAW_OXCGRT):
    """Load and clean the OxCGRT Denmark CSV. Returns a DataFrame indexed by date."""
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")

    rename = {
        "C1M_School closing": "school_closing",
        "C2M_Workplace closing": "workplace_closing",
        "C3M_Cancel public events": "cancel_events",
        "C4M_Restrictions on gatherings": "gathering_restrictions",
        "C5M_Close public transport": "close_transport",
        "C6M_Stay at home requirements": "stay_home",
        "C7M_Restrictions on internal movement": "internal_movement",
        "C8EV_International travel controls": "intl_travel",
        "H1_Public information campaigns": "info_campaigns",
        "H2_Testing policy": "testing_policy",
        "H3_Contact tracing": "contact_tracing",
        "H6M_Facial Coverings": "mask_policy",
        "H7_Vaccination policy": "vaccination_policy",
        "H8M_Protection of elderly people": "elderly_protection",
        "StringencyIndex_Average": "stringency_index",
        "GovernmentResponseIndex_Average": "govt_response_index",
        "ContainmentHealthIndex_Average": "containment_health_index",
        "ConfirmedCases": "confirmed_cases",
        "ConfirmedDeaths": "confirmed_deaths",
    }

    flag_rename = {
        "C1M_Flag": "school_flag",
        "C2M_Flag": "workplace_flag",
        "C3M_Flag": "cancel_events_flag",
        "C4M_Flag": "gathering_flag",
        "C6M_Flag": "stay_home_flag",
    }

    all_rename = {**rename, **flag_rename}
    keep_cols = ["date"] + list(all_rename.values())

    df = df.rename(columns=all_rename)
    df = df[keep_cols].copy()
    df = df.set_index("date").sort_index()
    df = df.ffill()

    return df


def load_google_mobility(raw_path=RAW_MOBILITY):
    """Load Google Mobility Denmark CSV. Returns DataFrame indexed by date with
    multipliers.
    """
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])

    short = {
        "retail_and_recreation_percent_change_from_baseline": "retail_pct",
        "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pct",
        "parks_percent_change_from_baseline": "parks_pct",
        "transit_stations_percent_change_from_baseline": "transit_pct",
        "workplaces_percent_change_from_baseline": "workplaces_pct",
        "residential_percent_change_from_baseline": "residential_pct",
    }
    df = df.rename(columns=short)
    df = df[["date"] + list(short.values())].copy()
    df = df.set_index("date").sort_index()

    for col in short.values():
        mult_col = col.replace("_pct", "_mult")
        df[mult_col] = (1.0 + df[col] / 100.0).clip(lower=0.0)

    mult_cols = [c for c in df.columns if c.endswith("_mult")]
    for col in mult_cols:
        df[col + "_smooth"] = df[col].rolling(7, center=True, min_periods=1).mean()

    return df


def oxcgrt_to_multipliers(start, end, smooth=True):
    """Build intervention multipliers from OxCGRT + Google Mobility."""
    oxcgrt = load_oxcgrt()
    mobility = load_google_mobility()

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    ox = oxcgrt.loc[start:end].copy()
    mob = mobility.loc[start:end].copy()

    if len(ox) == 0:
        raise ValueError(f"No OxCGRT data for {start} to {end}")

    n_days = len(ox)
    suffix = "_mult_smooth" if smooth else "_mult"

    if len(mob) >= n_days:
        work_mult = mob[f"workplaces{suffix}"].values[:n_days]
    else:
        work_map = {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.1}
        work_mult = ox["workplace_closing"].map(work_map).fillna(1.0).values

    if len(mob) >= n_days:
        community_mult = mob[f"retail{suffix}"].values[:n_days]
    else:
        gather_map = {0: 1.0, 1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3}
        home_map = {0: 1.0, 1: 0.85, 2: 0.5, 3: 0.3}
        community_mult = (
            ox["gathering_restrictions"].map(gather_map).fillna(1.0).values
            * ox["stay_home"].map(home_map).fillna(1.0).values
        )

    mask_map = {0: 1.0, 1: 0.95, 2: 0.85, 3: 0.75, 4: 0.70}
    mask_mult = ox["mask_policy"].map(mask_map).fillna(1.0).values
    community_mult = community_mult * mask_mult

    if len(mob) >= n_days:
        household_mult = mob[f"residential{suffix}"].values[:n_days]
    else:
        household_mult = np.ones(n_days)

    school_map = {0: 1.0, 1: 0.8, 2: 0.3, 3: 0.0}
    school_policy_mult = ox["school_closing"].map(school_map).fillna(1.0).values
    school_mult = (school_policy_mult
                   * danish_school_holiday_multiplier(ox.index))

    stay_home_map = {0: 1.0, 1: 0.85, 2: 0.5, 3: 0.3}
    stay_home_mult = ox["stay_home"].map(stay_home_map).fillna(1.0).values
    neighborhood_mult = community_mult * stay_home_mult

    stringency = ox["stringency_index"].values / 100.0

    return {
        "family": np.ones(n_days),
        "household": household_mult,
        "neighborhood": neighborhood_mult,
        "colleague": work_mult,
        "classmate": school_mult,
        "classmate_policy": school_policy_mult,
        "community": community_mult,
        "masks": mask_mult,
        "stringency": stringency,
        "dates": ox.index.values,
        "n_days": n_days,
        "school": school_mult,
        "workplace": work_mult,
        "raw_school_closing": ox["school_closing"].values,
        "raw_workplace_closing": ox["workplace_closing"].values,
        "raw_gathering_restrictions": ox["gathering_restrictions"].values,
        "raw_stay_home": ox["stay_home"].values,
        "raw_mask_policy": ox["mask_policy"].values,
    }


SCHOOL_HOLIDAY_WEEKS = {
    42,
    7, 8,
}
SUMMER_HOLIDAY_DAYS = 44
SCHOOL_HOLIDAY_RESIDUAL = 0.15


def _easter_sunday(year):
    """Anonymous Gregorian algorithm. Needed because the Danish Easter school holiday is
    movable and lands inside the genomic window in both 2021 and 2022.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    import datetime
    return datetime.date(year, month, day)


def danish_school_holiday_multiplier(dates, residual=None):
    """Per-day factor in [residual, 1.0]: `residual` on Danish school holidays."""
    import datetime

    import pandas as pd

    res = SCHOOL_HOLIDAY_RESIDUAL if residual is None else float(residual)
    idx = pd.DatetimeIndex(dates)
    iso = idx.isocalendar()
    week = np.asarray(iso["week"], dtype=int)
    mask = np.isin(week, list(SCHOOL_HOLIDAY_WEEKS))

    md = np.asarray([(d.month, d.day) for d in idx], dtype=int)
    mask |= ((md[:, 0] == 12) & (md[:, 1] >= 22)) | ((md[:, 0] == 1) & (md[:, 1] <= 1))

    dts = np.asarray([d.date() for d in idx])
    for year in sorted({d.year for d in idx}):
        e = _easter_sunday(year)
        _jun30 = datetime.date(year, 6, 30)
        _last_sat = _jun30 - datetime.timedelta(days=(_jun30.weekday() - 5) % 7)
        spans = [
            (_last_sat,
             _last_sat + datetime.timedelta(days=SUMMER_HOLIDAY_DAYS - 1)),
            (e - datetime.timedelta(days=7), e + datetime.timedelta(days=1)),
            (e + datetime.timedelta(days=26), e + datetime.timedelta(days=26)),
            (e + datetime.timedelta(days=39), e + datetime.timedelta(days=40)),
            (e + datetime.timedelta(days=50), e + datetime.timedelta(days=50)),
        ]
        for lo, hi in spans:
            mask |= (dts >= lo) & (dts <= hi)

    out = np.ones(len(idx), dtype=float)
    out[mask] = res
    return out


def save_intervention_schedule(start, end, output_path, smooth=True):
    """Build and save an intervention_schedule.pt from OxCGRT + Google Mobility."""
    mult = oxcgrt_to_multipliers(start, end, smooth=smooth)
    n_days = mult["n_days"]

    col_names = ["q_family", "q_household", "q_neighborhood",
                 "q_colleague", "q_classmate", "R_community", "susceptibility"]

    schedule = torch.ones(n_days, 7)
    schedule[:, 0] = torch.tensor(mult["family"], dtype=torch.float32)
    schedule[:, 1] = torch.tensor(mult["household"], dtype=torch.float32)
    schedule[:, 2] = torch.tensor(mult["neighborhood"], dtype=torch.float32)
    schedule[:, 3] = torch.tensor(mult["colleague"], dtype=torch.float32)
    schedule[:, 4] = torch.tensor(mult["classmate"], dtype=torch.float32)
    schedule[:, 5] = torch.tensor(mult["community"], dtype=torch.float32)

    from danish_abm.npi import SPILLOVER_NPI_NAMES
    npi_active = torch.zeros(n_days, len(SPILLOVER_NPI_NAMES))
    _col = {n: i for i, n in enumerate(SPILLOVER_NPI_NAMES)}
    _classmate = torch.tensor(
        mult.get("classmate_policy", mult["classmate"]), dtype=torch.float32)
    _colleague = schedule[:, 3]
    npi_active[:, _col["school_closure"]] = (_classmate < 0.5).float()
    npi_active[:, _col["workplace_closure"]] = (_colleague < 0.25).float()
    npi_active[:, _col["partial_workplace_closure"]] = (
        (_colleague >= 0.25) & (_colleague < 0.75)).float()

    torch.save(
        {
            "multipliers": schedule,
            "npi_active": npi_active,
            "npi_active_names": list(SPILLOVER_NPI_NAMES),
            "start_date": str(start),
            "end_date": str(end),
            "n_days": n_days,
            "source": "OxCGRT (CC BY 4.0) + Google Mobility Reports",
            "column_names": col_names,
        },
        output_path,
    )
    _act = npi_active.sum(dim=0)
    print("  npi_active day-counts: " + ", ".join(
        f"{n}={int(v)}" for n, v in zip(SPILLOVER_NPI_NAMES, _act) if v > 0))
    print(f"Saved intervention schedule: {output_path}")
    print(f"  Shape: {schedule.shape}")
    print(f"  Period: {start} to {end} ({n_days} days)")
    return schedule


def print_summary(start, end, smooth=True):
    """Print a human-readable summary of multipliers for a period."""
    mult = oxcgrt_to_multipliers(start, end, smooth=smooth)
    print(f"Intervention multipliers: {start} to {end} ({mult['n_days']} days)")
    print(f"  Sources: Google Mobility (colleague, community, household), OxCGRT (classmate, masks)")
    print(f"{'Multiplier':<20} {'Min':>6} {'Max':>6} {'Mean':>6}  {'Source'}")
    print("-" * 70)
    sources = {
        "family": "1.0 (no policy)",
        "household": "Google Mobility residential",
        "neighborhood": "community * stay_home",
        "colleague": "Google Mobility workplaces",
        "classmate": "OxCGRT C1M x Danish school holidays",
        "community": "Google Mobility retail * masks",
        "masks": "OxCGRT H6M (folded into community)",
    }
    for key in ["family", "household", "neighborhood", "colleague", "classmate", "community", "masks"]:
        v = mult[key]
        print(f"  {key:<18} {v.min():>6.2f} {v.max():>6.2f} {v.mean():>6.2f}  {sources[key]}")
    print(f"  {'stringency':<18} {mult['stringency'].min():>6.2f} {mult['stringency'].max():>6.2f} {mult['stringency'].mean():>6.2f}  OxCGRT composite index")


def measured_workplace_multiplier(mob, day_start, day_end, index, column="workplaces"):
    """The colleague-layer multiplier for a closure period, MEASURED not assumed."""
    _override = os.environ.get("ABM_COLLEAGUE_MULT")
    if _override not in (None, ""):
        _m = float(_override)
        if not 0.0 < _m <= 1.0:
            raise ValueError(
                f"ABM_COLLEAGUE_MULT={_override!r} must be in (0, 1]. A multiplier of 0 "
                f"is a total shutdown, which is the defect this function was written to "
                f"remove.")
        print(f"  [colleague-mult] OVERRIDE ABM_COLLEAGUE_MULT={_m:.3f} for days "
              f"{day_start}-{day_end}, replacing the measured Google Mobility value. "
              f"This is the pre-mandate counterfactual; do NOT compare its benefit "
              f"level against a measured-multiplier run without saying which is which.",
              flush=True)
        return _m, (f"OVERRIDE ABM_COLLEAGUE_MULT={_m:.3f} — mandate-increment "
                    f"counterfactual, NOT the measured in-wave reduction")

    _lo, _hi = None, None
    if mob is not None and len(mob) >= len(index):
        col = f"{column}_mult_smooth"
        if col not in mob.columns:
            col = f"{column}_mult"
        if col in mob.columns:
            vals = mob[col].values[day_start:day_end]
            vals = vals[~pd.isna(vals)]
            if len(vals):
                m = float(np.mean(vals))
                _lo, _hi = float(np.min(vals)), float(np.max(vals))
                if m > 0:
                    return m, (f"Google Mobility Denmark {column}, mean of "
                               f"{len(vals)}d, daily range {_lo:.3f}-{_hi:.3f}")
    return 0.1, ("FALLBACK: no Google Mobility coverage for this window, OxCGRT "
                 "level map floor 0.1 — NOT a measurement, report as an assumption")


def oxcgrt_to_npi_schedule(start, end):
    """Build an NPISchedule from OxCGRT timeline data."""
    from danish_abm.npi import NPISchedule, NPIDefinition

    oxcgrt = load_oxcgrt()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    ox = oxcgrt.loc[start_ts:end_ts].copy()
    n_days = len(ox)

    try:
        mob = load_google_mobility().loc[start_ts:end_ts].copy()
    except Exception as _e:
        mob = None
        print(f"  [oxcgrt] Google Mobility unavailable ({type(_e).__name__}: {_e}) — "
              f"workplace multiplier falls back to the OxCGRT level map")

    schedule = NPISchedule(n_days)

    def _find_active_periods(series, threshold=1):
        """Find (start_day, end_day) periods where series >= threshold."""
        active = (series >= threshold).astype(int)
        diff = active.diff().fillna(active.iloc[0])
        periods = []
        start_day = None
        for i, d in enumerate(diff):
            if d == 1 and start_day is None:
                start_day = i
            elif d == -1 and start_day is not None:
                periods.append((start_day, i))
                start_day = None
        if start_day is not None:
            periods.append((start_day, n_days))
        return periods

    for s, e in _find_active_periods(ox["school_closing"], threshold=2):
        level = ox["school_closing"].iloc[s:e].median()
        if level >= 3:
            schedule.add("school_closure", s, e)
        else:
            schedule.add("school_closure", s, e,
                         override_effects={"affected_layers": {"classmate": 0.3}})

    for s, e in _find_active_periods(ox["workplace_closing"], threshold=2):
        level = ox["workplace_closing"].iloc[s:e].median()
        name = "workplace_closure" if level >= 3 else "partial_workplace_closure"
        mult, prov = measured_workplace_multiplier(mob, s, e, ox.index)
        print(f"  [oxcgrt] {name} days {s}-{e}: colleague x{mult:.3f}  ({prov})")
        schedule.add(name, s, e,
                     override_effects={"affected_layers": {"colleague": float(mult)}})

    for s, e in _find_active_periods(ox["mask_policy"], threshold=2):
        schedule.add("mask_mandate", s, e)

    for s, e in _find_active_periods(ox["stay_home"], threshold=2):
        schedule.add("social_distancing", s, e)

    for s, e in _find_active_periods(ox["gathering_restrictions"], threshold=3):
        schedule.add("gathering_ban", s, e)

    from danish_abm.npi import NPIDefinition as _NPIDef
    _hol = danish_school_holiday_multiplier(ox.index)
    _hol_npi = _NPIDef(
        name="school_holiday",
        affected_layers={"classmate": SCHOOL_HOLIDAY_RESIDUAL},
    )
    _on = (_hol < 1.0).astype(int)
    _runs, _s = [], None
    for _i, _v in enumerate(_on):
        if _v and _s is None:
            _s = _i
        elif not _v and _s is not None:
            _runs.append((_s, _i))
            _s = None
    if _s is not None:
        _runs.append((_s, n_days))
    for _a, _b in _runs:
        schedule.add_custom(_hol_npi, _a, _b)
    if _runs:
        _tot = sum(b - a for a, b in _runs)
        print(f"  [oxcgrt] Danish school holidays: {len(_runs)} periods, "
              f"{_tot} of {n_days} days ({100.0 * _tot / max(n_days, 1):.1f}%), "
              f"classmate x{SCHOOL_HOLIDAY_RESIDUAL} — not counted as a policy",
              flush=True)

    return schedule


if __name__ == "__main__":
    df = load_oxcgrt()
    df.to_csv(CLEAN_CSV)
    print(f"Saved clean CSV: {CLEAN_CSV}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Rows: {len(df)}")
    print()

    print_summary("2020-09-01", "2020-12-31")
    print()
    print_summary("2021-01-11", "2021-06-30")
