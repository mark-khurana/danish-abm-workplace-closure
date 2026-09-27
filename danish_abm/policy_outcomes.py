"""Policy-relevant outcome metrics for the Danish ABM."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class HealthcareCapacity:
    """Healthcare system capacity for a country/region."""
    total_hospital_beds: int = 15_000
    total_icu_beds: int = 400
    surge_icu_beds: int = 1_000
    baseline_occupancy: float = 0.75
    icu_fraction: float = 0.25
    avg_hospital_los: float = 8.0
    avg_icu_los: float = 12.0
    avg_hospital_los_ci: Tuple[float, float] = (5.0, 14.0)
    avg_icu_los_ci: Tuple[float, float] = (7.0, 21.0)

    @property
    def available_hospital_beds(self) -> int:
        return int(self.total_hospital_beds * (1 - self.baseline_occupancy))

    @property
    def available_icu_beds(self) -> int:
        return int(self.total_icu_beds * (1 - self.baseline_occupancy))

    @classmethod
    def denmark(cls) -> "HealthcareCapacity":
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> "HealthcareCapacity":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


def compute_hospital_occupancy(
    daily_admissions: np.ndarray,
    avg_los: float = 8.0,
    los_distribution: str = "gamma",
) -> np.ndarray:
    """Compute daily hospital census from admission time series."""
    n_days = len(daily_admissions)
    admissions = np.asarray(daily_admissions, dtype=float)

    if los_distribution == "fixed":
        kernel = np.ones(min(int(avg_los), n_days))
    elif los_distribution == "exponential":
        t = np.arange(n_days)
        kernel = np.exp(-t / avg_los)
    elif los_distribution == "gamma":
        from scipy.stats import gamma as scipy_gamma
        shape = 2.0
        scale = avg_los / shape
        t = np.arange(n_days)
        kernel = 1.0 - scipy_gamma.cdf(t, a=shape, scale=scale)
    else:
        raise ValueError(f"Unknown LOS distribution: {los_distribution}")

    census = np.convolve(admissions, kernel)[:n_days]
    return census


def compute_icu_occupancy(
    daily_admissions: np.ndarray,
    icu_fraction: float = 0.25,
    avg_icu_los: float = 12.0,
) -> np.ndarray:
    """Compute daily ICU census from hospital admissions."""
    icu_admissions = np.asarray(daily_admissions, dtype=float) * icu_fraction
    return compute_hospital_occupancy(icu_admissions, avg_los=avg_icu_los, los_distribution="gamma")


@dataclass
class PolicyMetrics:
    """All policy-relevant metrics computed from a single simulation run."""

    hospital_occupancy: Optional[np.ndarray] = None
    icu_occupancy: Optional[np.ndarray] = None
    peak_hospital_occupancy: float = 0.0
    peak_hospital_day: int = 0
    peak_icu_occupancy: float = 0.0
    peak_icu_day: int = 0
    days_over_hospital_capacity: int = 0
    days_over_icu_capacity: int = 0
    max_hospital_utilization: float = 0.0
    max_icu_utilization: float = 0.0

    time_to_peak_infections: int = 0
    time_to_peak_hosp: int = 0
    time_to_peak_icu: int = 0
    weeks_to_act: float = 0.0
    doubling_time_at_peak: Optional[float] = None

    infections_by_age: Optional[Dict[str, int]] = None
    deaths_by_age: Optional[Dict[str, int]] = None
    hosp_by_age: Optional[Dict[str, int]] = None
    attack_rate_by_age: Optional[Dict[str, float]] = None
    cfr_by_age: Optional[Dict[str, float]] = None

    hotspot_municipalities: Optional[List[Dict[str, Any]]] = None
    geographic_gini: Optional[float] = None

    lives_saved: Optional[int] = None
    hosp_averted: Optional[int] = None
    infections_averted: Optional[int] = None
    nnt_to_prevent_death: Optional[float] = None
    nnt_to_prevent_hosp: Optional[float] = None

    days_to_effect: Optional[int] = None
    cumulative_at_divergence: Optional[int] = None

    @classmethod
    def compute(
        cls,
        daily_infections: np.ndarray,
        daily_deaths: np.ndarray,
        daily_hosp: np.ndarray,
        capacity: HealthcareCapacity = None,
        infections_by_age: Optional[Dict[str, int]] = None,
        deaths_by_age: Optional[Dict[str, int]] = None,
        hosp_by_age: Optional[Dict[str, int]] = None,
        population_by_age: Optional[Dict[str, int]] = None,
        municipality_infections: Optional[Dict[str, float]] = None,
        municipality_population: Optional[Dict[str, int]] = None,
        num_agents: int = 0,
        baseline_daily_infections: Optional[np.ndarray] = None,
        baseline_total_infections: int = 0,
        baseline_total_deaths: int = 0,
        baseline_total_hosp: int = 0,
        total_vaccinated: int = 0,
    ) -> "PolicyMetrics":
        """Compute all policy metrics from simulation output."""
        cap = capacity or HealthcareCapacity.denmark()
        metrics = cls()
        n_days = len(daily_infections)

        infections = np.asarray(daily_infections, dtype=float)
        deaths = np.asarray(daily_deaths, dtype=float)
        hosp = np.asarray(daily_hosp, dtype=float)

        metrics.hospital_occupancy = compute_hospital_occupancy(
            hosp, avg_los=cap.avg_hospital_los
        )
        metrics.icu_occupancy = compute_icu_occupancy(
            hosp, icu_fraction=cap.icu_fraction, avg_icu_los=cap.avg_icu_los
        )

        if len(metrics.hospital_occupancy) > 0:
            metrics.peak_hospital_occupancy = float(metrics.hospital_occupancy.max())
            metrics.peak_hospital_day = int(np.argmax(metrics.hospital_occupancy))

        if len(metrics.icu_occupancy) > 0:
            metrics.peak_icu_occupancy = float(metrics.icu_occupancy.max())
            metrics.peak_icu_day = int(np.argmax(metrics.icu_occupancy))

        avail_hosp = cap.available_hospital_beds
        avail_icu = cap.available_icu_beds

        metrics.days_over_hospital_capacity = int(
            (metrics.hospital_occupancy > avail_hosp).sum()
        )
        metrics.days_over_icu_capacity = int(
            (metrics.icu_occupancy > avail_icu).sum()
        )
        metrics.max_hospital_utilization = (
            metrics.peak_hospital_occupancy / max(avail_hosp, 1)
        )
        metrics.max_icu_utilization = (
            metrics.peak_icu_occupancy / max(avail_icu, 1)
        )

        if infections.sum() > 0:
            metrics.time_to_peak_infections = int(np.argmax(infections))
        if hosp.sum() > 0:
            metrics.time_to_peak_hosp = int(np.argmax(hosp))
        if len(metrics.icu_occupancy) > 0 and metrics.icu_occupancy.sum() > 0:
            metrics.time_to_peak_icu = int(np.argmax(metrics.icu_occupancy))

        half_cap = avail_hosp * 0.5
        exceed_mask = metrics.hospital_occupancy > half_cap
        if exceed_mask.any():
            first_day = int(np.argmax(exceed_mask))
            metrics.weeks_to_act = first_day / 7.0
        else:
            metrics.weeks_to_act = float("inf")

        peak_day = metrics.time_to_peak_infections
        if peak_day > 14:
            window = infections[peak_day - 14 : peak_day]
            nonzero = window[window > 0]
            if len(nonzero) > 3:
                growth_rates = np.diff(np.log(nonzero + 1))
                mean_rate = np.mean(growth_rates)
                if mean_rate > 0:
                    metrics.doubling_time_at_peak = float(np.log(2) / mean_rate)

        AGE_LABELS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
        if infections_by_age:
            metrics.infections_by_age = infections_by_age
        if deaths_by_age:
            metrics.deaths_by_age = deaths_by_age
        if hosp_by_age:
            metrics.hosp_by_age = hosp_by_age

        if infections_by_age and population_by_age:
            metrics.attack_rate_by_age = {}
            for age, count in infections_by_age.items():
                pop = population_by_age.get(age, 1)
                metrics.attack_rate_by_age[age] = count / max(pop, 1)

        if infections_by_age and deaths_by_age:
            metrics.cfr_by_age = {}
            for age in infections_by_age:
                inf = infections_by_age.get(age, 0)
                dth = deaths_by_age.get(age, 0)
                metrics.cfr_by_age[age] = dth / max(inf, 1)

        if municipality_infections:
            muni_items = sorted(
                municipality_infections.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            total_muni_inf = sum(v for _, v in muni_items)
            hotspots = []
            for name, count in muni_items[:10]:
                pop = municipality_population.get(name, 1) if municipality_population else 1
                hotspots.append({
                    "municipality": name,
                    "infections": int(count),
                    "attack_rate": count / max(pop, 1),
                    "share_of_total": count / max(total_muni_inf, 1),
                })
            metrics.hotspot_municipalities = hotspots

            vals = np.array([v for _, v in muni_items], dtype=float)
            if len(vals) > 1 and vals.sum() > 0:
                vals_sorted = np.sort(vals)
                n = len(vals_sorted)
                cum = np.cumsum(vals_sorted)
                metrics.geographic_gini = float(
                    (2 * np.sum((np.arange(1, n + 1) * vals_sorted)) / (n * vals_sorted.sum())) - (n + 1) / n
                )

        total_inf = int(infections.sum())
        total_dth = int(deaths.sum())
        total_hsp = int(hosp.sum())

        if baseline_total_infections > 0:
            metrics.infections_averted = baseline_total_infections - total_inf
        if baseline_total_deaths > 0:
            metrics.lives_saved = baseline_total_deaths - total_dth
        if baseline_total_hosp > 0:
            metrics.hosp_averted = baseline_total_hosp - total_hsp

        if total_vaccinated > 0:
            if metrics.lives_saved and metrics.lives_saved > 0:
                metrics.nnt_to_prevent_death = total_vaccinated / metrics.lives_saved
            if metrics.hosp_averted and metrics.hosp_averted > 0:
                metrics.nnt_to_prevent_hosp = total_vaccinated / metrics.hosp_averted

        if baseline_daily_infections is not None:
            baseline = np.asarray(baseline_daily_infections, dtype=float)
            min_len = min(len(infections), len(baseline))
            cum_self = np.cumsum(infections[:min_len])
            cum_base = np.cumsum(baseline[:min_len])
            diff = np.abs(cum_self - cum_base)
            threshold = max(cum_base[-1] * 0.05, 10)
            diverged = diff > threshold
            if diverged.any():
                metrics.days_to_effect = int(np.argmax(diverged))
                metrics.cumulative_at_divergence = int(cum_self[metrics.days_to_effect])

        return metrics

    def summary(self, sim_start: str = "", scale_factor: float = 1.0) -> str:
        """Human/LLM-readable summary of policy metrics."""
        lines = []
        sf = max(scale_factor, 1e-9)

        lines.append("HEALTHCARE CAPACITY:")
        lines.append(f"  Peak hospital occupancy:  {self.peak_hospital_occupancy / sf:>8,.0f} beds "
                      f"(day {self.peak_hospital_day})")
        lines.append(f"  Peak ICU occupancy:       {self.peak_icu_occupancy / sf:>8,.0f} beds "
                      f"(day {self.peak_icu_day})")
        lines.append(f"  Hospital utilization:     {self.max_hospital_utilization:>8.0%} of capacity")
        lines.append(f"  ICU utilization:          {self.max_icu_utilization:>8.0%} of capacity")
        if self.days_over_icu_capacity > 0:
            lines.append(f"  ⚠ Days over ICU capacity: {self.days_over_icu_capacity}")

        lines.append("")
        lines.append("TIMING:")
        lines.append(f"  Time to peak infections:  {self.time_to_peak_infections} days")
        lines.append(f"  Time to peak hosp:        {self.time_to_peak_hosp} days")
        if self.weeks_to_act < float("inf"):
            lines.append(f"  Weeks to act (50% hosp):  {self.weeks_to_act:.1f} weeks")
        else:
            lines.append(f"  Weeks to act:             Never reaches 50% hospital capacity")
        if self.doubling_time_at_peak is not None:
            lines.append(f"  Doubling time at peak:    {self.doubling_time_at_peak:.1f} days")

        if self.infections_by_age:
            lines.append("")
            lines.append("AGE-STRATIFIED OUTCOMES:")
            lines.append(f"  {'Age':<8s} {'Infections':>12s} {'Deaths':>8s} {'Hosp':>8s} {'CFR':>8s}")
            lines.append(f"  {'─'*48}")
            for age in self.infections_by_age:
                inf = self.infections_by_age.get(age, 0)
                dth = (self.deaths_by_age or {}).get(age, 0)
                hsp = (self.hosp_by_age or {}).get(age, 0)
                cfr = (self.cfr_by_age or {}).get(age, 0)
                lines.append(f"  {age:<8s} {inf:>12,d} {dth:>8,d} {hsp:>8,d} {cfr:>7.2%}")

        if self.hotspot_municipalities:
            lines.append("")
            lines.append("GEOGRAPHIC HOTSPOTS (top 5):")
            for h in self.hotspot_municipalities[:5]:
                lines.append(f"  {h['municipality']}: {h['infections']:,} infections "
                              f"({h['share_of_total']:.1%} of total)")
            if self.geographic_gini is not None:
                lines.append(f"  Geographic Gini: {self.geographic_gini:.2f}")

        if self.lives_saved is not None or self.hosp_averted is not None:
            lines.append("")
            lines.append("COMPARATIVE EFFECTIVENESS:")
            if self.infections_averted is not None:
                sign = "+" if self.infections_averted < 0 else ""
                lines.append(f"  Infections averted:  {sign}{self.infections_averted:>10,d}")
            if self.lives_saved is not None:
                lines.append(f"  Lives saved:         {self.lives_saved:>10,d}")
            if self.hosp_averted is not None:
                lines.append(f"  Hospitalizations averted: {self.hosp_averted:>7,d}")
            if self.nnt_to_prevent_death is not None:
                lines.append(f"  Vaccines per life saved:  {self.nnt_to_prevent_death:>7,.0f}")
            if self.nnt_to_prevent_hosp is not None:
                lines.append(f"  Vaccines per hosp averted: {self.nnt_to_prevent_hosp:>6,.0f}")

        if self.days_to_effect is not None:
            lines.append("")
            lines.append("TIME-TO-EFFECT:")
            lines.append(f"  Scenario diverges from baseline at day {self.days_to_effect}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict (JSON-safe, no numpy arrays)."""
        d = {}

        d["healthcare_capacity"] = {
            "peak_hospital_occupancy": self.peak_hospital_occupancy,
            "peak_hospital_day": self.peak_hospital_day,
            "peak_icu_occupancy": self.peak_icu_occupancy,
            "peak_icu_day": self.peak_icu_day,
            "days_over_hospital_capacity": self.days_over_hospital_capacity,
            "days_over_icu_capacity": self.days_over_icu_capacity,
            "max_hospital_utilization": round(self.max_hospital_utilization, 4),
            "max_icu_utilization": round(self.max_icu_utilization, 4),
        }

        d["timing"] = {
            "time_to_peak_infections": self.time_to_peak_infections,
            "time_to_peak_hosp": self.time_to_peak_hosp,
            "time_to_peak_icu": self.time_to_peak_icu,
            "weeks_to_act": self.weeks_to_act if self.weeks_to_act < 1e6 else None,
            "doubling_time_at_peak": self.doubling_time_at_peak,
        }

        if self.hospital_occupancy is not None:
            d["hospital_occupancy_ts"] = self.hospital_occupancy.tolist()
        if self.icu_occupancy is not None:
            d["icu_occupancy_ts"] = self.icu_occupancy.tolist()

        if self.infections_by_age:
            d["age_stratified"] = {
                "infections": self.infections_by_age,
                "deaths": self.deaths_by_age,
                "hospitalizations": self.hosp_by_age,
                "attack_rate": self.attack_rate_by_age,
                "cfr": self.cfr_by_age,
            }

        if self.hotspot_municipalities:
            d["geographic"] = {
                "hotspots": self.hotspot_municipalities,
                "gini": self.geographic_gini,
            }

        if self.lives_saved is not None:
            d["comparative_effectiveness"] = {
                "infections_averted": self.infections_averted,
                "lives_saved": self.lives_saved,
                "hosp_averted": self.hosp_averted,
                "nnt_death": self.nnt_to_prevent_death,
                "nnt_hosp": self.nnt_to_prevent_hosp,
            }

        if self.days_to_effect is not None:
            d["time_to_effect"] = {
                "days_to_divergence": self.days_to_effect,
                "cumulative_at_divergence": self.cumulative_at_divergence,
            }

        return d


@dataclass
class EnsemblePolicyMetrics:
    """Aggregated policy metrics across ensemble runs with uncertainty."""

    n_runs: int = 0
    ci_pct: float = 90.0

    hospital_occupancy_median: Optional[np.ndarray] = None
    hospital_occupancy_lo: Optional[np.ndarray] = None
    hospital_occupancy_hi: Optional[np.ndarray] = None
    icu_occupancy_median: Optional[np.ndarray] = None
    icu_occupancy_lo: Optional[np.ndarray] = None
    icu_occupancy_hi: Optional[np.ndarray] = None

    peak_hospital: Optional[Dict[str, float]] = None
    peak_icu: Optional[Dict[str, float]] = None
    time_to_peak: Optional[Dict[str, float]] = None
    weeks_to_act: Optional[Dict[str, float]] = None
    days_over_icu: Optional[Dict[str, float]] = None

    infections_by_age: Optional[Dict[str, Dict[str, float]]] = None
    deaths_by_age: Optional[Dict[str, Dict[str, float]]] = None

    @classmethod
    def from_metrics_list(cls, metrics_list: List[PolicyMetrics],
                          ci_pct: float = 90.0) -> "EnsemblePolicyMetrics":
        """Aggregate a list of PolicyMetrics from individual runs."""
        n = len(metrics_list)
        if n == 0:
            return cls()

        lo_pct = (100 - ci_pct) / 2
        hi_pct = 100 - lo_pct
        result = cls(n_runs=n, ci_pct=ci_pct)

        def _scalar_stats(values):
            arr = np.array(values, dtype=float)
            return {
                "median": float(np.median(arr)),
                "lo": float(np.percentile(arr, lo_pct)),
                "hi": float(np.percentile(arr, hi_pct)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }

        hosp_arrays = [m.hospital_occupancy for m in metrics_list if m.hospital_occupancy is not None]
        if hosp_arrays:
            min_len = min(len(a) for a in hosp_arrays)
            stacked = np.array([a[:min_len] for a in hosp_arrays])
            result.hospital_occupancy_median = np.median(stacked, axis=0)
            result.hospital_occupancy_lo = np.percentile(stacked, lo_pct, axis=0)
            result.hospital_occupancy_hi = np.percentile(stacked, hi_pct, axis=0)

        icu_arrays = [m.icu_occupancy for m in metrics_list if m.icu_occupancy is not None]
        if icu_arrays:
            min_len = min(len(a) for a in icu_arrays)
            stacked = np.array([a[:min_len] for a in icu_arrays])
            result.icu_occupancy_median = np.median(stacked, axis=0)
            result.icu_occupancy_lo = np.percentile(stacked, lo_pct, axis=0)
            result.icu_occupancy_hi = np.percentile(stacked, hi_pct, axis=0)

        result.peak_hospital = _scalar_stats([m.peak_hospital_occupancy for m in metrics_list])
        result.peak_icu = _scalar_stats([m.peak_icu_occupancy for m in metrics_list])
        result.time_to_peak = _scalar_stats([m.time_to_peak_infections for m in metrics_list])
        finite_weeks = [m.weeks_to_act for m in metrics_list if m.weeks_to_act < 1e6]
        if finite_weeks:
            result.weeks_to_act = _scalar_stats(finite_weeks)
        result.days_over_icu = _scalar_stats([m.days_over_icu_capacity for m in metrics_list])

        age_inf_all = [m.infections_by_age for m in metrics_list if m.infections_by_age]
        if age_inf_all:
            age_groups = list(age_inf_all[0].keys())
            result.infections_by_age = {}
            result.deaths_by_age = {}
            for ag in age_groups:
                inf_vals = [d.get(ag, 0) for d in age_inf_all]
                result.infections_by_age[ag] = _scalar_stats(inf_vals)
            death_all = [m.deaths_by_age for m in metrics_list if m.deaths_by_age]
            if death_all:
                for ag in age_groups:
                    dth_vals = [d.get(ag, 0) for d in death_all]
                    result.deaths_by_age[ag] = _scalar_stats(dth_vals)

        return result

    def summary(self) -> str:
        lines = [f"Ensemble Policy Metrics ({self.n_runs} runs, {self.ci_pct:.0f}% CI)"]
        ci = f"{self.ci_pct:.0f}%"

        if self.peak_hospital:
            ph = self.peak_hospital
            lines.append(f"  Peak hospital:  {ph['median']:>8,.0f}  [{ph['lo']:,.0f} - {ph['hi']:,.0f}]")
        if self.peak_icu:
            pi = self.peak_icu
            lines.append(f"  Peak ICU:       {pi['median']:>8,.0f}  [{pi['lo']:,.0f} - {pi['hi']:,.0f}]")
        if self.time_to_peak:
            tp = self.time_to_peak
            lines.append(f"  Time to peak:   {tp['median']:>8,.0f}d [{tp['lo']:,.0f} - {tp['hi']:,.0f}]")
        if self.weeks_to_act:
            wa = self.weeks_to_act
            lines.append(f"  Weeks to act:   {wa['median']:>8.1f}w [{wa['lo']:.1f} - {wa['hi']:.1f}]")
        if self.days_over_icu:
            di = self.days_over_icu
            lines.append(f"  Days over ICU:  {di['median']:>8,.0f}  [{di['lo']:,.0f} - {di['hi']:,.0f}]")

        if self.infections_by_age:
            lines.append("")
            lines.append(f"  {'Age':<8s} {'Infections (median)':>20s} {ci + ' CI':>20s}")
            for ag, stats in self.infections_by_age.items():
                lines.append(f"  {ag:<8s} {stats['median']:>20,.0f} [{stats['lo']:>8,.0f} - {stats['hi']:>8,.0f}]")

        return "\n".join(lines)
