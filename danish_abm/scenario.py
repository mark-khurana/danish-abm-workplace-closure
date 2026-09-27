"""Scenario system for the Danish ABM."""

from __future__ import annotations

import copy
import gc
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import OmegaConf

from .npi import NPISchedule, BUILTIN_NPIS
from .policy_outcomes import PolicyMetrics, HealthcareCapacity, EnsemblePolicyMetrics
from .uncertainty import UncertaintyConfig, UncertaintyMode, ParameterSample
from .triggers import (
    TriggerPolicy, TriggerEngine, TriggerCondition, TriggerMetric,
    TriggerOperator, get_trigger_preset, TRIGGER_PRESETS,
)


def _resilient_torch_save(obj, path, retries: int = 5, base_delay: float = 0.5):
    """torch.save with retries on transient file-system errors."""
    parent = os.path.dirname(os.path.abspath(path))
    last_err = None
    for attempt in range(retries):
        try:
            os.makedirs(parent, exist_ok=True)
            torch.save(obj, path)
            return
        except (OSError, RuntimeError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_err


from danish_abm.substeps.utils import interv_schedule_path


Q_LAYER_NAMES = ["family", "household", "neighborhood", "colleague", "classmate"]

WORKPLACE_SHARE_FITTED = 0.12


def _reweight_workplace_q(q_layers, share, fitted=WORKPLACE_SHARE_FITTED,
                          layer="colleague"):
    """Reweight q_layers for a sampled workplace transmission share, mean-1 preserving."""
    if share is None:
        return q_layers
    try:
        vals = [float(v) for v in q_layers]
    except (TypeError, ValueError):
        return q_layers
    if len(vals) != len(Q_LAYER_NAMES):
        raise ValueError(
            f"q_layers has {len(vals)} entries but Q_LAYER_NAMES has "
            f"{len(Q_LAYER_NAMES)} ({Q_LAYER_NAMES}). A layer was added or reordered "
            f"and this reweighting would hit the wrong one. Refusing to guess.")
    idx = Q_LAYER_NAMES.index(layer)
    mean0 = sum(vals) / len(vals)
    if mean0 <= 0 or fitted <= 0:
        return q_layers
    vals[idx] *= float(share) / float(fitted)
    mean1 = sum(vals) / len(vals)
    if mean1 <= 0:
        return q_layers
    return [v * mean0 / mean1 for v in vals]


def _perturb_spillover_logits(logits, z, floor=0.20):
    """Apply a standardised deviate to the displacement matrix using its declared stds."""
    if z is None or abs(float(z)) < 1e-12:
        return logits
    import math
    from danish_abm.npi import build_spillover_matrix
    try:
        _, means, stds = build_spillover_matrix()
        means = means.tolist()
        stds = stds.tolist()
    except Exception:
        return logits

    def _rows(obj):
        return [list(r) for r in obj]

    try:
        cur = _rows(logits)
    except TypeError:
        return logits
    if len(cur) != len(means) or any(len(r) != len(means[0]) for r in cur):
        raise ValueError(
            f"spillover_logits is {len(cur)}x{len(cur[0]) if cur else 0} but the prior "
            f"matrix is {len(means)}x{len(means[0])}. An NPI or a layer was added and "
            f"this perturbation would address the wrong cell. Refusing to guess.")
    zf = float(z)
    out = []
    for i, row in enumerate(cur):
        new = []
        for j, lg in enumerate(row):
            mult = math.exp(float(lg)) + zf * float(stds[i][j])
            new.append(math.log(max(mult, floor)))
        out.append(new)
    return out


@dataclass


class ScenarioSpec:
    """A lightweight policy scenario description."""

    name: str = "Untitled scenario"
    description: str = ""

    npis: List[dict] = field(default_factory=list)

    npi_base: Optional[str] = None
    remove_npis: List[str] = field(default_factory=list)

    trigger_policies: List[dict] = field(default_factory=list)
    trigger_preset: Optional[str] = None

    vaccination: dict = field(default_factory=lambda: {
        "enabled": True,
        "capacity_per_day": 2500,
        "start_date": None,
        "vaccine_type": "mrna_covid",
        "strategy": None,
        "schedule_path": None,
    })

    testing: dict = field(default_factory=lambda: {
        "enabled": True,
    })

    pathogen_preset: Optional[str] = None
    compartment_config: Optional[dict] = None

    surveillance: dict = field(default_factory=lambda: {
        "policy": None,
        "total_daily_tests": 5000,
        "prevalence_threshold": 0.01,
        "surrogate_update_frequency": 7,
    })

    spillover_enabled: bool = True
    spillover_scale: float = 1.0

    spillover_disabled_npis: Optional[List[str]] = None

    initial_infection_ratio: Optional[float] = None
    importation_rate: Optional[float] = None

    compare_to: Optional[str] = None


    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ScenarioSpec":
        """Load a scenario spec from YAML."""
        raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
        return cls._from_dict(raw.get("scenario", raw))

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSpec":
        """Build from a dict (e.g., LLM-generated JSON)."""
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> "ScenarioSpec":
        spec = cls()
        spec.name = d.get("name", spec.name)
        spec.description = d.get("description", "")
        spec.npis = d.get("npis", [])
        spec.npi_base = d.get("npi_base")
        spec.remove_npis = d.get("remove_npis", [])
        spec.trigger_policies = d.get("trigger_policies", [])
        spec.trigger_preset = d.get("trigger_preset")
        if "vaccination" in d:
            spec.vaccination.update(d["vaccination"])
        if "testing" in d:
            spec.testing.update(d["testing"])
        spec.pathogen_preset = d.get("pathogen_preset")
        spec.compartment_config = d.get("compartment_config")
        if "surveillance" in d:
            spec.surveillance.update(d["surveillance"])
        spec.initial_infection_ratio = d.get("initial_infection_ratio")
        spec.importation_rate = d.get("importation_rate")
        spec.compare_to = d.get("compare_to")
        return spec

    @classmethod
    def baseline(cls) -> "ScenarioSpec":
        """The default 2021 Denmark baseline scenario."""
        return cls(
            name="Baseline (2021 Denmark)",
            description="Calibrated baseline with real NPI schedule",
            npis=[
                {"name": "lockdown_hard", "start": "2021-01-01", "end": "2021-01-28"},
                {"name": "school_closure", "start": "2021-01-01", "end": "2021-02-14"},
                {"name": "partial_workplace_closure", "start": "2021-01-28", "end": "2021-04-06"},
                {"name": "mask_mandate", "start": "2021-01-01", "end": "2021-05-20"},
                {"name": "gathering_ban", "start": "2021-01-01", "end": "2021-04-06"},
                {"name": "social_distancing", "start": "2021-04-06", "end": "2021-05-20"},
                {"name": "travel_restrictions", "start": "2021-01-01", "end": "2021-06-30"},
            ],
        )


    def validate(self) -> List[str]:
        """Check for errors. Returns list of error strings (empty = valid)."""
        errors = []

        if self.npi_base and self.npi_base != "denmark_historical":
            errors.append(
                f"Unknown npi_base '{self.npi_base}'. Available: 'denmark_historical'"
            )
        if self.remove_npis and not self.npi_base:
            errors.append("remove_npis requires npi_base to be set")
        for npi_name in self.remove_npis:
            if npi_name not in BUILTIN_NPIS:
                errors.append(
                    f"remove_npis: unknown NPI '{npi_name}'. "
                    f"Available: {', '.join(sorted(BUILTIN_NPIS.keys()))}"
                )

        for i, npi in enumerate(self.npis):
            if "name" not in npi:
                errors.append(f"NPI #{i}: missing 'name'")
            elif npi["name"] not in BUILTIN_NPIS:
                errors.append(
                    f"NPI #{i}: unknown NPI '{npi['name']}'. "
                    f"Available: {', '.join(sorted(BUILTIN_NPIS.keys()))}"
                )
            if "start" not in npi:
                errors.append(f"NPI #{i} ({npi.get('name', '?')}): missing 'start' date")

        vax = self.vaccination
        if vax.get("capacity_per_day", 0) < 0:
            errors.append("Vaccination capacity cannot be negative")

        test = self.testing
        compliance = test.get("isolation_compliance", 0.8)
        if not 0 <= compliance <= 1:
            errors.append(f"Isolation compliance must be 0-1, got {compliance}")
        rate = test.get("daily_rate", 0.015)
        if not 0 <= rate <= 1:
            errors.append(f"Daily testing rate must be 0-1, got {rate}")

        if self.initial_infection_ratio is not None and self.initial_infection_ratio < 0:
            errors.append("Initial infection ratio cannot be negative")

        if self.pathogen_preset:
            from .pathogen_presets import PATHOGEN_PRESETS
            if self.pathogen_preset not in PATHOGEN_PRESETS:
                errors.append(
                    f"Unknown pathogen preset '{self.pathogen_preset}'. "
                    f"Available: {', '.join(sorted(PATHOGEN_PRESETS.keys()))}"
                )

        surv = self.surveillance
        surv_policy = surv.get("policy")
        if surv_policy is not None:
            from .surveillance import SURVEILLANCE_POLICIES
            if surv_policy not in SURVEILLANCE_POLICIES:
                errors.append(
                    f"Unknown surveillance policy '{surv_policy}'. "
                    f"Available: {', '.join(sorted(SURVEILLANCE_POLICIES.keys()))}"
                )
            total_tests = surv.get("total_daily_tests", 5000)
            if total_tests < 0:
                errors.append(f"Surveillance total_daily_tests must be >= 0, got {total_tests}")

        if self.trigger_preset and self.trigger_preset not in TRIGGER_PRESETS:
            errors.append(
                f"Unknown trigger preset '{self.trigger_preset}'. "
                f"Available: {', '.join(sorted(TRIGGER_PRESETS.keys()))}"
            )

        for i, tp in enumerate(self.trigger_policies):
            try:
                policy = TriggerPolicy.from_dict(tp)
                errors.extend(policy.validate())
            except (KeyError, ValueError) as e:
                errors.append(f"Trigger policy #{i}: {e}")

        return errors


    @property
    def has_triggers(self) -> bool:
        """Whether this scenario has any trigger-based reactive policies."""
        return bool(self.trigger_policies) or bool(self.trigger_preset)

    def build_trigger_engine(
        self, n_days: int, num_agents: int = 0,
        healthcare_capacity: Optional[dict] = None,
    ) -> Optional[TriggerEngine]:
        """Build a TriggerEngine from this scenario's trigger specs."""
        policies = []

        if self.trigger_preset:
            policies.extend(get_trigger_preset(self.trigger_preset))

        for tp_dict in self.trigger_policies:
            policies.append(TriggerPolicy.from_dict(tp_dict))

        if not policies:
            return None

        return TriggerEngine(
            policies=policies,
            n_days=n_days,
            num_agents=num_agents,
            healthcare_capacity=healthcare_capacity,
        )


    def build_npi_schedule(self, sim_start: str, n_days: int,
                           sim_end: str = None) -> NPISchedule:
        """Convert date-based NPI specs to day-indexed NPISchedule."""
        if self.npi_base == "denmark_historical":
            if sim_end is None:
                from datetime import timedelta
                end_dt = datetime.strptime(sim_start, "%Y-%m-%d") + timedelta(days=n_days)
                sim_end = end_dt.strftime("%Y-%m-%d")
            from .data.oxcgrt.prep_oxcgrt import oxcgrt_to_npi_schedule
            base_schedule = oxcgrt_to_npi_schedule(sim_start, sim_end)
            if self.remove_npis:
                return base_schedule.counterfactual(remove=self.remove_npis)
            return base_schedule

        schedule = NPISchedule(n_days=n_days)
        start_dt = datetime.strptime(sim_start, "%Y-%m-%d")

        for npi_entry in self.npis:
            name = npi_entry["name"]
            day_start = self._date_to_day(npi_entry["start"], start_dt)
            if "end" in npi_entry and npi_entry["end"] is not None:
                day_end = self._date_to_day(npi_entry["end"], start_dt)
            else:
                day_end = n_days

            day_start = max(0, day_start)
            day_end = min(n_days, day_end)

            if day_start >= day_end:
                continue

            override = {}
            if "strength" in npi_entry:
                strength = float(npi_entry["strength"])
                base_npi = BUILTIN_NPIS[name]
                scaled_layers = {}
                for layer, mult in base_npi.affected_layers.items():
                    scaled_layers[layer] = 1.0 - strength * (1.0 - mult)
                override["affected_layers"] = scaled_layers
                comm = base_npi.community_effect
                override["community_effect"] = 1.0 - strength * (1.0 - comm)

            schedule.add(name, day_start, day_end, override_effects=override or None)

        return schedule

    @staticmethod
    def _date_to_day(date_str: str, sim_start: datetime) -> int:
        """Convert an ISO date string to a simulation day index."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return (dt - sim_start).days


    def to_dict(self) -> dict:
        d = {
            "scenario": {
                "name": self.name,
                "description": self.description,
                "npis": self.npis,
                "vaccination": self.vaccination,
                "testing": self.testing,
                "initial_infection_ratio": self.initial_infection_ratio,
                "importation_rate": self.importation_rate,
                "compare_to": self.compare_to,
            }
        }
        if self.npi_base:
            d["scenario"]["npi_base"] = self.npi_base
        if self.remove_npis:
            d["scenario"]["remove_npis"] = self.remove_npis
        if self.pathogen_preset:
            d["scenario"]["pathogen_preset"] = self.pathogen_preset
        if self.compartment_config:
            d["scenario"]["compartment_config"] = self.compartment_config
        if self.surveillance.get("policy") is not None:
            d["scenario"]["surveillance"] = self.surveillance
        if self.trigger_policies:
            d["scenario"]["trigger_policies"] = self.trigger_policies
        if self.trigger_preset:
            d["scenario"]["trigger_preset"] = self.trigger_preset
        return d

    def to_yaml(self, path: Union[str, Path]):
        """Save as YAML."""
        OmegaConf.save(OmegaConf.create(self.to_dict()), str(path))

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [f"Scenario: {self.name}"]
        if self.description:
            lines.append(f"  {self.description}")
        if self.pathogen_preset:
            lines.append(f"  Pathogen: {self.pathogen_preset}")
        elif self.compartment_config:
            comp_names = self.compartment_config.get("compartments", [])
            lines.append(f"  Custom compartments: {' -> '.join(comp_names)}")
        if self.npi_base:
            lines.append(f"  NPI base: {self.npi_base}")
            if self.remove_npis:
                lines.append(f"  Removed NPIs: {', '.join(self.remove_npis)}")
        lines.append(f"  NPIs: {len(self.npis)}")
        for npi in self.npis:
            strength = f" (strength={npi['strength']})" if "strength" in npi else ""
            end = npi.get("end", "end of sim")
            lines.append(f"    {npi['name']}: {npi['start']} to {end}{strength}")
        vax = self.vaccination
        lines.append(f"  Vaccination: {'ON' if vax.get('enabled', True) else 'OFF'}"
                      f" ({vax.get('capacity_per_day', 0)} doses/day)")
        test = self.testing
        lines.append(f"  Testing: {'ON' if test.get('enabled', True) else 'OFF'}"
                      f" ({test.get('daily_rate', 0):.1%}/day, "
                      f"{test.get('isolation_compliance', 0):.0%} compliance)")
        if self.trigger_preset:
            lines.append(f"  Trigger preset: {self.trigger_preset}")
        if self.trigger_policies:
            lines.append(f"  Trigger policies: {len(self.trigger_policies)}")
            for tp in self.trigger_policies:
                cond = tp.get("condition", {})
                lines.append(
                    f"    {tp.get('name', '?')}: "
                    f"IF {cond.get('metric', '?')} {cond.get('operator', '>')} "
                    f"{cond.get('threshold', '?')} → {tp.get('action', '?')}"
                )
        surv = self.surveillance
        if surv.get("policy") is not None:
            lines.append(f"  Surveillance: {surv['policy']} "
                          f"({surv.get('total_daily_tests', 5000)} tests/day, "
                          f"threshold={surv.get('prevalence_threshold', 0.01)})")
        return "\n".join(lines)


@dataclass
class ScenarioResult:
    """Structured results from a scenario run."""

    scenario_name: str = ""
    sim_start: str = ""
    sim_end: str = ""
    n_days: int = 0
    num_agents: int = 0
    scale_factor: float = 1.0
    elapsed_seconds: float = 0.0

    total_infections: int = 0
    total_deaths: int = 0
    total_hospitalizations: int = 0
    total_vaccinated: int = 0
    total_tests: int = 0
    total_detected: int = 0

    peak_daily_infections: float = 0.0
    peak_infection_day: int = 0
    peak_infection_date: str = ""
    peak_daily_deaths: float = 0.0
    peak_daily_hosp: float = 0.0

    layer_pressure: dict = field(default_factory=dict)

    final_susceptible_pct: float = 0.0
    final_recovered_pct: float = 0.0
    final_dead_pct: float = 0.0

    daily_infections: Optional[list] = None
    daily_deaths: Optional[list] = None
    daily_hosp: Optional[list] = None
    daily_detected: Optional[list] = None
    daily_tests: Optional[list] = None
    daily_vaccinated: Optional[list] = None

    iar_timeseries: Optional[list] = None
    iar_mean: Optional[float] = None

    strain_totals: Optional[Dict[str, int]] = None

    municipality_totals: Optional[Dict[str, float]] = None
    municipality_daily_infections: Optional[List[List[float]]] = None

    infections_by_age: Optional[Dict[str, int]] = None
    deaths_by_age: Optional[Dict[str, int]] = None
    hosp_by_age: Optional[Dict[str, int]] = None
    population_by_age: Optional[Dict[str, int]] = None

    outcome_analysis: Optional[Any] = None

    policy_metrics: Optional[Any] = None

    parameter_sample: Optional[Any] = None

    trigger_log: Optional[List[dict]] = None
    trigger_active_per_day: Optional[List[list]] = None

    delta_infections: Optional[int] = None
    delta_deaths: Optional[int] = None
    delta_hosp: Optional[int] = None
    delta_infections_pct: Optional[float] = None
    delta_deaths_pct: Optional[float] = None
    baseline_name: Optional[str] = None


    @property
    def attack_rate(self) -> float:
        """Fraction of population infected."""
        return self.total_infections / max(self.num_agents, 1)

    @property
    def case_fatality_rate(self) -> float:
        """Deaths / infections."""
        return self.total_deaths / max(self.total_infections, 1)

    @property
    def infections_real_scale(self) -> int:
        """Total infections scaled to real population."""
        return int(self.total_infections / max(self.scale_factor, 1e-9))

    @property
    def deaths_real_scale(self) -> int:
        return int(self.total_deaths / max(self.scale_factor, 1e-9))

    @property
    def hospitalizations_real_scale(self) -> int:
        return int(self.total_hospitalizations / max(self.scale_factor, 1e-9))


    def compute_policy_metrics(
        self,
        capacity: HealthcareCapacity = None,
        baseline: "ScenarioResult" = None,
    ) -> PolicyMetrics:
        """Compute full policy metrics from this result."""
        daily_inf = np.array(self.daily_infections or [0] * self.n_days, dtype=float)
        daily_dth = np.array(self.daily_deaths or [0] * self.n_days, dtype=float)
        daily_hsp = np.array(self.daily_hosp or [0] * self.n_days, dtype=float)

        baseline_inf = None
        baseline_total_inf = 0
        baseline_total_dth = 0
        baseline_total_hsp = 0
        if baseline is not None:
            baseline_inf = np.array(baseline.daily_infections or [], dtype=float)
            baseline_total_inf = baseline.total_infections
            baseline_total_dth = baseline.total_deaths
            baseline_total_hsp = baseline.total_hospitalizations

        self.policy_metrics = PolicyMetrics.compute(
            daily_infections=daily_inf,
            daily_deaths=daily_dth,
            daily_hosp=daily_hsp,
            capacity=capacity,
            infections_by_age=self.infections_by_age,
            deaths_by_age=self.deaths_by_age,
            hosp_by_age=self.hosp_by_age,
            population_by_age=self.population_by_age,
            municipality_infections=self.municipality_totals,
            num_agents=self.num_agents,
            baseline_daily_infections=baseline_inf,
            baseline_total_infections=baseline_total_inf,
            baseline_total_deaths=baseline_total_dth,
            baseline_total_hosp=baseline_total_hsp,
            total_vaccinated=self.total_vaccinated,
        )
        return self.policy_metrics


    def compare(self, baseline: "ScenarioResult") -> "ScenarioResult":
        """Fill in delta fields by comparing to a baseline result."""
        self.baseline_name = baseline.scenario_name
        self.delta_infections = self.total_infections - baseline.total_infections
        self.delta_deaths = self.total_deaths - baseline.total_deaths
        self.delta_hosp = self.total_hospitalizations - baseline.total_hospitalizations
        self.delta_infections_pct = (
            self.delta_infections / max(baseline.total_infections, 1) * 100
        )
        self.delta_deaths_pct = (
            self.delta_deaths / max(baseline.total_deaths, 1) * 100
        )
        return self


    def summary(self) -> str:
        """Concise human/LLM-readable summary."""
        lines = []
        lines.append(f"=== {self.scenario_name} ===")
        lines.append(f"Period: {self.sim_start} to {self.sim_end} ({self.n_days} days)")
        lines.append(f"Population: {self.num_agents:,} agents "
                      f"(representing {int(self.num_agents / max(self.scale_factor, 1e-9)):,})")
        lines.append("")

        lines.append("OUTCOMES (model scale):")
        lines.append(f"  Infections:       {self.total_infections:>10,}")
        lines.append(f"  Deaths:           {self.total_deaths:>10,}")
        lines.append(f"  Hospitalizations: {self.total_hospitalizations:>10,}")
        if self.total_vaccinated > 0:
            lines.append(f"  Vaccinated:       {self.total_vaccinated:>10,}")
        if self.total_detected > 0:
            lines.append(f"  Detected cases:   {self.total_detected:>10,}")
        lines.append("")

        lines.append("OUTCOMES (real population scale):")
        lines.append(f"  Infections:       {self.infections_real_scale:>10,}")
        lines.append(f"  Deaths:           {self.deaths_real_scale:>10,}")
        lines.append(f"  Hospitalizations: {self.hospitalizations_real_scale:>10,}")
        lines.append("")

        lines.append("KEY METRICS:")
        lines.append(f"  Attack rate:         {self.attack_rate:.1%}")
        lines.append(f"  Case fatality rate:  {self.case_fatality_rate:.2%}")
        lines.append(f"  Peak infections:     {self.peak_daily_infections:,.0f}/day "
                      f"(day {self.peak_infection_day}, {self.peak_infection_date})")
        lines.append("")

        lines.append("FINAL STATE:")
        lines.append(f"  Susceptible: {self.final_susceptible_pct:.1%}")
        lines.append(f"  Recovered:   {self.final_recovered_pct:.1%}")
        lines.append(f"  Dead:        {self.final_dead_pct:.2%}")

        if self.delta_infections is not None:
            lines.append("")
            lines.append(f"COMPARED TO {self.baseline_name}:")
            sign_i = "+" if self.delta_infections >= 0 else ""
            sign_d = "+" if self.delta_deaths >= 0 else ""
            sign_h = "+" if self.delta_hosp >= 0 else ""
            lines.append(f"  Infections:       {sign_i}{self.delta_infections:,} "
                          f"({sign_i}{self.delta_infections_pct:.1f}%)")
            lines.append(f"  Deaths:           {sign_d}{self.delta_deaths:,} "
                          f"({sign_d}{self.delta_deaths_pct:.1f}%)")
            lines.append(f"  Hospitalizations: {sign_h}{self.delta_hosp:,}")

        if self.strain_totals:
            lines.append("")
            lines.append("STRAIN BREAKDOWN:")
            for strain, count in self.strain_totals.items():
                lines.append(f"  {strain}: {count:,}")

        if self.trigger_log:
            lines.append("")
            lines.append("TRIGGER POLICY EVENTS:")
            start_dt = datetime.strptime(self.sim_start, "%Y-%m-%d")
            for event in self.trigger_log:
                day = event["day"]
                date = (start_dt + timedelta(days=day)).strftime("%Y-%m-%d")
                reason = f" ({event['reason']})" if event.get("reason") else ""
                lines.append(f"  Day {day:>3d} ({date}): {event['policy']} {event['event']}{reason}")
            if self.trigger_active_per_day:
                active_days = sum(1 for d in self.trigger_active_per_day if d)
                lines.append(f"  Total days with active triggers: {active_days}/{self.n_days}")

        if self.policy_metrics:
            lines.append("")
            lines.append(self.policy_metrics.summary(
                sim_start=self.sim_start,
                scale_factor=self.scale_factor,
            ))

        lines.append("")
        lines.append(f"Runtime: {self.elapsed_seconds:.1f}s")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Full result as a dict (for JSON serialization)."""
        d = {
            "scenario_name": self.scenario_name,
            "period": {"start": self.sim_start, "end": self.sim_end, "n_days": self.n_days},
            "population": {"agents": self.num_agents, "scale_factor": self.scale_factor},
            "outcomes": {
                "infections": self.total_infections,
                "deaths": self.total_deaths,
                "hospitalizations": self.total_hospitalizations,
                "vaccinated": self.total_vaccinated,
                "detected": self.total_detected,
            },
            "outcomes_real_scale": {
                "infections": self.infections_real_scale,
                "deaths": self.deaths_real_scale,
                "hospitalizations": self.hospitalizations_real_scale,
            },
            "peak": {
                "daily_infections": self.peak_daily_infections,
                "day": self.peak_infection_day,
                "date": self.peak_infection_date,
            },
            "rates": {
                "attack_rate": round(self.attack_rate, 4),
                "case_fatality_rate": round(self.case_fatality_rate, 4),
            },
            "final_state": {
                "susceptible_pct": round(self.final_susceptible_pct, 4),
                "recovered_pct": round(self.final_recovered_pct, 4),
                "dead_pct": round(self.final_dead_pct, 6),
            },
            "runtime_seconds": round(self.elapsed_seconds, 1),
        }
        if self.delta_infections is not None:
            d["comparison"] = {
                "baseline": self.baseline_name,
                "delta_infections": self.delta_infections,
                "delta_deaths": self.delta_deaths,
                "delta_hospitalizations": self.delta_hosp,
                "delta_infections_pct": round(self.delta_infections_pct, 1),
                "delta_deaths_pct": round(self.delta_deaths_pct, 1),
            }
        if self.strain_totals:
            d["strains"] = self.strain_totals
        if self.trigger_log:
            d["trigger_events"] = self.trigger_log
        return d

    def save(self, path: Union[str, Path]):
        """Save result as JSON."""
        with open(str(path), "w") as f:
            json.dump(self.to_dict(), f, indent=2)


SPEC_OVERRIDE_MAP = {
    "testing": (
        ("enabled",                "testing_enabled",                 True),
        ("daily_rate",             "base_testing_rate",               0.015),
        ("isolation_compliance",   "isolation_compliance",            0.8),
        ("isolation_duration",     "isolation_duration",              10),
        ("symptomatic_multiplier", "symptomatic_testing_multiplier",  5.0),
    ),
    "vaccination": (
        ("vaccine_type",           "vaccination_vaccine_type",        "mrna_covid"),
        ("age_efficacy",           "vaccination_age_efficacy",        {"7": 0.75}),
    ),
}


def _assert_register_vaccination(meta, dc, vax):
    """A wave with a register vaccination file must run the register schedule."""
    if not vax.get("enabled", True) or vax.get("strategy") is not None:
        return
    if getattr(dc, "individual_vaccination_csv", None) or getattr(dc, "vaccination_schedule_pt", None):
        if meta.get("vaccination_strategy") != "registry" or not meta.get("vaccination_schedule_path"):
            raise RuntimeError(
                "this wave has a register vaccination file but the scenario would run "
                f"vaccination_strategy={meta.get('vaccination_strategy')!r} "
                f"(schedule {meta.get('vaccination_schedule_path')!r}). Refusing: the fit "
                "used the register, so a scenario on any other rollout is not comparable.")


def apply_spec_overrides(meta, spec_dict, group, log=None):
    """Write a spec group's keys into meta: spec wins, else config, else fallback."""
    inherited = []
    for spec_key, meta_key, fallback in SPEC_OVERRIDE_MAP[group]:
        if spec_dict and spec_key in spec_dict:
            meta[meta_key] = spec_dict[spec_key]
        elif meta_key in meta and meta[meta_key] is not None:
            inherited.append(meta_key)
        else:
            meta[meta_key] = fallback
    if inherited and log:
        log(f"  [scenario] inherited from the wave config (not overridden by the "
            f"{group} spec): {', '.join(sorted(inherited))}")
    return inherited


def npi_active_days(active, names=None):
    """Active-day count per NPI, as {name: n_days}."""
    if active is None:
        return {}
    if names is None:
        from danish_abm.npi import SPILLOVER_NPI_NAMES
        names = list(SPILLOVER_NPI_NAMES)
    counts = {}
    for i, nm in enumerate(names):
        if i < active.shape[1]:
            counts[nm] = int((active[:, i] > 0.5).sum())
    return counts


def _suppress_named_spillover(active, off_names, old_data=None):
    """Zero npi_active for the NAMED NPIs only, leaving the rest in force."""
    names = None
    if isinstance(old_data, dict):
        names = old_data.get("npi_active_names")
    if not names:
        from danish_abm.npi import SPILLOVER_NPI_NAMES
        names = list(SPILLOVER_NPI_NAMES)
    index = {nm: i for i, nm in enumerate(names)}

    unknown = [nm for nm in off_names if nm not in index]
    if unknown:
        raise ValueError(
            f"spillover_disabled_npis contains {unknown}, which are not in "
            f"npi_active_names {list(names)}. Suppressing nothing would make the "
            f"arm identical to arm A and report a measured displacement cost of "
            f"zero, so this must fail rather than pass silently.")

    active = active.clone()
    zeroed = []
    for nm in off_names:
        i = index[nm]
        if i < active.shape[1]:
            n_before = int((active[:, i] > 0.5).sum())
            active[:, i] = 0.0
            zeroed.append(f"{nm}({n_before}d)")
    print(f"  [schedule] displacement suppressed for {', '.join(zeroed)} — "
          f"every other NPI's spillover remains in force")
    return active


def _schedule_payload(npi_tensor, npi_schedule, old_data=None, spec=None):
    """Build the intervention_schedule.pt payload, PRESERVING npi_active."""
    payload = {"multipliers": npi_tensor}

    n_entries = len(getattr(npi_schedule, "entries", []) or [])

    active = None
    try:
        active = npi_schedule.build_npi_active_tensor()
        if active is not None and float(active.sum()) == 0.0 and n_entries > 0:
            active = None
    except Exception:
        active = None

    if active is None and n_entries == 0:
        if isinstance(old_data, dict) and old_data.get("npi_active") is not None:
            active = torch.zeros_like(old_data["npi_active"])
            print("  [schedule] empty NPI schedule — npi_active zeroed (not "
                  "inherited from the previous scenario)")
    elif active is None and isinstance(old_data, dict):
        active = old_data.get("npi_active")

    if active is not None and spec is not None:
        _off = getattr(spec, "spillover_disabled_npis", None)
        if _off:
            active = _suppress_named_spillover(active, _off, old_data)
        if not getattr(spec, "spillover_enabled", True):
            active = torch.zeros_like(active)
            print("  [schedule] spillover_enabled=False — displacement disabled "
                  "for EVERY NPI while closures remain in force. This is a "
                  "whole-package switch; for a per-policy contrast use "
                  "spillover_disabled_npis instead.")
        else:
            sc = float(getattr(spec, "spillover_scale", 1.0))
            if sc != 1.0:
                payload["spillover_scale"] = float(sc)
                print(f"  [schedule] spillover_scale={sc:g} "
                      f"(displacement strength scaled in log space)")

    if active is not None:
        payload["npi_active"] = active
        if isinstance(old_data, dict) and "npi_active_names" in old_data:
            payload["npi_active_names"] = old_data["npi_active_names"]
    ents = []
    for e in (getattr(npi_schedule, "entries", []) or []):
        npi = e.get("npi") if isinstance(e, dict) else None
        if npi is None or not hasattr(npi, "name"):
            continue
        ents.append({"name": str(npi.name), "day_start": int(e["day_start"]),
                     "day_end": int(e["day_end"]),
                     "affected_layers": {str(k): float(v)
                                         for k, v in (getattr(npi, "affected_layers", None) or {}).items()}})
    payload["entries"] = ents
    return payload


class ScenarioRunner:
    """Runs policy scenarios against a pre-calibrated model."""

    def __init__(self, config: dict, cal_params: dict, r2: torch.Tensor,
                 data_config=None):
        self.base_config = config
        self.cal_params = cal_params
        self.r2 = r2
        self.data_config = data_config

        meta = config["simulation_metadata"]
        self.n_days = meta["num_steps_per_episode"]
        self.n_agents = meta["num_agents"]
        self.sim_start = data_config.start_date if data_config else "2021-01-01"
        self.sim_end = data_config.end_date if data_config else "2021-06-30"
        self.scale_factor = data_config.scale_factor if data_config else 1.0


    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path],
                        data_config: Union[str, Path] = None,
                        device: str = "auto") -> "ScenarioRunner":
        """Load a pre-calibrated model from checkpoint."""
        from .data_config import DataConfig

        ckpt = torch.load(str(checkpoint_path), weights_only=False)
        r2 = ckpt["R2"]
        cal_params = ckpt["params"]

        if data_config is None:
            data_config = "danish_abm/yamls/data/denmark_2021_jan_jun.yaml"
        dc = (data_config if isinstance(data_config, DataConfig)
              else DataConfig.from_yaml(data_config))

        base_yaml = os.path.join(dc.repo_root, "danish_abm", "yamls", "config.yaml")
        cfg_omega = OmegaConf.load(base_yaml)
        _pop_dir = str(dc.population_dir)
        cfg_omega.simulation_metadata.population_dir = _pop_dir
        cfg_omega.simulation_metadata.num_agents = dc.num_agents
        cfg_omega.simulation_metadata.disease_stage_file = str(dc.population_dir / "disease_stages.csv")
        cfg_omega.simulation_metadata.age_group_file = str(dc.population_dir / "age.pickle")
        cfg_omega.simulation_metadata.infection_network_file = str(dc.network_file)
        cfg_omega.simulation_metadata.mapping_path = str(dc.population_dir / "mapping.json")
        cfg_omega.simulation_metadata.num_steps_per_episode = dc.n_days
        cfg_omega.simulation_metadata.NUM_WEEKS = (dc.n_days + 6) // 7
        cfg_omega.simulation_metadata.INFINITY_TIME = dc.n_days + 50
        config = OmegaConf.to_container(cfg_omega, resolve=True)
        dc.patch_model_config(config)

        meta = config["simulation_metadata"]
        meta["device"] = device
        meta["calibration"] = True

        meta["multi_strain"] = True
        from danish_abm.strains import StrainConfig as _SC
        meta["strain_config"] = {
            "preset": _SC.preset_for_variant(meta.get("variant_name"))}
        meta["disease_params_distributional"] = True
        if "disease_params" not in meta:
            meta["disease_params"] = {
                "latent_period": {"mean": 3.0, "ci95": [1.0, 6.0]},
                "infectious_period": {"mean": 5.0, "ci95": [3.0, 8.0]},
                "generation_time": {"mean": 5.5, "ci95": [2.5, 9.5]},
                "immunity_waning": {"mean": 180.0, "ci95": [90.0, 365.0]},
            }
        else:
            _dp = meta["disease_params"]
            print("  [scenario] disease_params resolved to: "
                  + ", ".join(f"{k}={(v or {}).get('mean') if isinstance(v, dict) else v}"
                              for k, v in sorted(_dp.items()))
                  + "  (source: whatever DataConfig.patch_model_config supplied, else "
                    "danish_abm/yamls/config.yaml -- NOT <wave>/config.yaml)")
        if "seasonal" in cal_params:
            meta["seasonal"] = cal_params["seasonal"]
        else:
            _seas = meta.get("seasonal", None)
            print(f"  [scenario] checkpoint predates seasonal-forcing provenance; "
                  f"using the config's value ({_seas!r}), which is what the "
                  f"calibration ran with. Recalibrate to record it explicitly.")
            meta["seasonal"] = _seas
        if "hospitalization_delay" not in meta:
            meta["hospitalization_delay"] = 6
        if "REINFECTION_SUSCEPTIBILITY" not in meta:
            meta["REINFECTION_SUSCEPTIBILITY"] = 0.0
        meta["daily_importation_rate"] = cal_params.get("importation_rate", 0.00005)
        meta.setdefault("initial_infection_ratio", 0.01)

        if "PIS" in cal_params:
            meta["prior_immunity_scale"] = cal_params["PIS"]

        return cls(config, cal_params, r2, data_config=dc)


    def _apply_pre_wave_immunity(self, runner, config):
        """Infer and apply pre-wave immunity from event files."""
        import pandas as pd
        from datetime import datetime

        dc = self.data_config
        if dc is None:
            return

        sim_start = pd.Timestamp(dc.start_date)
        meta = config["simulation_metadata"]
        n_agents = meta["num_agents"]

        disease_stage = runner.state["agents"]["citizens"]["disease_stage"]
        device = disease_stage.device
        modified = False

        if not bool(meta.get("prewave_from_register_events", False)):
            from .substeps.utils import apply_prior_immunity
            n_set, _wm, _pi = apply_prior_immunity(
                runner,
                data_config=dc,
                meta=meta,
                prior_immune_fraction=dc.prior_immune_fraction,
                prior_immunity_scale=meta.get("prior_immunity_scale"),
                verbose=True,
            )
            print(f"  Pre-wave state from apply_prior_immunity (same constructor as "
                  f"the calibration): {n_set:,} agents immune, 0 pre-wave dead. Set "
                  f"simulation_metadata.prewave_from_register_events=true to use the "
                  f"register event files instead.")
            return

        registry_vax = (
            meta.get("vaccination_strategy") == "registry"
            and meta.get("vaccination_schedule_path")
        )
        vax_disabled = not meta.get("vaccination_enabled", False)
        skip_vax_immunity = registry_vax or vax_disabled
        if registry_vax:
            print("  Pre-wave vaccinations handled by registry schedule "
                  "(_pre_sim); skipping RECOVERED-marking branch")
        elif vax_disabled:
            print("  No-vaccination scenario: skipping pre-wave vaccine immunity "
                  "(RECOVERED-marking) so it represents zero vaccine protection")
        if (not skip_vax_immunity) and dc.individual_vaccination_csv and dc.individual_vaccination_csv.exists():
            try:
                from .agent_properties import read_tabular
                vax_df = read_tabular(dc.individual_vaccination_csv)

                col_map = {}
                for c in vax_df.columns:
                    cl = c.strip().lower().replace(" ", "_")
                    if cl in ("date", "test_date", "dato", "vaccination_date"):
                        col_map[c] = "date"
                    elif cl in ("agent_id", "id", "person_id"):
                        col_map[c] = "agent_id"
                vax_df = vax_df.rename(columns=col_map)
                vax_df["date"] = pd.to_datetime(vax_df["date"])

                pre_wave_vax = vax_df[vax_df["date"] < sim_start]
                if len(pre_wave_vax) > 0:
                    doses_per_agent = pre_wave_vax.groupby("agent_id").size()
                    n_vaccinated = len(doses_per_agent)

                    for aid, n_doses in doses_per_agent.items():
                        aid = int(aid)
                        if aid < n_agents:
                            disease_stage[aid, 0] = 3.0
                            modified = True

                    if n_vaccinated > 0:
                        print(f"  Pre-wave immunity: {n_vaccinated:,} agents vaccinated "
                              f"({len(pre_wave_vax):,} doses before {dc.start_date})")
            except Exception as e:
                print(f"  Warning: could not process pre-wave vaccinations: {e}")

        if dc.individual_test_csv and dc.individual_test_csv.exists():
            try:
                from .agent_properties import read_tabular
                test_df = read_tabular(dc.individual_test_csv)

                col_map = {}
                for c in test_df.columns:
                    cl = c.strip().lower().replace(" ", "_")
                    if cl in ("date", "test_date", "dato"):
                        col_map[c] = "date"
                    elif cl in ("agent_id", "id", "person_id"):
                        col_map[c] = "agent_id"
                    elif cl in ("result", "positive", "pos"):
                        col_map[c] = "result"
                test_df = test_df.rename(columns=col_map)
                test_df["date"] = pd.to_datetime(test_df["date"])

                _res = test_df["result"]
                if _res.dtype.kind in ("O", "U") or str(_res.dtype) == "string":
                    _is_pos = _res.astype(str).str.strip().str.lower().isin(
                        ["1", "positive", "pos", "true"])
                else:
                    _is_pos = pd.to_numeric(_res, errors="coerce") == 1
                pre_wave_pos = test_df[(test_df["date"] < sim_start) & _is_pos]
                print(f"  Pre-wave positive tests matched: {len(pre_wave_pos):,} "
                      f"of {len(test_df):,} records")
                if len(pre_wave_pos) > 0:
                    from .registry_data import _build_agent_id_map
                    _agents_file = dc.individual_test_csv.parent.parent / "agents.parquet"
                    _id_map = _build_agent_id_map(_agents_file)
                    prior_infected = pre_wave_pos["agent_id"].unique()
                    if _id_map is None:
                        print(f"  Warning: no agents.parquet at {_agents_file} — "
                              f"skipping pre-wave test immunity rather than "
                              f"assigning {len(prior_infected):,} agents by "
                              f"unmapped register id")
                        prior_infected = []
                    n_infected = 0
                    n_unmapped = 0
                    for aid in prior_infected:
                        row = _id_map.get(int(aid))
                        if row is None:
                            n_unmapped += 1
                            continue
                        if row < n_agents:
                            disease_stage[row, 0] = 3.0
                            n_infected += 1
                            modified = True
                    if n_unmapped:
                        print(f"  {n_unmapped:,} pre-wave positive ids not in "
                              f"the model population (dropped)")

                    if n_infected > 0:
                        print(f"  Pre-wave immunity: {n_infected:,} agents with prior "
                              f"positive test before {dc.start_date}")
            except Exception as e:
                print(f"  Warning: could not process pre-wave tests: {e}")

        if dc.individual_vaccination_csv:
            events_dir = dc.individual_vaccination_csv.parent
            from .agent_properties import find_file
            mort_file = find_file(events_dir, "mortality")
            if mort_file is not None:
                try:
                    from .agent_properties import read_tabular
                    mort_df = read_tabular(mort_file)

                    col_map = {}
                    for c in mort_df.columns:
                        cl = c.strip().lower().replace(" ", "_")
                        if cl in ("date", "dato", "death_date"):
                            col_map[c] = "date"
                        elif cl in ("agent_id", "id", "person_id"):
                            col_map[c] = "agent_id"
                    mort_df = mort_df.rename(columns=col_map)
                    mort_df["date"] = pd.to_datetime(mort_df["date"])

                    pre_wave_dead = mort_df[mort_df["date"] < sim_start]
                    if len(pre_wave_dead) > 0:
                        n_dead = 0
                        for aid in pre_wave_dead["agent_id"].unique():
                            aid = int(aid)
                            if aid < n_agents:
                                disease_stage[aid, 0] = 4.0
                                n_dead += 1
                                modified = True

                        if n_dead > 0:
                            print(f"  Pre-wave mortality: {n_dead:,} agents died "
                                  f"before {dc.start_date}")
                except Exception as e:
                    print(f"  Warning: could not process pre-wave mortality: {e}")

        if not modified:
            from .substeps.utils import apply_prior_immunity
            n_set, _wm, _pi = apply_prior_immunity(
                runner,
                data_config=dc,
                meta=config.get("simulation_metadata", {}),
                prior_immune_fraction=dc.prior_immune_fraction,
            )
            if n_set > 0:
                modified = True

        if modified:
            runner.state["agents"]["citizens"]["disease_stage"] = disease_stage


    def run(self, spec: ScenarioSpec) -> ScenarioResult:
        """Execute a scenario and return structured results."""
        from . import get_registry, get_runner

        errors = spec.validate()
        if errors:
            raise ValueError(f"Invalid scenario:\n" + "\n".join(f"  - {e}" for e in errors))

        config = copy.deepcopy(self.base_config)
        meta = config["simulation_metadata"]

        vax = spec.vaccination
        meta["vaccination_enabled"] = vax.get("enabled", True)
        meta["vaccination_daily_capacity"] = vax.get("capacity_per_day", 2500)
        apply_spec_overrides(meta, vax, "vaccination")
        if vax.get("start_date"):
            start_dt = datetime.strptime(self.sim_start, "%Y-%m-%d")
            vax_dt = datetime.strptime(vax["start_date"], "%Y-%m-%d")
            meta["vaccination_start_day"] = (vax_dt - start_dt).days
        else:
            meta["vaccination_start_day"] = 27
        if vax.get("strategy") is not None:
            meta["vaccination_strategy"] = vax["strategy"]
        if vax.get("schedule_path") is not None:
            meta["vaccination_schedule_path"] = vax["schedule_path"]
        _assert_register_vaccination(meta, self.data_config, vax)

        apply_spec_overrides(meta, spec.testing, "testing", log=print)

        if meta.get("isolation_reduction") is None and "isolation_reduction" in self.cal_params:
            meta["isolation_reduction"] = self.cal_params["isolation_reduction"]

        surv = spec.surveillance
        if surv.get("policy") is not None:
            meta["surveillance_policy"] = surv["policy"]
            meta["surveillance_total_daily_tests"] = surv.get("total_daily_tests", 5000)
            meta["surveillance_prevalence_threshold"] = surv.get("prevalence_threshold", 0.01)
            meta["surveillance_surrogate_update_frequency"] = surv.get("surrogate_update_frequency", 7)

        if spec.initial_infection_ratio is not None:
            meta["initial_infection_ratio"] = spec.initial_infection_ratio
        if spec.importation_rate is not None:
            meta["daily_importation_rate"] = spec.importation_rate

        if spec.pathogen_preset:
            meta["pathogen_preset"] = spec.pathogen_preset
            from .compartments import CompartmentModel
            cm = CompartmentModel.from_preset(spec.pathogen_preset)
            cm_overrides = cm.to_config_overrides()
            for k, v in cm_overrides.items():
                if isinstance(v, dict) and k in meta and isinstance(meta[k], dict):
                    meta[k].update(v)
                else:
                    meta[k] = v
        elif spec.compartment_config:
            meta["compartment_config"] = spec.compartment_config
            from .compartments import CompartmentModel
            cm = CompartmentModel.from_dict(spec.compartment_config)
            cm_overrides = cm.to_config_overrides()
            for k, v in cm_overrides.items():
                if isinstance(v, dict) and k in meta and isinstance(meta[k], dict):
                    meta[k].update(v)
                else:
                    meta[k] = v

        config["substeps"]["0"]["transition"]["new_transmission"]["arguments"]["R2"]["value"] = \
            self.r2.squeeze().tolist()

        if "M" in self.cal_params:
            config["substeps"]["1"]["transition"]["seirm_progression"]["arguments"]["M"]["value"] = \
                self.cal_params["M"]
        if "H" in self.cal_params:
            config["substeps"]["1"]["transition"]["seirm_progression"]["arguments"]["H"]["value"] = \
                self.cal_params["H"]

        tx_args = config["substeps"]["0"]["transition"]["new_transmission"]["arguments"]
        if "q_layers" in self.cal_params:
            tx_args["q_layers"]["value"] = self.cal_params["q_layers"]
        if "R_community" in self.cal_params:
            tx_args["R_community"]["value"] = self.cal_params["R_community"]
        if "spatial_mixing" in self.cal_params:
            import math
            sm = self.cal_params["spatial_mixing"]
            w_muni = max(sm["municipality"], 1e-6)
            w_reg = max(sm["region"], 1e-6)
            w_nat = max(sm["national"], 1e-6)
            tx_args["spatial_mixing_logits"]["value"] = [
                math.log(w_muni), math.log(w_reg), math.log(w_nat)
            ]
            tx_args["spatial_local_pop_slope"]["value"] = sm.get("pop_slope", 0.3)
        if "spillover_logits" in self.cal_params:
            tx_args["spillover_logits"]["value"] = self.cal_params["spillover_logits"]

        npi_schedule = spec.build_npi_schedule(self.sim_start, self.n_days,
                                                sim_end=self.sim_end)
        pop_dir = meta["population_dir"]
        interv_path = interv_schedule_path(pop_dir)
        had_existing = os.path.exists(interv_path)
        old_data = torch.load(interv_path, weights_only=False) if had_existing else None
        npi_tensor = npi_schedule.build_multiplier_tensor()
        _resilient_torch_save(_schedule_payload(npi_tensor, npi_schedule, old_data, spec),
                              interv_path)

        trigger_engine = spec.build_trigger_engine(
            n_days=self.n_days,
            num_agents=self.n_agents,
            healthcare_capacity=meta.get("healthcare_capacity"),
        )

        if trigger_engine is not None:
            meta["scale_factor"] = self.scale_factor

        reg = get_registry()
        runner = get_runner(config, reg)

        def _gpu_mem(label=""):
            if torch.cuda.is_available():
                a = torch.cuda.memory_allocated() / 1e9
                r = torch.cuda.memory_reserved() / 1e9
                print(f"  [GPU] {label}: {a:.2f}GB allocated, {r:.2f}GB reserved", flush=True)

        _gpu_mem("before init")
        runner.init()
        _gpu_mem("after init")

        from .substeps.utils import patch_runner_no_snapshots
        patch_runner_no_snapshots(runner)
        _gpu_mem("after patch (state_trajectory cleared)")

        if trigger_engine is not None:
            from . import runtime_set
            runtime_set(config, "trigger_engine", trigger_engine)

        self._apply_pre_wave_immunity(runner, config)
        _gpu_mem("after pre-wave immunity")

        t0 = time.time()
        with torch.no_grad():
            runner.step(self.n_days)
        elapsed = time.time() - t0

        result = self._extract_result(runner, spec, elapsed)

        if trigger_engine is not None:
            result.trigger_log = trigger_engine.activation_log
            result.trigger_active_per_day = []
            for name, mod in runner.named_modules():
                if type(mod).__name__ == "NewTransmission":
                    result.trigger_active_per_day = mod.tracking.get("active_triggers", [])
                    break

        if old_data is not None:
            _resilient_torch_save(old_data, interv_path)
        elif os.path.exists(interv_path):
            os.remove(interv_path)

        from . import runtime_clear
        runtime_clear(config)
        del runner, reg
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def run_ensemble(self, spec: ScenarioSpec, n_runs: int = 5,
                     ci_pct: float = 90.0, verbose: bool = True) -> dict:
        """Run a scenario multiple times for uncertainty quantification."""
        lo_pct = (100 - ci_pct) / 2
        hi_pct = 100 - lo_pct
        results = []
        for i in range(n_runs):
            if verbose:
                print(f"    Run {i+1}/{n_runs}...", end=" ", flush=True)
            r = self.run(spec)
            results.append(r)
            if verbose:
                print(f"({r.elapsed_seconds:.1f}s, {r.total_infections:,} inf)")

        n = self.n_days

        def _ts_stats(key):
            arrs = []
            for r in results:
                series = getattr(r, key)
                if series:
                    arrs.append(series[:n])
                else:
                    arrs.append([0] * n)
            arr = np.array(arrs, dtype=float)
            return {
                "median": np.median(arr, axis=0),
                "lo": np.percentile(arr, lo_pct, axis=0),
                "hi": np.percentile(arr, hi_pct, axis=0),
            }

        def _scalar_stats(key):
            vals = np.array([getattr(r, key) for r in results], dtype=float)
            return {
                "median": float(np.median(vals)),
                "lo": float(np.percentile(vals, lo_pct)),
                "hi": float(np.percentile(vals, hi_pct)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

        return {
            "results": results,
            "n_runs": n_runs,
            "ci_pct": ci_pct,
            "infections_ts": _ts_stats("daily_infections"),
            "deaths_ts": _ts_stats("daily_deaths"),
            "hosp_ts": _ts_stats("daily_hosp"),
            "total_infections": _scalar_stats("total_infections"),
            "total_deaths": _scalar_stats("total_deaths"),
            "total_hosp": _scalar_stats("total_hospitalizations"),
            "peak_infections": _scalar_stats("peak_daily_infections"),
        }

    def run_with_uncertainty(
        self,
        spec: ScenarioSpec,
        param_sample: ParameterSample,
    ) -> ScenarioResult:
        """Run a scenario with a specific parameter sample applied."""
        from . import get_registry, get_runner

        errors = spec.validate()
        if errors:
            raise ValueError(f"Invalid scenario:\n" + "\n".join(f"  - {e}" for e in errors))

        config = copy.deepcopy(self.base_config)
        config = param_sample.apply_to_config(config)
        meta = config["simulation_metadata"]

        vax = spec.vaccination
        meta["vaccination_enabled"] = vax.get("enabled", True)
        meta["vaccination_daily_capacity"] = vax.get("capacity_per_day", 2500)
        apply_spec_overrides(meta, vax, "vaccination")
        if vax.get("start_date"):
            start_dt = datetime.strptime(self.sim_start, "%Y-%m-%d")
            vax_dt = datetime.strptime(vax["start_date"], "%Y-%m-%d")
            meta["vaccination_start_day"] = (vax_dt - start_dt).days
        else:
            meta["vaccination_start_day"] = 27
        if vax.get("strategy") is not None:
            meta["vaccination_strategy"] = vax["strategy"]
        if vax.get("schedule_path") is not None:
            meta["vaccination_schedule_path"] = vax["schedule_path"]
        _assert_register_vaccination(meta, self.data_config, vax)

        apply_spec_overrides(meta, spec.testing, "testing", log=print)

        if meta.get("isolation_reduction") is None and "isolation_reduction" in self.cal_params:
            meta["isolation_reduction"] = self.cal_params["isolation_reduction"]

        surv = spec.surveillance
        if surv.get("policy") is not None:
            meta["surveillance_policy"] = surv["policy"]
            meta["surveillance_total_daily_tests"] = surv.get("total_daily_tests", 5000)
            meta["surveillance_prevalence_threshold"] = surv.get("prevalence_threshold", 0.01)
            meta["surveillance_surrogate_update_frequency"] = surv.get("surrogate_update_frequency", 7)

        if spec.initial_infection_ratio is not None:
            meta["initial_infection_ratio"] = spec.initial_infection_ratio
        if spec.importation_rate is not None:
            meta["daily_importation_rate"] = spec.importation_rate

        r2 = self.r2.clone()
        r_scale = param_sample.values.get("R_scale", 1.0)
        r2 = r2 * r_scale

        config["substeps"]["0"]["transition"]["new_transmission"]["arguments"]["R2"]["value"] = \
            r2.squeeze().tolist()

        if "M" in self.cal_params and "M" not in param_sample.values:
            config["substeps"]["1"]["transition"]["seirm_progression"]["arguments"]["M"]["value"] = \
                self.cal_params["M"]
        if "H" in self.cal_params and "H" not in param_sample.values:
            config["substeps"]["1"]["transition"]["seirm_progression"]["arguments"]["H"]["value"] = \
                self.cal_params["H"]

        _m_scale = param_sample.values.get("M_scale", 1.0)
        _h_scale = param_sample.values.get("H_scale", 1.0)
        _prog_args = config["substeps"]["1"]["transition"]["seirm_progression"]["arguments"]
        if abs(_m_scale - 1.0) > 1e-9 and "M" in self.cal_params:
            _prog_args["M"]["value"] = self.cal_params["M"] * _m_scale
        if abs(_h_scale - 1.0) > 1e-9 and "H" in self.cal_params:
            _prog_args["H"]["value"] = self.cal_params["H"] * _h_scale

        tx_args = config["substeps"]["0"]["transition"]["new_transmission"]["arguments"]
        if "q_layers" in self.cal_params:
            tx_args["q_layers"]["value"] = _reweight_workplace_q(
                self.cal_params["q_layers"],
                param_sample.values.get("workplace_share"))
        if "R_community" in self.cal_params:
            tx_args["R_community"]["value"] = self.cal_params["R_community"]
        if "spatial_mixing" in self.cal_params:
            import math
            sm = self.cal_params["spatial_mixing"]
            w_muni = max(sm["municipality"], 1e-6)
            w_reg = max(sm["region"], 1e-6)
            w_nat = max(sm["national"], 1e-6)
            tx_args["spatial_mixing_logits"]["value"] = [
                math.log(w_muni), math.log(w_reg), math.log(w_nat)
            ]
            tx_args["spatial_local_pop_slope"]["value"] = sm.get("pop_slope", 0.3)
        if "spillover_logits" in self.cal_params:
            tx_args["spillover_logits"]["value"] = _perturb_spillover_logits(
                self.cal_params["spillover_logits"],
                param_sample.values.get("spillover_z"))

        npi_scale = param_sample.values.get("npi_effectiveness_scale", 1.0)

        npi_schedule = spec.build_npi_schedule(self.sim_start, self.n_days,
                                                sim_end=self.sim_end)
        pop_dir = meta["population_dir"]
        interv_path = interv_schedule_path(pop_dir)
        had_existing = os.path.exists(interv_path)
        old_data = torch.load(interv_path, weights_only=False) if had_existing else None

        npi_tensor = npi_schedule.build_multiplier_tensor()
        if abs(npi_scale - 1.0) > 1e-6:
            npi_tensor = (1.0 - npi_scale * (1.0 - npi_tensor)).clamp(min=0.0)
        _resilient_torch_save(_schedule_payload(npi_tensor, npi_schedule, old_data, spec),
                              interv_path)

        trigger_engine = spec.build_trigger_engine(
            n_days=self.n_days,
            num_agents=self.n_agents,
            healthcare_capacity=meta.get("healthcare_capacity"),
        )
        if trigger_engine is not None:
            meta["scale_factor"] = self.scale_factor

        reg = get_registry()
        runner = get_runner(config, reg)

        def _gpu_mem(label=""):
            if torch.cuda.is_available():
                a = torch.cuda.memory_allocated() / 1e9
                r = torch.cuda.memory_reserved() / 1e9
                print(f"  [GPU] {label}: {a:.2f}GB allocated, {r:.2f}GB reserved", flush=True)

        _gpu_mem("before init")
        runner.init()
        _gpu_mem("after init")

        from .substeps.utils import patch_runner_no_snapshots
        patch_runner_no_snapshots(runner)
        _gpu_mem("after patch")

        if trigger_engine is not None:
            from . import runtime_set
            runtime_set(config, "trigger_engine", trigger_engine)

        self._apply_pre_wave_immunity(runner, config)
        _gpu_mem("after pre-wave immunity")

        t0 = time.time()
        with torch.no_grad():
            runner.step(self.n_days)
        elapsed = time.time() - t0

        result = self._extract_result(runner, spec, elapsed)
        result.parameter_sample = param_sample

        if trigger_engine is not None:
            result.trigger_log = trigger_engine.activation_log
            result.trigger_active_per_day = []
            for name, mod in runner.named_modules():
                if type(mod).__name__ == "NewTransmission":
                    result.trigger_active_per_day = mod.tracking.get("active_triggers", [])
                    break

        if old_data is not None:
            _resilient_torch_save(old_data, interv_path)
        elif os.path.exists(interv_path):
            os.remove(interv_path)

        from . import runtime_clear
        runtime_clear(config)
        del runner, reg
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def run_ensemble_uncertain(
        self,
        spec: ScenarioSpec,
        uncertainty: UncertaintyConfig = None,
        n_runs: int = 20,
        ci_pct: float = 90.0,
        capacity: HealthcareCapacity = None,
        baseline_result: "ScenarioResult" = None,
        seed: int = None,
        verbose: bool = True,
        per_run_callback=None,
        run_indices=None,
    ) -> dict:
        """Run an ensemble with full uncertainty propagation."""
        if uncertainty is None:
            uncertainty = UncertaintyConfig.stochastic_only()

        samples = uncertainty.get_ensemble_plan(n_runs, seed=seed)
        plan = list(enumerate(samples))
        if run_indices is not None:
            _keep = set(int(i) for i in run_indices)
            plan = [(i, smp) for i, smp in plan if i in _keep]
            if not plan:
                raise ValueError(f"run_indices {sorted(_keep)[:5]}... select no draw of {len(samples)}")
        samples = [smp for _, smp in plan]
        actual_runs = len(samples)

        lo_pct = (100 - ci_pct) / 2
        hi_pct = 100 - lo_pct
        results = []
        all_metrics = []

        for i, sample in plan:
            if verbose:
                label = sample.structural_label or ""
                if label:
                    label = f" [{label}]"
                n_overrides = len(sample.values)
                print(f"    Run {i+1}/{actual_runs}{label} ({n_overrides} param overrides)...",
                      end=" ", flush=True)

            if seed is not None:
                torch.manual_seed(seed + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + i)

            if uncertainty.mode == UncertaintyMode.STOCHASTIC:
                r = self.run(spec)
            else:
                r = self.run_with_uncertainty(spec, sample)

            pm = r.compute_policy_metrics(
                capacity=capacity,
                baseline=baseline_result,
            )
            all_metrics.append(pm)

            if per_run_callback is not None:
                per_run_callback(i, r)
                r.outcome_analysis = None
            results.append(r)

            if verbose:
                print(f"({r.elapsed_seconds:.1f}s, {r.total_infections:,} inf, "
                      f"peak_hosp={pm.peak_hospital_occupancy:.0f})")

        n = self.n_days

        def _ts_stats(key):
            arrs = []
            for r in results:
                series = getattr(r, key)
                if series:
                    arrs.append(series[:n])
                else:
                    arrs.append([0] * n)
            arr = np.array(arrs, dtype=float)
            return {
                "median": np.median(arr, axis=0),
                "lo": np.percentile(arr, lo_pct, axis=0),
                "hi": np.percentile(arr, hi_pct, axis=0),
            }

        def _scalar_stats(key):
            vals = np.array([getattr(r, key) for r in results], dtype=float)
            return {
                "median": float(np.median(vals)),
                "lo": float(np.percentile(vals, lo_pct)),
                "hi": float(np.percentile(vals, hi_pct)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

        hosp_occ_arrs = [m.hospital_occupancy for m in all_metrics if m.hospital_occupancy is not None]
        icu_occ_arrs = [m.icu_occupancy for m in all_metrics if m.icu_occupancy is not None]

        def _occ_stats(arrs):
            if not arrs:
                return None
            min_len = min(len(a) for a in arrs)
            stacked = np.array([a[:min_len] for a in arrs])
            return {
                "median": np.median(stacked, axis=0),
                "lo": np.percentile(stacked, lo_pct, axis=0),
                "hi": np.percentile(stacked, hi_pct, axis=0),
            }

        from .uncertainty import decompose_variance, parameter_sensitivity
        result_dicts = [
            {"total_infections": r.total_infections,
             "total_deaths": r.total_deaths,
             "total_hospitalizations": r.total_hospitalizations,
             "peak_daily_infections": r.peak_daily_infections,
             "sample_idx": i}
            for i, r in enumerate(results)
        ]
        var_decomp = decompose_variance(result_dicts, samples, "total_infections")
        sensitivity = parameter_sensitivity(result_dicts, samples, "total_infections")

        ensemble_policy = EnsemblePolicyMetrics.from_metrics_list(all_metrics, ci_pct=ci_pct)

        return {
            "results": results,
            "samples": samples,
            "n_runs": actual_runs,
            "ci_pct": ci_pct,
            "uncertainty_mode": uncertainty.mode.name,
            "infections_ts": _ts_stats("daily_infections"),
            "deaths_ts": _ts_stats("daily_deaths"),
            "hosp_ts": _ts_stats("daily_hosp"),
            "hospital_occupancy_ts": _occ_stats(hosp_occ_arrs),
            "icu_occupancy_ts": _occ_stats(icu_occ_arrs),
            "total_infections": _scalar_stats("total_infections"),
            "total_deaths": _scalar_stats("total_deaths"),
            "total_hosp": _scalar_stats("total_hospitalizations"),
            "peak_infections": _scalar_stats("peak_daily_infections"),
            "policy_metrics": all_metrics,
            "ensemble_policy": ensemble_policy,
            "variance_decomposition": var_decomp,
            "sensitivity": sensitivity,
        }

    def run_comparison(self, baseline_spec: ScenarioSpec,
                       *scenario_specs: ScenarioSpec) -> List[ScenarioResult]:
        """Run a baseline + multiple scenarios, auto-computing deltas."""
        baseline = self.run(baseline_spec)
        results = [baseline]
        for spec in scenario_specs:
            result = self.run(spec)
            result.compare(baseline)
            results.append(result)
        return results


    def _extract_result(self, runner, spec: ScenarioSpec, elapsed: float) -> ScenarioResult:
        """Build a ScenarioResult from runner state after simulation."""
        start_dt = datetime.strptime(self.sim_start, "%Y-%m-%d")
        n = self.n_days

        result = ScenarioResult(
            scenario_name=spec.name,
            sim_start=self.sim_start,
            sim_end=self.sim_end,
            n_days=n,
            num_agents=self.n_agents,
            scale_factor=self.scale_factor,
            elapsed_seconds=elapsed,
        )

        daily_inf = runner.state["environment"]["daily_infected"].detach().cpu().numpy().squeeze()[:n]
        daily_deaths = runner.state["environment"]["daily_deaths"].detach().cpu().numpy().squeeze()[:n]
        daily_hosp = runner.state["environment"]["daily_hosp"].detach().cpu().numpy().squeeze()[:n]

        result.daily_infections = daily_inf.tolist()
        result.daily_deaths = daily_deaths.tolist()
        result.daily_hosp = daily_hosp.tolist()

        for name, mod in runner.named_modules():
            cls = type(mod).__name__
            if cls == "NewTransmission":
                tracking = mod.tracking
                infections = tracking.get("new_infections", [])[:n]
                result.total_infections = int(sum(infections))

                if infections:
                    peak_idx = int(np.argmax(infections))
                    result.peak_daily_infections = float(infections[peak_idx])
                    result.peak_infection_day = peak_idx
                    result.peak_infection_date = (start_dt + timedelta(days=peak_idx)).strftime("%Y-%m-%d")

                if tracking.get("S") and tracking.get("R") and tracking.get("M"):
                    s_final = tracking["S"][-1] if tracking["S"] else 0
                    r_final = tracking["R"][-1] if tracking["R"] else 0
                    m_final = tracking["M"][-1] if tracking["M"] else 0
                    m_baseline = tracking["M"][0] if tracking["M"] else 0
                    m_inwave = max(m_final - m_baseline, 0)
                    result.final_susceptible_pct = s_final / self.n_agents
                    result.final_recovered_pct = r_final / self.n_agents
                    result.final_dead_pct = m_inwave / self.n_agents

                if mod._strain_manager and hasattr(mod._strain_manager, "daily_by_strain"):
                    result.strain_totals = {}
                    for strain_name, counts in mod._strain_manager.daily_by_strain.items():
                        result.strain_totals[strain_name] = int(sum(counts[:n]))

                age_inf = tracking.get("new_infections_by_age", [])
                if age_inf:
                    AGE_LABELS = ["0-9", "10-19", "20-29", "30-39",
                                  "40-49", "50-59", "60-69", "70+"]
                    result.infections_by_age = {}
                    for day_counts in age_inf[:n]:
                        for i, count in enumerate(day_counts):
                            label = AGE_LABELS[i] if i < len(AGE_LABELS) else str(i)
                            result.infections_by_age[label] = (
                                result.infections_by_age.get(label, 0) + int(count)
                            )

                muni_inf = tracking.get("municipality_infections")
                if muni_inf and hasattr(mod, "_municipality_names"):
                    result.municipality_totals = {}
                    result.municipality_daily_infections = [
                        list(day_counts) for day_counts in muni_inf[:n]
                    ]
                    for day_counts in muni_inf[:n]:
                        for i, count in enumerate(day_counts):
                            mname = str(mod._municipality_names[i]) if i < len(mod._municipality_names) else str(i)
                            result.municipality_totals[mname] = result.municipality_totals.get(mname, 0) + count

            elif cls == "SEIRMProgression":
                dp_tracking = getattr(mod, "tracking", {})
                AGE_LABELS = ["0-9", "10-19", "20-29", "30-39",
                              "40-49", "50-59", "60-69", "70+"]
                deaths_by_age_list = dp_tracking.get("deaths_by_age", [])
                if deaths_by_age_list:
                    result.deaths_by_age = {}
                    for day_counts in deaths_by_age_list[:n]:
                        for i, count in enumerate(day_counts):
                            label = AGE_LABELS[i] if i < len(AGE_LABELS) else str(i)
                            result.deaths_by_age[label] = (
                                result.deaths_by_age.get(label, 0) + int(round(count))
                            )
                hosp_by_age_list = dp_tracking.get("hosp_by_age", [])
                if hosp_by_age_list:
                    result.hosp_by_age = {}
                    for day_counts in hosp_by_age_list[:n]:
                        for i, count in enumerate(day_counts):
                            label = AGE_LABELS[i] if i < len(AGE_LABELS) else str(i)
                            result.hosp_by_age[label] = (
                                result.hosp_by_age.get(label, 0) + int(round(count))
                            )

            elif cls == "VaccinationRollout":
                vax_daily = getattr(mod, "_daily_vaccinated", [])
                result.daily_vaccinated = list(vax_daily)
                result.total_vaccinated = int(sum(vax_daily))

            elif cls == "TestingAndIsolation":
                test_tracking = mod.tracking
                result.total_tests = int(sum(test_tracking.get("daily_tests", [])))
                result.total_detected = int(sum(test_tracking.get("daily_positives", [])))
                result.daily_detected = list(test_tracking.get("daily_positives", []))
                result.daily_tests = list(test_tracking.get("daily_tests", []))

        result.total_deaths = int(daily_deaths.sum())
        result.total_hospitalizations = int(daily_hosp.sum())
        result.peak_daily_deaths = float(daily_deaths.max())
        result.peak_daily_hosp = float(daily_hosp.max())

        for _nm, _mod in runner.named_modules():
            if hasattr(_mod, "layer_pressure_shares"):
                try:
                    result.layer_pressure = _mod.layer_pressure_shares()
                except Exception:
                    result.layer_pressure = {}
                break

        try:
            from .agent_properties import AgentPropertyRegistry, AgentEventLog, OutcomeAnalysis
            pop_dir = self.base_config["simulation_metadata"]["population_dir"]
            extra = self.base_config["simulation_metadata"].get("extra_agent_properties", None)
            prop_registry = AgentPropertyRegistry.from_population_dir(
                pop_dir, n_agents=self.n_agents, extra_pickles=extra,
            )
            event_log = AgentEventLog.from_runner(runner, n)
            result.outcome_analysis = OutcomeAnalysis(
                prop_registry, event_log,
                scale_factor=self.scale_factor,
                scenario_name=spec.name,
            )
        except Exception as e:
            import traceback
            print(f"  [WARN] outcome_analysis build failed ({type(e).__name__}: {e}); "
                  f"stratified exports will be empty for this run.", flush=True)
            if getattr(self, "_oa_warned", False) is False:
                traceback.print_exc()
                self._oa_warned = True

        return result


def main():
    """CLI entry point: python -m danish_abm.scenario run <scenario.yaml>"""
    import argparse

    parser = argparse.ArgumentParser(description="Run ABM scenarios")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run a scenario")
    run_p.add_argument("scenario", help="Path to scenario YAML")
    run_p.add_argument("--checkpoint", required=True, help="Path to calibrated_params.pt")
    run_p.add_argument("--data", default=None, help="Path to data config YAML")
    run_p.add_argument("--output", default=None, help="Output JSON path")
    run_p.add_argument("--device", default="auto")

    cmp_p = sub.add_parser("compare", help="Run baseline + scenarios and compare")
    cmp_p.add_argument("scenarios", nargs="+", help="Scenario YAML files (first = baseline)")
    cmp_p.add_argument("--checkpoint", required=True)
    cmp_p.add_argument("--data", default=None)
    cmp_p.add_argument("--output-dir", default=".")
    cmp_p.add_argument("--device", default="auto")

    val_p = sub.add_parser("validate", help="Validate a scenario YAML")
    val_p.add_argument("scenario", help="Path to scenario YAML")

    args = parser.parse_args()

    if args.command == "validate":
        spec = ScenarioSpec.from_yaml(args.scenario)
        errors = spec.validate()
        if errors:
            print("ERRORS:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("Valid.")
            print(spec.describe())

    elif args.command == "run":
        runner = ScenarioRunner.from_checkpoint(args.checkpoint, args.data, args.device)
        spec = ScenarioSpec.from_yaml(args.scenario)
        print(spec.describe())
        print()
        result = runner.run(spec)
        print(result.summary())
        if args.output:
            result.save(args.output)
            print(f"\nSaved: {args.output}")

    elif args.command == "compare":
        runner = ScenarioRunner.from_checkpoint(args.checkpoint, args.data, args.device)
        specs = [ScenarioSpec.from_yaml(p) for p in args.scenarios]
        results = runner.run_comparison(specs[0], *specs[1:])
        for r in results:
            print(r.summary())
            print()
        os.makedirs(args.output_dir, exist_ok=True)
        for r in results:
            safe_name = r.scenario_name.lower().replace(" ", "_")[:40]
            r.save(os.path.join(args.output_dir, f"{safe_name}.json"))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
