"""Trigger-based reactive policy system for the Danish ABM."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from .npi import NPIDefinition, BUILTIN_NPIS, get_npi, NUM_SCHEDULE_COLS


class TriggerMetric(str, Enum):
    """Observable metrics that can trigger policy activation."""

    DAILY_INFECTIONS = "daily_infections"
    DAILY_DEATHS = "daily_deaths"
    DAILY_HOSPITALIZATIONS = "daily_hospitalizations"
    SEVEN_DAY_INCIDENCE = "7day_incidence"
    FOURTEEN_DAY_INCIDENCE = "14day_incidence"
    R_EFF = "r_eff"
    POSITIVITY_RATE = "positivity_rate"
    HOSPITAL_OCCUPANCY = "hospital_occupancy"
    HOSPITAL_OCCUPANCY_FRACTION = "hospital_occupancy_fraction"
    ICU_OCCUPANCY = "icu_occupancy"
    ICU_OCCUPANCY_FRACTION = "icu_occupancy_fraction"
    CUMULATIVE_INFECTIONS = "cumulative_infections"
    SUSCEPTIBLE_FRACTION = "susceptible_fraction"

    MUNICIPALITY_7DAY_INCIDENCE = "municipality_7day_incidence"
    MUNICIPALITY_POSITIVITY_RATE = "municipality_positivity_rate"
    HOTSPOT_COUNT = "hotspot_count"


class TriggerOperator(str, Enum):
    """Comparison operator for trigger conditions."""
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="

    def evaluate(self, value: float, threshold: float) -> bool:
        if self == TriggerOperator.GT:
            return value > threshold
        elif self == TriggerOperator.GE:
            return value >= threshold
        elif self == TriggerOperator.LT:
            return value < threshold
        elif self == TriggerOperator.LE:
            return value <= threshold
        return False


@dataclass
class TriggerCondition:
    """A condition that evaluates to True/False at each timestep."""
    metric: TriggerMetric
    operator: TriggerOperator
    threshold: float
    sustained_days: int = 1

    def to_dict(self) -> dict:
        return {
            "metric": self.metric.value,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "sustained_days": self.sustained_days,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TriggerCondition":
        return cls(
            metric=TriggerMetric(d["metric"]),
            operator=TriggerOperator(d["operator"]) if "operator" in d else TriggerOperator.GT,
            threshold=float(d["threshold"]),
            sustained_days=int(d.get("sustained_days", 1)),
        )


@dataclass
class TriggerPolicy:
    """A reactive policy that activates an NPI when a condition is met."""
    name: str
    condition: TriggerCondition
    action: str
    strength: float = 1.0
    release_condition: Optional[TriggerCondition] = None
    cooldown_days: int = 0
    min_active_days: int = 1
    max_active_days: Optional[int] = None
    ramp_up_days: int = 0
    ramp_down_days: int = 0
    earliest_day: int = 0
    latest_day: Optional[int] = None
    custom_npi: Optional[NPIDefinition] = None

    def get_npi(self) -> NPIDefinition:
        """Resolve the NPI definition for this policy."""
        if self.custom_npi is not None:
            return self.custom_npi
        return get_npi(self.action)

    def validate(self) -> List[str]:
        errors = []
        if self.custom_npi is None and self.action not in BUILTIN_NPIS:
            errors.append(
                f"Trigger '{self.name}': unknown NPI '{self.action}'. "
                f"Available: {', '.join(sorted(BUILTIN_NPIS.keys()))}"
            )
        if not 0 <= self.strength <= 1:
            errors.append(f"Trigger '{self.name}': strength must be 0-1, got {self.strength}")
        if self.cooldown_days < 0:
            errors.append(f"Trigger '{self.name}': cooldown_days must be >= 0")
        if self.min_active_days < 1:
            errors.append(f"Trigger '{self.name}': min_active_days must be >= 1")
        if self.max_active_days is not None and self.max_active_days < self.min_active_days:
            errors.append(f"Trigger '{self.name}': max_active_days < min_active_days")
        return errors

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "condition": self.condition.to_dict(),
            "action": self.action,
            "strength": self.strength,
            "cooldown_days": self.cooldown_days,
            "min_active_days": self.min_active_days,
        }
        if self.release_condition:
            d["release_condition"] = self.release_condition.to_dict()
        if self.max_active_days is not None:
            d["max_active_days"] = self.max_active_days
        if self.ramp_up_days > 0:
            d["ramp_up_days"] = self.ramp_up_days
        if self.ramp_down_days > 0:
            d["ramp_down_days"] = self.ramp_down_days
        if self.earliest_day > 0:
            d["earliest_day"] = self.earliest_day
        if self.latest_day is not None:
            d["latest_day"] = self.latest_day
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TriggerPolicy":
        release = None
        if "release_condition" in d and d["release_condition"]:
            release = TriggerCondition.from_dict(d["release_condition"])

        return cls(
            name=d["name"],
            condition=TriggerCondition.from_dict(d["condition"]),
            action=d.get("action", d.get("npi", "")),
            strength=float(d.get("strength", 1.0)),
            release_condition=release,
            cooldown_days=int(d.get("cooldown_days", 0)),
            min_active_days=int(d.get("min_active_days", 1)),
            max_active_days=d.get("max_active_days"),
            ramp_up_days=int(d.get("ramp_up_days", 0)),
            ramp_down_days=int(d.get("ramp_down_days", 0)),
            earliest_day=int(d.get("earliest_day", 0)),
            latest_day=d.get("latest_day"),
        )


@dataclass
class _PolicyState:
    """Internal tracking state for a single trigger policy."""
    active: bool = False
    active_since: int = -1
    deactivated_at: int = -1
    condition_streak: int = 0
    release_streak: int = 0
    ramping_down: bool = False
    ramp_down_start: int = -1


class TriggerEngine:
    """Evaluates trigger policies each simulation step."""

    def __init__(
        self,
        policies: List[TriggerPolicy],
        n_days: int,
        num_agents: int = 0,
        healthcare_capacity: Optional[dict] = None,
    ):
        self.policies = policies
        self.n_days = n_days
        self.num_agents = num_agents
        self.healthcare_capacity = healthcare_capacity or {
            "hospital_beds": 15000,
            "icu_beds": 400,
        }
        self._states = [_PolicyState() for _ in policies]

        self._npi_vectors = []
        for p in policies:
            npi = p.get_npi()
            base_vec = npi.get_multiplier_vector()
            vec = 1.0 - p.strength * (1.0 - base_vec)
            self._npi_vectors.append(vec)

        self.activation_log: List[dict] = []

    def step(self, t: int, metrics: Dict[str, float]) -> torch.Tensor:
        """Evaluate all trigger policies and return combined multiplier."""
        combined = torch.ones(NUM_SCHEDULE_COLS)

        for i, (policy, state) in enumerate(zip(self.policies, self._states)):
            self._evaluate_policy(t, metrics, i, policy, state)

            if state.active or state.ramping_down:
                effective = self._effective_strength(t, policy, state)
                if effective > 0:
                    vec = 1.0 - effective * (1.0 - self._npi_vectors[i])
                    combined = combined * vec

        return combined

    def _evaluate_policy(
        self, t: int, metrics: Dict[str, float],
        idx: int, policy: TriggerPolicy, state: _PolicyState,
    ):
        """Update the state of a single policy based on current metrics."""
        if t < policy.earliest_day:
            return
        if policy.latest_day is not None and t > policy.latest_day:
            if state.active:
                self._deactivate(t, idx, policy, state, reason="past latest_day")
            return

        metric_val = metrics.get(policy.condition.metric.value)

        if state.active:
            days_active = t - state.active_since

            if policy.max_active_days is not None and days_active >= policy.max_active_days:
                self._deactivate(t, idx, policy, state, reason="max_active_days")
                return

            if policy.release_condition is not None and days_active >= policy.min_active_days:
                release_val = metrics.get(policy.release_condition.metric.value)
                if release_val is not None:
                    if policy.release_condition.operator.evaluate(
                        release_val, policy.release_condition.threshold
                    ):
                        state.release_streak += 1
                    else:
                        state.release_streak = 0

                    if state.release_streak >= policy.release_condition.sustained_days:
                        self._deactivate(t, idx, policy, state, reason="release_condition")
                        return

        elif not state.ramping_down:
            if state.deactivated_at >= 0:
                if t - state.deactivated_at < policy.cooldown_days:
                    return

            if metric_val is not None:
                if policy.condition.operator.evaluate(
                    metric_val, policy.condition.threshold
                ):
                    state.condition_streak += 1
                else:
                    state.condition_streak = 0

                if state.condition_streak >= policy.condition.sustained_days:
                    self._activate(t, idx, policy, state)

        if state.ramping_down:
            days_ramping = t - state.ramp_down_start
            if days_ramping >= policy.ramp_down_days:
                state.ramping_down = False

    def _activate(self, t: int, idx: int, policy: TriggerPolicy, state: _PolicyState):
        state.active = True
        state.active_since = t
        state.condition_streak = 0
        state.release_streak = 0
        state.ramping_down = False
        self.activation_log.append({
            "day": t, "policy": policy.name, "event": "activated",
        })

    def _deactivate(self, t: int, idx: int, policy: TriggerPolicy, state: _PolicyState,
                     reason: str = ""):
        state.active = False
        state.deactivated_at = t
        state.release_streak = 0
        state.condition_streak = 0

        if policy.ramp_down_days > 0:
            state.ramping_down = True
            state.ramp_down_start = t

        self.activation_log.append({
            "day": t, "policy": policy.name, "event": "deactivated",
            "reason": reason,
        })

    def _effective_strength(self, t: int, policy: TriggerPolicy, state: _PolicyState) -> float:
        """Compute effective strength accounting for ramp-up and ramp-down."""
        if state.ramping_down:
            if policy.ramp_down_days <= 0:
                return 0.0
            days_ramping = t - state.ramp_down_start
            frac = 1.0 - min(days_ramping / policy.ramp_down_days, 1.0)
            return frac

        if state.active:
            if policy.ramp_up_days > 0:
                days_active = t - state.active_since
                frac = min(days_active / policy.ramp_up_days, 1.0)
                return frac
            return 1.0

        return 0.0

    def get_active_policies(self, t: int = None) -> List[str]:
        """Return names of currently active policies."""
        return [
            p.name for p, s in zip(self.policies, self._states)
            if s.active or s.ramping_down
        ]

    def describe(self) -> str:
        """Human-readable summary of trigger policies."""
        lines = [f"TriggerEngine: {len(self.policies)} policies"]
        for p in self.policies:
            cond = p.condition
            lines.append(
                f"  {p.name}: IF {cond.metric.value} {cond.operator.value} {cond.threshold}"
                f" for {cond.sustained_days}d → {p.action}"
                f" (strength={p.strength})"
            )
            if p.release_condition:
                rc = p.release_condition
                lines.append(
                    f"    RELEASE: {rc.metric.value} {rc.operator.value} {rc.threshold}"
                    f" for {rc.sustained_days}d"
                )
            extras = []
            if p.cooldown_days > 0:
                extras.append(f"cooldown={p.cooldown_days}d")
            if p.max_active_days:
                extras.append(f"max={p.max_active_days}d")
            if p.ramp_up_days > 0:
                extras.append(f"ramp_up={p.ramp_up_days}d")
            if p.ramp_down_days > 0:
                extras.append(f"ramp_down={p.ramp_down_days}d")
            if extras:
                lines.append(f"    ({', '.join(extras)})")
        return "\n".join(lines)


def compute_trigger_metrics(
    state: dict,
    tracking: dict,
    t: int,
    num_agents: int,
    healthcare_capacity: Optional[dict] = None,
    scale_factor: float = 1.0,
) -> Dict[str, float]:
    """Extract current metric values from simulation state."""
    hc = healthcare_capacity or {"hospital_beds": 15000, "icu_beds": 400}
    metrics = {}

    new_inf = tracking.get("new_infections", [])
    if new_inf and t < len(new_inf):
        metrics["daily_infections"] = float(new_inf[t])
    elif new_inf:
        metrics["daily_infections"] = float(new_inf[-1])
    else:
        metrics["daily_infections"] = 0.0

    real_pop = num_agents / max(scale_factor, 1e-9)
    if len(new_inf) >= 7:
        week_sum = sum(new_inf[-7:]) / max(scale_factor, 1e-9)
        metrics["7day_incidence"] = week_sum / max(real_pop, 1) * 100_000
    else:
        total = sum(new_inf) / max(scale_factor, 1e-9) if new_inf else 0
        metrics["7day_incidence"] = total / max(real_pop, 1) * 100_000

    if len(new_inf) >= 14:
        two_week_sum = sum(new_inf[-14:]) / max(scale_factor, 1e-9)
        metrics["14day_incidence"] = two_week_sum / max(real_pop, 1) * 100_000
    else:
        total = sum(new_inf) / max(scale_factor, 1e-9) if new_inf else 0
        metrics["14day_incidence"] = total / max(real_pop, 1) * 100_000

    r_eff = tracking.get("R_eff", [])
    if r_eff and t < len(r_eff):
        window = r_eff[max(0, len(r_eff)-7):]
        metrics["r_eff"] = sum(window) / max(len(window), 1)
    elif r_eff:
        metrics["r_eff"] = float(r_eff[-1])
    else:
        metrics["r_eff"] = 0.0

    metrics["cumulative_infections"] = float(sum(new_inf)) if new_inf else 0.0

    s_counts = tracking.get("S", [])
    if s_counts:
        metrics["susceptible_fraction"] = float(s_counts[-1]) / max(num_agents, 1)
    else:
        metrics["susceptible_fraction"] = 1.0

    try:
        daily_deaths = state["environment"]["daily_deaths"].detach().cpu()
        if t < daily_deaths.numel():
            metrics["daily_deaths"] = float(daily_deaths.squeeze()[t])
        else:
            metrics["daily_deaths"] = 0.0
    except (KeyError, RuntimeError):
        metrics["daily_deaths"] = 0.0

    try:
        daily_hosp = state["environment"]["daily_hosp"].detach().cpu()
        if t < daily_hosp.numel():
            metrics["daily_hospitalizations"] = float(daily_hosp.squeeze()[t])
        else:
            metrics["daily_hospitalizations"] = 0.0
    except (KeyError, RuntimeError):
        metrics["daily_hospitalizations"] = 0.0

    avg_los = 8
    try:
        daily_hosp_arr = state["environment"]["daily_hosp"].detach().cpu().squeeze()
        recent_start = max(0, t - avg_los)
        occupancy = float(daily_hosp_arr[recent_start:t+1].sum()) if t >= 0 else 0.0
        real_occupancy = occupancy / max(scale_factor, 1e-9)
        metrics["hospital_occupancy"] = real_occupancy
        metrics["hospital_occupancy_fraction"] = real_occupancy / max(hc["hospital_beds"], 1)
    except (KeyError, RuntimeError, IndexError):
        metrics["hospital_occupancy"] = 0.0
        metrics["hospital_occupancy_fraction"] = 0.0

    icu_fraction = 0.25
    avg_icu_los = 12
    try:
        daily_hosp_arr = state["environment"]["daily_hosp"].detach().cpu().squeeze()
        recent_start = max(0, t - avg_icu_los)
        icu_occ = float(daily_hosp_arr[recent_start:t+1].sum()) * icu_fraction
        real_icu = icu_occ / max(scale_factor, 1e-9)
        metrics["icu_occupancy"] = real_icu
        metrics["icu_occupancy_fraction"] = real_icu / max(hc["icu_beds"], 1)
    except (KeyError, RuntimeError, IndexError):
        metrics["icu_occupancy"] = 0.0
        metrics["icu_occupancy_fraction"] = 0.0

    metrics["positivity_rate"] = 0.0

    muni_inf = tracking.get("municipality_infections", [])
    if muni_inf:
        import torch as _torch

        n_munis = len(muni_inf[0]) if muni_inf else 0

        muni_pop = tracking.get("municipality_population")
        if muni_pop is None:
            muni_pop = [num_agents / max(n_munis, 1)] * n_munis

        if len(muni_inf) >= 7:
            recent = muni_inf[-7:]
        else:
            recent = muni_inf
        muni_7d = [0.0] * n_munis
        for day_counts in recent:
            for i, count in enumerate(day_counts):
                muni_7d[i] += count
        muni_incidence = [
            muni_7d[i] / max(muni_pop[i] * scale_factor, 1e-9) * 100_000
            for i in range(n_munis)
        ]
        metrics["municipality_7day_incidence"] = max(muni_incidence) if muni_incidence else 0.0

        testing_tracking = tracking.get("_testing_tracking_ref")
        if testing_tracking and testing_tracking.get("daily_tests"):
            daily_tests = testing_tracking["daily_tests"]
            daily_positives = testing_tracking["daily_positives"]
            if daily_tests and daily_positives:
                recent_tests = sum(daily_tests[-7:]) if len(daily_tests) >= 7 else sum(daily_tests)
                recent_pos = sum(daily_positives[-7:]) if len(daily_positives) >= 7 else sum(daily_positives)
                metrics["positivity_rate"] = recent_pos / max(recent_tests, 1)
        metrics["municipality_positivity_rate"] = metrics.get("positivity_rate", 0.0)

        hotspot_threshold = 100.0
        metrics["hotspot_count"] = float(sum(1 for inc in muni_incidence if inc > hotspot_threshold))
    else:
        metrics["municipality_7day_incidence"] = 0.0
        metrics["municipality_positivity_rate"] = 0.0
        metrics["hotspot_count"] = 0.0

    return metrics


def denmark_tiered_response() -> List[TriggerPolicy]:
    """Denmark-style tiered response system (green/yellow/red)."""
    return [
        TriggerPolicy(
            name="yellow_alert",
            condition=TriggerCondition(
                metric=TriggerMetric.SEVEN_DAY_INCIDENCE,
                operator=TriggerOperator.GE,
                threshold=20.0,
                sustained_days=3,
            ),
            action="mask_mandate",
            strength=0.8,
            release_condition=TriggerCondition(
                metric=TriggerMetric.SEVEN_DAY_INCIDENCE,
                operator=TriggerOperator.LT,
                threshold=20.0,
                sustained_days=7,
            ),
            cooldown_days=14,
            min_active_days=7,
        ),
        TriggerPolicy(
            name="red_alert",
            condition=TriggerCondition(
                metric=TriggerMetric.SEVEN_DAY_INCIDENCE,
                operator=TriggerOperator.GE,
                threshold=120.0,
                sustained_days=3,
            ),
            action="lockdown_soft",
            strength=0.7,
            release_condition=TriggerCondition(
                metric=TriggerMetric.SEVEN_DAY_INCIDENCE,
                operator=TriggerOperator.LT,
                threshold=80.0,
                sustained_days=7,
            ),
            cooldown_days=21,
            min_active_days=14,
            ramp_up_days=3,
            ramp_down_days=7,
        ),
        TriggerPolicy(
            name="dark_red_alert",
            condition=TriggerCondition(
                metric=TriggerMetric.SEVEN_DAY_INCIDENCE,
                operator=TriggerOperator.GE,
                threshold=300.0,
                sustained_days=2,
            ),
            action="lockdown_hard",
            strength=1.0,
            release_condition=TriggerCondition(
                metric=TriggerMetric.SEVEN_DAY_INCIDENCE,
                operator=TriggerOperator.LT,
                threshold=150.0,
                sustained_days=10,
            ),
            cooldown_days=28,
            min_active_days=14,
            ramp_up_days=2,
            ramp_down_days=7,
        ),
    ]


def icu_capacity_response(
    icu_beds: int = 400,
    warn_fraction: float = 0.5,
    critical_fraction: float = 0.8,
) -> List[TriggerPolicy]:
    """ICU-capacity-driven response — the "flatten the curve" strategy."""
    return [
        TriggerPolicy(
            name="icu_warning",
            condition=TriggerCondition(
                metric=TriggerMetric.ICU_OCCUPANCY_FRACTION,
                operator=TriggerOperator.GE,
                threshold=warn_fraction,
                sustained_days=2,
            ),
            action="social_distancing",
            strength=0.8,
            release_condition=TriggerCondition(
                metric=TriggerMetric.ICU_OCCUPANCY_FRACTION,
                operator=TriggerOperator.LT,
                threshold=warn_fraction * 0.7,
                sustained_days=7,
            ),
            cooldown_days=14,
            min_active_days=7,
        ),
        TriggerPolicy(
            name="icu_critical",
            condition=TriggerCondition(
                metric=TriggerMetric.ICU_OCCUPANCY_FRACTION,
                operator=TriggerOperator.GE,
                threshold=critical_fraction,
                sustained_days=1,
            ),
            action="lockdown_hard",
            strength=1.0,
            release_condition=TriggerCondition(
                metric=TriggerMetric.ICU_OCCUPANCY_FRACTION,
                operator=TriggerOperator.LT,
                threshold=critical_fraction * 0.5,
                sustained_days=10,
            ),
            cooldown_days=21,
            min_active_days=14,
            ramp_up_days=2,
            ramp_down_days=7,
        ),
    ]


def rt_based_response() -> List[TriggerPolicy]:
    """Rt-based response — restrictions when reproduction number rises."""
    return [
        TriggerPolicy(
            name="rt_moderate",
            condition=TriggerCondition(
                metric=TriggerMetric.R_EFF,
                operator=TriggerOperator.GE,
                threshold=1.2,
                sustained_days=7,
            ),
            action="social_distancing",
            strength=0.7,
            release_condition=TriggerCondition(
                metric=TriggerMetric.R_EFF,
                operator=TriggerOperator.LT,
                threshold=0.9,
                sustained_days=14,
            ),
            cooldown_days=14,
            min_active_days=14,
        ),
        TriggerPolicy(
            name="rt_high",
            condition=TriggerCondition(
                metric=TriggerMetric.R_EFF,
                operator=TriggerOperator.GE,
                threshold=1.5,
                sustained_days=5,
            ),
            action="lockdown_soft",
            strength=0.8,
            release_condition=TriggerCondition(
                metric=TriggerMetric.R_EFF,
                operator=TriggerOperator.LT,
                threshold=1.0,
                sustained_days=14,
            ),
            cooldown_days=21,
            min_active_days=14,
            ramp_up_days=3,
        ),
    ]


TRIGGER_PRESETS: Dict[str, callable] = {
    "denmark_tiered": denmark_tiered_response,
    "icu_capacity": icu_capacity_response,
    "rt_based": rt_based_response,
}


def get_trigger_preset(name: str, **kwargs) -> List[TriggerPolicy]:
    """Get a preset trigger policy set by name."""
    if name not in TRIGGER_PRESETS:
        raise KeyError(
            f"Unknown trigger preset '{name}'. "
            f"Available: {', '.join(sorted(TRIGGER_PRESETS.keys()))}"
        )
    return TRIGGER_PRESETS[name](**kwargs)
