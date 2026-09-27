"""Immunity and vaccination system for the Danish ABM."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import numpy as np


@dataclass
class WaningFunction:
    """Parameterised waning of immunity over time."""
    kind: str = "exponential"
    half_life: float = 180.0
    midpoint: float = 180.0
    rate: float = 30.0
    duration: float = 365.0
    residual: float = 0.0
    half_life_ci95: Optional[tuple] = None

    def __call__(self, days_since: torch.Tensor) -> torch.Tensor:
        """Compute remaining fraction of immunity."""
        days = days_since.float().clamp(min=0)

        if self.kind == "none":
            return torch.ones_like(days)

        if self.kind == "exponential":
            decay = torch.pow(torch.tensor(0.5), days / self.half_life)
            return torch.max(decay, torch.tensor(self.residual))

        if self.kind == "sigmoid":
            logit = -(days - self.midpoint) / self.rate
            sig = torch.sigmoid(logit)
            return self.residual + (1.0 - self.residual) * sig

        if self.kind == "step":
            return torch.where(days < self.duration,
                               torch.ones_like(days),
                               torch.full_like(days, self.residual))

        if self.kind == "linear":
            frac = 1.0 - days / self.duration
            return torch.max(frac, torch.tensor(self.residual))

        raise ValueError(f"Unknown waning kind: {self.kind}")

    def sample_half_life(self, n: int) -> torch.Tensor:
        """Sample per-agent half-lives if CI is specified."""
        if self.half_life_ci95 is None:
            return torch.full((n,), self.half_life)
        from .disease_params import fit_gamma_from_ci
        params = fit_gamma_from_ci(self.half_life, self.half_life_ci95)
        dist = torch.distributions.Gamma(
            concentration=torch.tensor(params["shape"]),
            rate=torch.tensor(1.0 / params["scale"]),
        )
        return dist.sample((n,)).clamp(min=7.0)

    def describe(self) -> str:
        if self.kind == "none":
            return "No waning (permanent)"
        if self.kind == "exponential":
            ci = f" CI{self.half_life_ci95}" if self.half_life_ci95 else ""
            return f"Exponential(half_life={self.half_life:.0f}d{ci}, residual={self.residual})"
        if self.kind == "sigmoid":
            return f"Sigmoid(midpoint={self.midpoint:.0f}d, rate={self.rate:.0f}, residual={self.residual})"
        if self.kind == "step":
            return f"Step(duration={self.duration:.0f}d, residual={self.residual})"
        if self.kind == "linear":
            return f"Linear(duration={self.duration:.0f}d, residual={self.residual})"
        return f"{self.kind}(...)"


def no_waning():
    return WaningFunction(kind="none")

def exponential_waning(half_life=180.0, residual=0.0, ci95=None):
    return WaningFunction(kind="exponential", half_life=half_life,
                          residual=residual, half_life_ci95=ci95)

def sigmoid_waning(midpoint=180.0, rate=30.0, residual=0.0):
    return WaningFunction(kind="sigmoid", midpoint=midpoint, rate=rate, residual=residual)

def step_waning(duration=180.0, residual=0.2):
    return WaningFunction(kind="step", duration=duration, residual=residual)


@dataclass
class VaccineProfile:
    """A vaccine or prophylactic product."""
    name: str
    target_immunity: Dict[str, float] = field(default_factory=dict)
    waning: WaningFunction = field(default_factory=lambda: exponential_waning(180.0))
    doses: int = 2
    booster_multiplier: float = 1.0

    def describe(self) -> str:
        targets = ", ".join(f"{k}={v:.0%}" for k, v in self.target_immunity.items())
        return f"{self.name}: [{targets}], waning={self.waning.describe()}, doses={self.doses}"


VACCINE_PROFILES: Dict[str, VaccineProfile] = {
    "mrna_covid": VaccineProfile(
        name="mRNA (BNT162b2 / mRNA-1273)",
        target_immunity={"sars_cov_2": 0.95},
        waning=exponential_waning(half_life=150, residual=0.1),
        doses=2,
        booster_multiplier=0.9,
    ),
    "adenoviral_covid": VaccineProfile(
        name="Adenoviral (ChAdOx1)",
        target_immunity={"sars_cov_2": 0.70},
        waning=exponential_waning(half_life=120, residual=0.05),
        doses=2,
        booster_multiplier=0.85,
    ),
    "flu_inactivated": VaccineProfile(
        name="Inactivated influenza",
        target_immunity={"influenza": 0.50},
        waning=step_waning(duration=180, residual=0.1),
        doses=1,
    ),
    "rsv_nirsevimab": VaccineProfile(
        name="Nirsevimab (RSV prophylaxis)",
        target_immunity={"rsv": 0.75},
        waning=exponential_waning(half_life=90, residual=0.0),
        doses=1,
    ),
    "rsv_maternal": VaccineProfile(
        name="RSVpreF maternal vaccine",
        target_immunity={"rsv": 0.60},
        waning=exponential_waning(half_life=120, residual=0.0),
        doses=1,
    ),
}


class ImmunityTracker:
    """Tracks per-agent immune history from infection and vaccination."""

    def __init__(self, n_agents: int, antigen_names: List[str],
                 waning: Optional[WaningFunction] = None,
                 device: torch.device = None):
        self.n_agents = n_agents
        self.antigen_names = list(antigen_names)
        self.antigen_idx = {name: i for i, name in enumerate(self.antigen_names)}
        self.n_antigens = len(antigen_names)
        self.device = device or torch.device("cpu")
        self.default_waning = waning or exponential_waning(180.0)

        self.waning_fns: Dict[str, WaningFunction] = {
            name: self.default_waning for name in antigen_names
        }

        INFINITY = 1e6
        self.immunity_time = torch.full(
            (n_agents, self.n_antigens), INFINITY, device=self.device
        )
        self.immunity_initial = torch.zeros(
            (n_agents, self.n_antigens), device=self.device
        )

    def set_waning(self, antigen: str, waning: WaningFunction):
        """Override the waning function for a specific antigen."""
        self.waning_fns[antigen] = waning

    def record_event(self, agent_mask: torch.Tensor, antigen: str,
                     time: int, initial_efficacy: float = 1.0):
        """Record an immunising event (infection recovery or vaccination)."""
        idx = self.antigen_idx[antigen]
        mask = agent_mask.bool().squeeze()
        self.immunity_time[mask, idx] = float(time)
        self.immunity_initial[mask, idx] = initial_efficacy

    def record_vaccination(self, agent_mask: torch.Tensor, vaccine: VaccineProfile,
                           time: int, is_booster: bool = False):
        """Record vaccination using a VaccineProfile."""
        recorded = 0
        for antigen, efficacy in vaccine.target_immunity.items():
            if antigen not in self.antigen_idx:
                continue
            eff = efficacy * vaccine.booster_multiplier if is_booster else efficacy
            self.record_event(agent_mask, antigen, time, initial_efficacy=eff)
            self.waning_fns[antigen] = vaccine.waning
            recorded += 1

        if recorded == 0:
            raise ValueError(
                f"Vaccine {vaccine.name!r} targets {sorted(vaccine.target_immunity)} "
                f"but the immunity tracker only knows {self.antigen_names}. No "
                f"infection immunity would be recorded. Declare the antigen on the "
                f"StrainConfig (vaccine_antigen / vaccine_ve) so the tracker is built "
                f"with it."
            )

    def get_immunity(self, time: int) -> torch.Tensor:
        """Get current immunity levels for all agents and antigens."""
        days_since = float(time) - self.immunity_time
        current = torch.zeros_like(self.immunity_initial)

        for name, wfn in self.waning_fns.items():
            idx = self.antigen_idx[name]
            col_days = days_since[:, idx]
            col_initial = self.immunity_initial[:, idx]
            has_immunity = col_initial > 0
            if has_immunity.any():
                waned = wfn(col_days[has_immunity])
                current[has_immunity, idx] = col_initial[has_immunity] * waned

        return current

    def effective_susceptibility(self, target_strain: str,
                                 cross_immunity: torch.Tensor,
                                 time: int) -> torch.Tensor:
        """Compute per-agent susceptibility to a target strain."""
        current = self.get_immunity(time)

        if cross_immunity.shape[0] != self.n_antigens:
            raise ValueError(
                f"cross_immunity column has {cross_immunity.shape[0]} entries but "
                f"the tracker has {self.n_antigens} antigens {self.antigen_names}. "
                f"The two must be built from the same StrainConfig."
            )

        if cross_immunity.dim() == 1:
            ci_col = cross_immunity
        else:
            ci_col = cross_immunity

        one_minus_protection = torch.ones(self.n_agents, device=self.device)
        for j in range(self.n_antigens):
            protection_j = ci_col[j] * current[:, j]
            one_minus_protection = one_minus_protection * (1.0 - protection_j)

        return one_minus_protection.clamp(0.0, 1.0)

    def describe(self) -> str:
        lines = [f"ImmunityTracker: {self.n_agents} agents, {self.n_antigens} antigens"]
        for name in self.antigen_names:
            idx = self.antigen_idx[name]
            n_immune = (self.immunity_initial[:, idx] > 0).sum().item()
            wfn = self.waning_fns[name]
            lines.append(f"  {name}: {n_immune} immunised, waning={wfn.describe()}")
        return "\n".join(lines)


class LearnableWaning(nn.Module):
    """Waning function with learnable half-life."""

    def __init__(self, half_life_init: float = 180.0,
                 half_life_bounds: tuple = (30.0, 730.0),
                 residual: float = 0.0):
        super().__init__()
        self.lo, self.hi = half_life_bounds
        self.residual = residual
        normed = (half_life_init - self.lo) / (self.hi - self.lo)
        normed = max(0.01, min(0.99, normed))
        self._logit = nn.Parameter(torch.tensor(np.log(normed / (1 - normed))))

    @property
    def half_life(self) -> torch.Tensor:
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self._logit)

    def forward(self, days_since: torch.Tensor) -> torch.Tensor:
        hl = self.half_life
        decay = torch.pow(torch.tensor(0.5), days_since.float().clamp(min=0) / hl)
        return torch.max(decay, torch.tensor(self.residual))
