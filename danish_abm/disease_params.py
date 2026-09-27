"""Distributional disease parameters for the Danish ABM."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
import numpy as np
from scipy.stats import gamma as scipy_gamma
from scipy.optimize import minimize_scalar, minimize


def fit_gamma_from_ci(mean: float, ci95: Tuple[float, float]) -> Dict[str, float]:
    """Fit a Gamma distribution given mean and 95% CI bounds."""
    lo, hi = ci95
    assert lo < mean < hi, f"Mean {mean} must be between CI bounds ({lo}, {hi})"

    def objective(log_shape):
        a = np.exp(log_shape)
        b = mean / a
        q_lo = scipy_gamma.ppf(0.025, a=a, scale=b)
        q_hi = scipy_gamma.ppf(0.975, a=a, scale=b)
        return (q_lo - lo)**2 + (q_hi - hi)**2

    result = minimize_scalar(objective, bounds=(-3, 6), method='bounded')
    a = np.exp(result.x)
    b = mean / a
    return {"shape": float(a), "scale": float(b), "rate": float(1.0 / b)}


def fit_lognormal_from_ci(mean: float, ci95: Tuple[float, float]) -> Dict[str, float]:
    """Fit a LogNormal distribution given mean and 95% CI bounds."""
    lo, hi = ci95
    assert lo < mean < hi

    def objective(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        pred_mean = np.exp(mu + sigma**2 / 2)
        q_lo = np.exp(mu - 1.96 * sigma)
        q_hi = np.exp(mu + 1.96 * sigma)
        return (pred_mean - mean)**2 + (q_lo - lo)**2 + (q_hi - hi)**2

    mu0 = np.log(mean) - 0.5 * 0.5**2
    result = minimize(objective, x0=[mu0, np.log(0.5)], method='Nelder-Mead')
    mu, log_sigma = result.x
    return {"mu": float(mu), "sigma": float(np.exp(log_sigma))}


@dataclass
class ParamSpec:
    """A single disease parameter specified with uncertainty."""
    mean: float
    ci95: Optional[Tuple[float, float]] = None
    distribution: str = "gamma"
    dist_params: Optional[Dict[str, float]] = None
    min_value: float = 1.0

    def __post_init__(self):
        if self.ci95 is None:
            self.distribution = "fixed"
        if self.dist_params is None and self.ci95 is not None:
            self._fit()

    def _fit(self):
        if self.distribution == "gamma":
            self.dist_params = fit_gamma_from_ci(self.mean, self.ci95)
        elif self.distribution == "lognormal":
            self.dist_params = fit_lognormal_from_ci(self.mean, self.ci95)

    def sample(self, n: int, device: torch.device = None) -> torch.Tensor:
        """Sample n durations, rounded to integers, clamped >= min_value."""
        if self.distribution == "fixed":
            return torch.full((n,), self.mean, device=device).round().clamp(min=self.min_value)

        if self.distribution == "gamma":
            a = self.dist_params["shape"]
            b = self.dist_params["scale"]
            dist = torch.distributions.Gamma(
                concentration=torch.tensor(a),
                rate=torch.tensor(1.0 / b),
            )
            samples = dist.sample((n,))
        elif self.distribution == "lognormal":
            mu = self.dist_params["mu"]
            sigma = self.dist_params["sigma"]
            dist = torch.distributions.LogNormal(
                loc=torch.tensor(mu),
                scale=torch.tensor(sigma),
            )
            samples = dist.sample((n,))
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        result = samples.round().clamp(min=self.min_value)
        if device is not None:
            result = result.to(device)
        return result

    def summary(self) -> str:
        if self.distribution == "fixed":
            return f"Fixed({self.mean})"
        params_str = ", ".join(f"{k}={v:.3f}" for k, v in self.dist_params.items())
        ci_str = f"95% CI [{self.ci95[0]:.1f}, {self.ci95[1]:.1f}]" if self.ci95 else ""
        return f"{self.distribution.capitalize()}({params_str}) mean={self.mean:.1f} {ci_str}"


@dataclass
class DiseaseParams:
    """Collection of disease timing parameters with uncertainty."""
    latent_period: ParamSpec = field(default_factory=lambda: ParamSpec(mean=3.0))
    infectious_period: ParamSpec = field(default_factory=lambda: ParamSpec(mean=5.0))
    generation_time: ParamSpec = field(default_factory=lambda: ParamSpec(mean=5.5))
    immunity_waning: ParamSpec = field(default_factory=lambda: ParamSpec(mean=999.0))

    @classmethod
    def from_ci(cls, latent_period=None, infectious_period=None,
                generation_time=None, immunity_waning=None, **kwargs) -> "DiseaseParams":
        """Create from dicts with 'mean' and optional 'ci95' keys."""
        def _make_spec(d, default_mean):
            if d is None:
                return ParamSpec(mean=default_mean)
            if isinstance(d, (int, float)):
                return ParamSpec(mean=float(d))
            return ParamSpec(**d)

        return cls(
            latent_period=_make_spec(latent_period, 3.0),
            infectious_period=_make_spec(infectious_period, 5.0),
            generation_time=_make_spec(generation_time, 5.5),
            immunity_waning=_make_spec(immunity_waning, 999.0),
        )

    def sample_latent(self, n: int, device=None) -> torch.Tensor:
        return self.latent_period.sample(n, device)

    def sample_infectious(self, n: int, device=None) -> torch.Tensor:
        return self.infectious_period.sample(n, device)

    def sample_waning(self, n: int, device=None) -> torch.Tensor:
        return self.immunity_waning.sample(n, device)

    def generation_time_integrals(self, n_steps: int) -> torch.Tensor:
        """Compute gamma CDF integrals for the generation time distribution."""
        if self.generation_time.distribution == "fixed":
            p = self.generation_time.dist_params or {}
            a = p.get("shape", self.generation_time.mean)
            b = p.get("scale", 1.0)
        else:
            a = self.generation_time.dist_params["shape"]
            b = self.generation_time.dist_params["scale"]

        integrals = np.array([
            scipy_gamma.cdf(t, a=a, scale=b) - scipy_gamma.cdf(t - 1, a=a, scale=b)
            for t in range(n_steps)
        ])
        return torch.tensor(integrals, dtype=torch.float32)

    def to_config_overrides(self) -> dict:
        """Generate config overrides for injection into the YAML config."""
        return {
            "EXPOSED_TO_INFECTED_TIME": round(self.latent_period.mean),
            "INFECTED_TO_RECOVERED_TIME": round(self.infectious_period.mean),
            "RECOVERED_TO_SUSCEPTIBLE_TIME": round(self.immunity_waning.mean),
            "disease_params_distributional": True,
            "disease_params": {
                "latent_period": {
                    "mean": self.latent_period.mean,
                    "ci95": list(self.latent_period.ci95) if self.latent_period.ci95 else None,
                    "distribution": self.latent_period.distribution,
                },
                "infectious_period": {
                    "mean": self.infectious_period.mean,
                    "ci95": list(self.infectious_period.ci95) if self.infectious_period.ci95 else None,
                    "distribution": self.infectious_period.distribution,
                },
                "generation_time": {
                    "mean": self.generation_time.mean,
                    "ci95": list(self.generation_time.ci95) if self.generation_time.ci95 else None,
                    "distribution": self.generation_time.distribution,
                },
                "immunity_waning": {
                    "mean": self.immunity_waning.mean,
                    "ci95": list(self.immunity_waning.ci95) if self.immunity_waning.ci95 else None,
                    "distribution": self.immunity_waning.distribution,
                },
            },
        }

    def describe(self) -> str:
        lines = ["Disease Parameters:"]
        lines.append(f"  Latent period:     {self.latent_period.summary()}")
        lines.append(f"  Infectious period: {self.infectious_period.summary()}")
        lines.append(f"  Generation time:   {self.generation_time.summary()}")
        lines.append(f"  Immunity waning:   {self.immunity_waning.summary()}")
        return "\n".join(lines)


PATHOGEN_PRESETS: Dict[str, DiseaseParams] = {}


def register_pathogen(name: str, params: DiseaseParams):
    PATHOGEN_PRESETS[name] = params


def get_pathogen(name: str) -> DiseaseParams:
    if name not in PATHOGEN_PRESETS:
        raise KeyError(f"Unknown pathogen '{name}'. Available: {list(PATHOGEN_PRESETS.keys())}")
    return PATHOGEN_PRESETS[name]


register_pathogen("sars_cov_2", DiseaseParams.from_ci(
    latent_period={"mean": 3.0, "ci95": (1.0, 6.0)},
    infectious_period={"mean": 5.0, "ci95": (3.0, 8.0)},
    generation_time={"mean": 5.5, "ci95": (2.5, 9.5)},
    immunity_waning={"mean": 180.0, "ci95": (90.0, 365.0)},
))

register_pathogen("influenza", DiseaseParams.from_ci(
    latent_period={"mean": 2.0, "ci95": (1.0, 4.0)},
    infectious_period={"mean": 5.0, "ci95": (3.0, 7.0)},
    generation_time={"mean": 3.0, "ci95": (1.5, 5.5)},
    immunity_waning={"mean": 365.0, "ci95": (180.0, 730.0)},
))

register_pathogen("rsv", DiseaseParams.from_ci(
    latent_period={"mean": 4.0, "ci95": (2.0, 7.0)},
    infectious_period={"mean": 6.0, "ci95": (3.0, 10.0)},
    generation_time={"mean": 7.0, "ci95": (4.0, 11.0)},
    immunity_waning={"mean": 200.0, "ci95": (120.0, 365.0)},
))


def disease_params_from_compartment_model(cm) -> DiseaseParams:
    """Extract DiseaseParams timing information from a CompartmentModel."""
    latent = {"mean": 3.0}
    infectious = {"mean": 5.0}
    gen_time = {"mean": 5.5}
    waning = {"mean": 999.0}

    for t in cm.transitions:
        if t.from_state == "E" and t.to_state == "I" and t.duration:
            latent = {"mean": t.duration.mean}
            if t.duration.ci95:
                latent["ci95"] = t.duration.ci95
        elif t.from_state == "I" and t.to_state == "R" and t.duration:
            infectious = {"mean": t.duration.mean}
            if t.duration.ci95:
                infectious["ci95"] = t.duration.ci95
        elif t.mechanism == "waning" and t.duration:
            waning = {"mean": t.duration.mean}
            if t.duration.ci95:
                waning["ci95"] = t.duration.ci95

    if cm.generation_time:
        gen_time = {"mean": cm.generation_time["mean"]}
        if "ci95" in cm.generation_time:
            gen_time["ci95"] = tuple(cm.generation_time["ci95"])

    return DiseaseParams.from_ci(
        latent_period=latent,
        infectious_period=infectious,
        generation_time=gen_time,
        immunity_waning=waning,
    )
