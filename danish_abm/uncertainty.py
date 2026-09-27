"""Parameter uncertainty propagation for the Danish ABM."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class UncertaintyMode(Enum):
    STOCHASTIC = auto()
    PARAMETER = auto()
    STRUCTURAL = auto()
    FULL = auto()


@dataclass
class ParameterPrior:
    """A prior distribution for a single model parameter."""
    name: str
    distribution: str
    params: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    bounds: Optional[Tuple[float, float]] = None

    def sample(self, rng: np.random.Generator = None) -> float:
        """Draw one sample from this prior."""
        rng = rng or np.random.default_rng()

        if self.distribution == "fixed":
            val = self.params.get("value", 0.0)

        elif self.distribution == "normal":
            val = rng.normal(self.params["mean"], self.params["std"])

        elif self.distribution == "lognormal":
            mean = self.params["mean"]
            cv = self.params.get("cv", 0.15)
            sigma2 = np.log(1 + cv**2)
            mu = np.log(mean) - sigma2 / 2
            val = rng.lognormal(mu, np.sqrt(sigma2))

        elif self.distribution == "beta":
            val = rng.beta(self.params["a"], self.params["b"])

        elif self.distribution == "gamma":
            val = rng.gamma(self.params["shape"], self.params["scale"])

        elif self.distribution == "uniform":
            val = rng.uniform(self.params["low"], self.params["high"])

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        if self.bounds:
            val = np.clip(val, self.bounds[0], self.bounds[1])

        return float(val)

    def describe(self) -> str:
        if self.distribution == "fixed":
            return f"{self.name}: fixed at {self.params.get('value', 0)}"
        if self.distribution == "lognormal":
            return (f"{self.name}: LogNormal(mean={self.params['mean']:.4f}, "
                    f"cv={self.params.get('cv', 0.15):.0%})")
        if self.distribution == "beta":
            a, b = self.params["a"], self.params["b"]
            return f"{self.name}: Beta(a={a:.1f}, b={b:.1f}), mean={a/(a+b):.3f}"
        if self.distribution == "gamma":
            sh, sc = self.params["shape"], self.params["scale"]
            return f"{self.name}: Gamma(shape={sh:.2f}, scale={sc:.2f}), mean={sh*sc:.1f}"
        if self.distribution == "uniform":
            return f"{self.name}: Uniform({self.params['low']:.3f}, {self.params['high']:.3f})"
        if self.distribution == "normal":
            return f"{self.name}: Normal(mean={self.params['mean']:.3f}, std={self.params['std']:.3f})"
        return f"{self.name}: {self.distribution}({self.params})"


@dataclass
class ParameterSample:
    """A single draw from all parameter priors."""
    values: Dict[str, float] = field(default_factory=dict)
    structural_label: Optional[str] = None

    def apply_to_config(self, config: dict) -> dict:
        """Apply parameter overrides to a model config. Returns modified copy."""
        config = copy.deepcopy(config)
        meta = config["simulation_metadata"]

        v = self.values

        if "H" in v:
            config["substeps"]["1"]["transition"]["seirm_progression"]["arguments"]["H"]["value"] = v["H"]
        if "M" in v:
            config["substeps"]["1"]["transition"]["seirm_progression"]["arguments"]["M"]["value"] = v["M"]

        def _override_disease_param(dp, key, new_mean, config_key):
            if key not in dp:
                dp[key] = {}
            old_mean = dp[key].get("mean", new_mean)
            dp[key]["mean"] = new_mean
            ci = dp[key].get("ci95")
            if ci and old_mean > 0:
                ratio = new_mean / old_mean
                dp[key]["ci95"] = [max(0.5, ci[0] * ratio), ci[1] * ratio]
            meta[config_key] = round(new_mean)

        dp = meta.get("disease_params", {})

        if "latent_period_mean" in v:
            _override_disease_param(dp, "latent_period", v["latent_period_mean"],
                                    "EXPOSED_TO_INFECTED_TIME")

        if "infectious_period_mean" in v:
            _override_disease_param(dp, "infectious_period", v["infectious_period_mean"],
                                    "INFECTED_TO_RECOVERED_TIME")

        if "immunity_waning_mean" in v:
            _override_disease_param(dp, "immunity_waning", v["immunity_waning_mean"],
                                    "RECOVERED_TO_SUSCEPTIBLE_TIME")

        meta["disease_params"] = dp

        if "importation_rate" in v:
            meta["daily_importation_rate"] = v["importation_rate"]

        if "initial_infection_ratio" in v:
            meta["initial_infection_ratio"] = v["initial_infection_ratio"]

        if "vaccine_efficacy_scale" in v:
            meta["_vaccine_efficacy_scale"] = v["vaccine_efficacy_scale"]

        if "test_sensitivity" in v:
            meta["test_sensitivity"] = v["test_sensitivity"]
        if "isolation_compliance" in v:
            meta["isolation_compliance"] = v["isolation_compliance"]

        if "npi_effectiveness_scale" in v:
            meta["_npi_effectiveness_scale"] = v["npi_effectiveness_scale"]

        if "reinfection_susceptibility" in v:
            meta["REINFECTION_SUSCEPTIBILITY"] = v["reinfection_susceptibility"]

        if "charlson_or_scale" in v:
            meta["_charlson_or_scale"] = v["charlson_or_scale"]
        if "age_ifr_scale" in v:
            meta["_age_ifr_scale"] = v["age_ifr_scale"]

        if "hospitalization_delay" in v:
            meta["hospitalization_delay"] = round(v["hospitalization_delay"])

        return config

    def describe(self) -> str:
        parts = []
        if self.structural_label:
            parts.append(f"[{self.structural_label}]")
        for k, val in sorted(self.values.items()):
            parts.append(f"{k}={val:.4g}")
        return ", ".join(parts)


@dataclass
class StructuralAssumption:
    """An alternative model structure or fixed parameter choice."""
    name: str
    description: str = ""
    overrides: Dict[str, float] = field(default_factory=dict)

    def to_sample(self) -> ParameterSample:
        return ParameterSample(values=dict(self.overrides), structural_label=self.name)


@dataclass
class UncertaintyConfig:
    """Full uncertainty specification for a model run."""
    mode: UncertaintyMode = UncertaintyMode.PARAMETER
    priors: Dict[str, ParameterPrior] = field(default_factory=dict)
    structural_assumptions: List[StructuralAssumption] = field(default_factory=list)


    def sample(self, rng: np.random.Generator = None) -> ParameterSample:
        """Draw one parameter set from all priors."""
        rng = rng or np.random.default_rng()
        values = {}
        for name, prior in self.priors.items():
            values[name] = prior.sample(rng)
        return ParameterSample(values=values)

    def sample_n(self, n: int, seed: int = None) -> List[ParameterSample]:
        """Draw n parameter sets using Latin Hypercube Sampling for better coverage."""
        rng = np.random.default_rng(seed)

        if not self.priors:
            return [ParameterSample() for _ in range(n)]

        names = list(self.priors.keys())
        k = len(names)
        samples = []

        lhs_uniform = np.zeros((n, k))
        for j in range(k):
            perm = rng.permutation(n)
            for i in range(n):
                lhs_uniform[perm[i], j] = (i + rng.uniform()) / n

        for idx in range(n):
            values = {}
            for j, name in enumerate(names):
                prior = self.priors[name]
                u = lhs_uniform[idx, j]
                values[name] = _quantile(prior, u)
                if prior.bounds:
                    values[name] = np.clip(values[name], prior.bounds[0], prior.bounds[1])
            samples.append(ParameterSample(values=values))

        return samples

    def get_structural_samples(self) -> List[ParameterSample]:
        """Get one ParameterSample per structural assumption."""
        return [a.to_sample() for a in self.structural_assumptions]

    def get_ensemble_plan(self, n_runs: int, seed: int = None) -> List[ParameterSample]:
        """Generate the full set of parameter samples for an ensemble."""
        if self.mode == UncertaintyMode.STOCHASTIC:
            return [ParameterSample() for _ in range(n_runs)]

        if self.mode == UncertaintyMode.PARAMETER:
            return self.sample_n(n_runs, seed=seed)

        if self.mode == UncertaintyMode.STRUCTURAL:
            return self.get_structural_samples()

        if self.mode == UncertaintyMode.FULL:
            param_samples = self.sample_n(n_runs, seed=seed)
            struct_samples = self.get_structural_samples()
            combined = []
            for struct in (struct_samples or [ParameterSample()]):
                for param in param_samples:
                    merged = ParameterSample(
                        values={**param.values, **struct.values},
                        structural_label=struct.structural_label,
                    )
                    combined.append(merged)
            return combined

        return [ParameterSample() for _ in range(n_runs)]


    @classmethod
    def stochastic_only(cls) -> "UncertaintyConfig":
        """No parameter uncertainty — just stochastic variance."""
        return cls(mode=UncertaintyMode.STOCHASTIC)

    @classmethod
    def default_covid(cls) -> "UncertaintyConfig":
        """Default parameter priors for COVID-19 modeling."""
        priors = {
            "R_scale": ParameterPrior(
                name="R_scale",
                distribution="lognormal",
                params={"mean": 1.0, "cv": 0.15},
                description="Multiplicative scaling on weekly R (±15%)",
                bounds=(0.5, 2.0),
            ),
            "H": ParameterPrior(
                name="H",
                distribution="beta",
                params={"a": 5.0, "b": 95.0},
                description="Probability of hospitalization given infection",
                bounds=(0.01, 0.15),
            ),
            "M": ParameterPrior(
                name="M",
                distribution="beta",
                params={"a": 12.0, "b": 88.0},
                description="Probability of death given infection",
                bounds=(0.02, 0.25),
            ),
            "latent_period_mean": ParameterPrior(
                name="latent_period_mean",
                distribution="gamma",
                params={"shape": 9.0, "scale": 0.333},
                description="Mean latent period (days)",
                bounds=(1.0, 7.0),
            ),
            "infectious_period_mean": ParameterPrior(
                name="infectious_period_mean",
                distribution="gamma",
                params={"shape": 12.5, "scale": 0.4},
                description="Mean infectious period (days)",
                bounds=(2.0, 8.0),
            ),
            "immunity_waning_mean": ParameterPrior(
                name="immunity_waning_mean",
                distribution="gamma",
                params={"shape": 4.0, "scale": 45.0},
                description="Mean immunity waning time (days)",
                bounds=(60.0, 730.0),
            ),
            "importation_rate": ParameterPrior(
                name="importation_rate",
                distribution="lognormal",
                params={"mean": 5e-5, "cv": 0.5},
                description="Daily importation rate per susceptible",
                bounds=(1e-6, 5e-4),
            ),
            "vaccine_efficacy_scale": ParameterPrior(
                name="vaccine_efficacy_scale",
                distribution="beta",
                params={"a": 45.0, "b": 5.0},
                description="Multiplicative scaling on vaccine efficacy",
                bounds=(0.5, 1.0),
            ),
            "npi_effectiveness_scale": ParameterPrior(
                name="npi_effectiveness_scale",
                distribution="lognormal",
                params={"mean": 1.0, "cv": 0.2},
                description="How well NPIs actually work vs assumed (±20%)",
                bounds=(0.5, 1.5),
            ),
            "test_sensitivity": ParameterPrior(
                name="test_sensitivity",
                distribution="beta",
                params={"a": 40.0, "b": 10.0},
                description="Test sensitivity (true positive rate)",
                bounds=(0.5, 1.0),
            ),
            "isolation_compliance": ParameterPrior(
                name="isolation_compliance",
                distribution="beta",
                params={"a": 16.0, "b": 4.0},
                description="Fraction of detected cases that actually isolate",
                bounds=(0.3, 1.0),
            ),
            "reinfection_susceptibility": ParameterPrior(
                name="reinfection_susceptibility",
                distribution="beta",
                params={"a": 5.0, "b": 5.0},
                description="Susceptibility on reinfection (0=full protection, 1=none)",
                bounds=(0.1, 0.9),
            ),
            "charlson_or_scale": ParameterPrior(
                name="charlson_or_scale",
                distribution="lognormal",
                params={"mean": 1.0, "cv": 0.15},
                description="Multiplicative scaling on Charlson ORs (±15%, Christensen 2020)",
                bounds=(0.7, 1.3),
            ),
            "age_ifr_scale": ParameterPrior(
                name="age_ifr_scale",
                distribution="lognormal",
                params={"mean": 1.0, "cv": 0.15},
                description="Multiplicative scaling on age-relative IFR (±15%, O'Driscoll 2021)",
                bounds=(0.7, 1.3),
            ),
        }

        structural = [
            StructuralAssumption(
                name="fast_waning",
                description="Immunity wanes in ~3 months (pessimistic)",
                overrides={"immunity_waning_mean": 90.0},
            ),
            StructuralAssumption(
                name="slow_waning",
                description="Immunity wanes in ~12 months (optimistic)",
                overrides={"immunity_waning_mean": 365.0},
            ),
            StructuralAssumption(
                name="high_severity",
                description="Higher severity variant (H=0.08, M=0.15)",
                overrides={"H": 0.08, "M": 0.15},
            ),
            StructuralAssumption(
                name="low_severity",
                description="Lower severity (Omicron-like: H=0.02, M=0.05)",
                overrides={"H": 0.02, "M": 0.05},
            ),
            StructuralAssumption(
                name="poor_npi_compliance",
                description="NPIs only 60% as effective as modeled",
                overrides={"npi_effectiveness_scale": 0.6},
            ),
        ]

        return cls(
            mode=UncertaintyMode.PARAMETER,
            priors=priors,
            structural_assumptions=structural,
        )

    @classmethod
    def default_influenza(cls) -> "UncertaintyConfig":
        """Default priors for influenza modeling."""
        priors = {
            "R_scale": ParameterPrior(
                name="R_scale",
                distribution="lognormal",
                params={"mean": 1.0, "cv": 0.2},
                bounds=(0.5, 2.0),
            ),
            "H": ParameterPrior(
                name="H",
                distribution="beta",
                params={"a": 2.0, "b": 98.0},
                bounds=(0.005, 0.08),
            ),
            "M": ParameterPrior(
                name="M",
                distribution="beta",
                params={"a": 1.0, "b": 999.0},
                bounds=(0.0001, 0.01),
            ),
            "latent_period_mean": ParameterPrior(
                name="latent_period_mean",
                distribution="gamma",
                params={"shape": 8.0, "scale": 0.25},
                bounds=(1.0, 5.0),
            ),
            "infectious_period_mean": ParameterPrior(
                name="infectious_period_mean",
                distribution="gamma",
                params={"shape": 12.5, "scale": 0.4},
                bounds=(2.0, 8.0),
            ),
            "vaccine_efficacy_scale": ParameterPrior(
                name="vaccine_efficacy_scale",
                distribution="beta",
                params={"a": 10.0, "b": 10.0},
                bounds=(0.2, 0.8),
            ),
        }
        return cls(mode=UncertaintyMode.PARAMETER, priors=priors)


    @classmethod
    def wild_type(cls) -> "UncertaintyConfig":
        """Parameter priors for SARS-CoV-2 wild-type (Wuhan/D614G)."""
        priors = {
            "R_scale": ParameterPrior(
                name="R_scale", distribution="lognormal",
                params={"mean": 1.0, "cv": 0.15},
                description="Multiplicative on calibrated R(t)",
                bounds=(0.5, 2.0),
            ),
            "H": ParameterPrior(
                name="H", distribution="beta",
                params={"a": 4.0, "b": 96.0},
                description="IHR — Salje et al. 2020 Science (France): 3.6% [2.1–6.2%]",
                bounds=(0.01, 0.10),
            ),
            "M": ParameterPrior(
                name="M", distribution="beta",
                params={"a": 3.0, "b": 497.0},
                description="IFR — Meyerowitz-Katz & Merone 2021: 0.68% [0.53–0.82%]",
                bounds=(0.002, 0.015),
            ),
            "latent_period_mean": ParameterPrior(
                name="latent_period_mean", distribution="gamma",
                params={"shape": 13.0, "scale": 0.4},
                description="Incubation — Lauer et al. 2020 Ann Intern Med: 5.1d [4.5–5.8]",
                bounds=(3.0, 8.0),
            ),
            "infectious_period_mean": ParameterPrior(
                name="infectious_period_mean", distribution="gamma",
                params={"shape": 12.5, "scale": 0.4},
                description="Infectious — He et al. 2020 Nat Med: ~5d [3–8] from onset",
                bounds=(2.0, 8.0),
            ),
            "immunity_waning_mean": ParameterPrior(
                name="immunity_waning_mean", distribution="gamma",
                params={"shape": 4.0, "scale": 91.0},
                description="Waning — Lumley et al. 2021 NEJM: >6 months robust, waning ~1yr",
                bounds=(180.0, 730.0),
            ),
            "importation_rate": ParameterPrior(
                name="importation_rate", distribution="lognormal",
                params={"mean": 5e-5, "cv": 0.5}, bounds=(1e-6, 5e-4),
            ),
            "vaccine_efficacy_scale": ParameterPrior(
                name="vaccine_efficacy_scale", distribution="fixed",
                params={"value": 1.0},
                description="No vaccines available in wild-type wave",
            ),
            "npi_effectiveness_scale": ParameterPrior(
                name="npi_effectiveness_scale", distribution="lognormal",
                params={"mean": 1.0, "cv": 0.2}, bounds=(0.5, 1.5),
            ),
            "test_sensitivity": ParameterPrior(
                name="test_sensitivity", distribution="beta",
                params={"a": 48.6, "b": 1.4},
                description="PCR — Dinnes et al. 2022 Cochrane: 97.2% [94.2–98.7%]",
                bounds=(0.85, 1.0),
            ),
            "isolation_compliance": ParameterPrior(
                name="isolation_compliance", distribution="beta",
                params={"a": 16.0, "b": 4.0},
                bounds=(0.3, 1.0),
            ),
            "reinfection_susceptibility": ParameterPrior(
                name="reinfection_susceptibility", distribution="fixed",
                params={"value": 0.0},
                description="Negligible reinfection in first wave",
            ),
        }
        return cls(mode=UncertaintyMode.PARAMETER, priors=priors)

    @classmethod
    def alpha(cls) -> "UncertaintyConfig":
        """Parameter priors for SARS-CoV-2 Alpha (B.1.1.7)."""
        priors = {
            "R_scale": ParameterPrior(
                name="R_scale", distribution="lognormal",
                params={"mean": 1.0, "cv": 0.15}, bounds=(0.5, 2.0),
            ),
            "H": ParameterPrior(
                name="H", distribution="beta",
                params={"a": 5.0, "b": 95.0},
                description="IHR — Nyberg et al. 2022 Lancet ID: HR 1.52 vs wild-type → ~5%",
                bounds=(0.02, 0.10),
            ),
            "M": ParameterPrior(
                name="M", distribution="beta",
                params={"a": 4.0, "b": 496.0},
                description="IFR — Challen et al. 2021 BMJ: HR 1.64 [1.32–2.04] vs wild-type",
                bounds=(0.003, 0.020),
            ),
            "latent_period_mean": ParameterPrior(
                name="latent_period_mean", distribution="gamma",
                params={"shape": 13.5, "scale": 0.333},
                description="Incubation — Tanaka et al. 2022 J Clin Med: 4.5d [3.5–6.0]",
                bounds=(2.5, 7.0),
            ),
            "infectious_period_mean": ParameterPrior(
                name="infectious_period_mean", distribution="gamma",
                params={"shape": 12.5, "scale": 0.4},
                description="Infectious — similar to wild-type (He et al. 2020)",
                bounds=(2.0, 8.0),
            ),
            "immunity_waning_mean": ParameterPrior(
                name="immunity_waning_mean", distribution="gamma",
                params={"shape": 4.0, "scale": 67.5},
                description="Waning — Nordström et al. 2022 Lancet ID: ~9 months",
                bounds=(90.0, 500.0),
            ),
            "importation_rate": ParameterPrior(
                name="importation_rate", distribution="lognormal",
                params={"mean": 5e-5, "cv": 0.5}, bounds=(1e-6, 5e-4),
            ),
            "vaccine_efficacy_scale": ParameterPrior(
                name="vaccine_efficacy_scale", distribution="beta",
                params={"a": 40.0, "b": 10.0},
                description="VE vs Alpha — Lopez Bernal et al. 2021 NEJM: 93.7% (2 doses BNT)",
                bounds=(0.5, 1.0),
            ),
            "npi_effectiveness_scale": ParameterPrior(
                name="npi_effectiveness_scale", distribution="lognormal",
                params={"mean": 1.0, "cv": 0.2}, bounds=(0.5, 1.5),
            ),
            "test_sensitivity": ParameterPrior(
                name="test_sensitivity", distribution="beta",
                params={"a": 48.6, "b": 1.4},
                description="PCR — Dinnes et al. 2022 Cochrane: 97.2%",
                bounds=(0.85, 1.0),
            ),
            "isolation_compliance": ParameterPrior(
                name="isolation_compliance", distribution="beta",
                params={"a": 16.0, "b": 4.0}, bounds=(0.3, 1.0),
            ),
            "reinfection_susceptibility": ParameterPrior(
                name="reinfection_susceptibility", distribution="beta",
                params={"a": 3.0, "b": 17.0},
                description="Low reinfection risk — Hansen et al. 2021 Lancet: 80% protection",
                bounds=(0.02, 0.4),
            ),
        }
        return cls(mode=UncertaintyMode.PARAMETER, priors=priors)

    @classmethod
    def omicron(cls) -> "UncertaintyConfig":
        """Parameter priors for SARS-CoV-2 Omicron (BA.1/BA.2)."""
        priors = {
            "R_scale": ParameterPrior(
                name="R_scale", distribution="lognormal",
                params={"mean": 1.0, "cv": 0.20},
                description="Wider uncertainty for Omicron (Liu & Rocklöv 2022)",
                bounds=(0.4, 2.5),
            ),
            "H": ParameterPrior(
                name="H", distribution="beta",
                params={"a": 2.0, "b": 98.0},
                description="IHR — Nyberg et al. 2022 Lancet: HR 0.33 vs Delta → ~2%",
                bounds=(0.005, 0.06),
            ),
            "M": ParameterPrior(
                name="M", distribution="beta",
                params={"a": 1.0, "b": 999.0},
                description="IFR — Axfors & Ioannidis 2023: 0.03–0.07% (vaccinated pop)",
                bounds=(0.0002, 0.005),
            ),
            "latent_period_mean": ParameterPrior(
                name="latent_period_mean", distribution="gamma",
                params={"shape": 14.0, "scale": 0.25},
                description="Incubation — Jansen et al. 2022 Eurosurveillance: 3.4d [2.5–4.9]",
                bounds=(2.0, 6.0),
            ),
            "infectious_period_mean": ParameterPrior(
                name="infectious_period_mean", distribution="gamma",
                params={"shape": 12.5, "scale": 0.4},
                description="Infectious — Boucau et al. 2022 NEJM: ~5d viral clearance",
                bounds=(2.0, 8.0),
            ),
            "immunity_waning_mean": ParameterPrior(
                name="immunity_waning_mean", distribution="gamma",
                params={"shape": 4.0, "scale": 45.0},
                description="Waning — Tartof et al. 2022 Lancet: faster waning vs Omicron",
                bounds=(60.0, 365.0),
            ),
            "importation_rate": ParameterPrior(
                name="importation_rate", distribution="lognormal",
                params={"mean": 1e-4, "cv": 0.5},
                bounds=(1e-5, 1e-3),
            ),
            "vaccine_efficacy_scale": ParameterPrior(
                name="vaccine_efficacy_scale", distribution="beta",
                params={"a": 32.5, "b": 17.5},
                description="VE vs Omicron — Andrews et al. 2022 NEJM: ~65% post-booster",
                bounds=(0.3, 0.9),
            ),
            "npi_effectiveness_scale": ParameterPrior(
                name="npi_effectiveness_scale", distribution="lognormal",
                params={"mean": 1.0, "cv": 0.25},
                bounds=(0.4, 1.5),
            ),
            "test_sensitivity": ParameterPrior(
                name="test_sensitivity", distribution="beta",
                params={"a": 34.2, "b": 15.8},
                description="Antigen-weighted — Dinnes et al. 2022 Cochrane (Denmark used mass RAT)",
                bounds=(0.4, 0.95),
            ),
            "isolation_compliance": ParameterPrior(
                name="isolation_compliance", distribution="beta",
                params={"a": 12.0, "b": 8.0},
                description="Lower compliance in Omicron era (fatigue, mild symptoms)",
                bounds=(0.2, 0.9),
            ),
            "reinfection_susceptibility": ParameterPrior(
                name="reinfection_susceptibility", distribution="beta",
                params={"a": 5.0, "b": 5.0},
                description="Immune escape — Altarawneh et al. 2022 NEJM: 56% protection from prior infection",
                bounds=(0.2, 0.8),
            ),
        }
        return cls(mode=UncertaintyMode.PARAMETER, priors=priors)

    @classmethod
    def for_variant(cls, variant_name: str) -> "UncertaintyConfig":
        """Get variant-specific uncertainty config by name."""
        name = variant_name.lower().replace("-", "_").replace(".", "_")
        if name in ("wild_type", "wuhan", "d614g", "b_1"):
            return cls.wild_type()
        elif name in ("alpha", "b_1_1_7"):
            return cls.alpha()
        elif name in ("delta", "b_1_617_2"):
            return cls.alpha()
        elif name.startswith("omicron") or name.startswith("ba"):
            return cls.omicron()
        else:
            return cls.default_covid()

    @classmethod
    def calibrated_scales(cls, wave_num: int = 1) -> "UncertaintyConfig":
        """Parameter uncertainty CENTRED ON the calibrated point estimate."""
        sev_cv = 0.30 if wave_num == 3 else 0.22
        ihr_cv = 0.30 if wave_num == 3 else 0.25
        priors = {
            "R_scale": ParameterPrior(
                name="R_scale", distribution="lognormal",
                params={"mean": 1.0, "cv": 0.08}, bounds=(0.8, 1.3),
                description="Multiplicative on calibrated R(t). Tightened from "
                            "cv 0.12 / (0.6,1.6): R(t) is calibrated to "
                            "surveillance, so a 40% downward excursion is not "
                            "supported and drove implausibly wide attack-rate "
                            "bands for near-threshold waves (W1 baseline CI "
                            "spanned 1-22%). Final epidemic size is hypersensitive "
                            "to R near threshold, so this bound is the dominant "
                            "lever on infection-band width."),
            "M_scale": ParameterPrior(
                name="M_scale", distribution="lognormal",
                params={"mean": 1.0, "cv": sev_cv}, bounds=(0.5, 2.0),
                description="Multiplicative on calibrated IFR; "
                            "Meyerowitz-Katz & Merone 2021 / O'Driscoll 2021"),
            "H_scale": ParameterPrior(
                name="H_scale", distribution="lognormal",
                params={"mean": 1.0, "cv": ihr_cv}, bounds=(0.5, 2.0),
                description="Multiplicative on calibrated IHR; "
                            "Salje et al. 2020 Science 3.6% [2.1-6.2%]"),
            "npi_effectiveness_scale": ParameterPrior(
                name="npi_effectiveness_scale", distribution="lognormal",
                params={"mean": 1.0, "cv": 0.18}, bounds=(0.5, 1.5),
                description="Multiplicative on NPI effectiveness; "
                            "Brauner et al. 2021 Science effect-size ranges"),
            "workplace_share": ParameterPrior(
                name="workplace_share", distribution="uniform",
                params={"low": 0.08, "high": 0.20}, bounds=(0.08, 0.20),
                description="Share of transmission occurring in workplaces, from "
                            "Curran-Sebastian et al. 2026 (293,841 Danish genomes): "
                            "reported as a lower bound of 'over 10%', jointly for "
                            "workplaces and schools. Sampled over the declared "
                            "0.08-0.20 range rather than fixed, so no point value "
                            "has to be justified. Enters as a mean-1-preserving "
                            "reweighting of q_layers."),
            "spillover_z": ParameterPrior(
                name="spillover_z", distribution="normal",
                params={"mean": 0.0, "std": 1.0}, bounds=(-2.5, 2.5),
                description="Standardised deviate on the per-NPI displacement matrix, "
                            "applied as mean + z*std using the standard deviations "
                            "already declared in SPILLOVER_PRIORS (Jarvis et al. 2020 "
                            "CoMix). Propagates the prior's own uncertainty; does NOT "
                            "make displacement an estimate -- wave 1's schedule makes "
                            "the multiplier product constant over the wave and therefore "
                            "unidentifiable."),
        }
        return cls(mode=UncertaintyMode.PARAMETER, priors=priors)

    @classmethod
    def from_wave_config(cls, wave_config_path: str) -> "UncertaintyConfig":
        """Build variant-specific priors from a wave config YAML."""
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.to_container(OmegaConf.load(wave_config_path), resolve=True) or {}
        except ImportError:
            import yaml
            with open(wave_config_path) as f:
                cfg = yaml.safe_load(f) or {}

        variant = cfg.get("variant", {})
        variant_name = variant.get("name", "wild_type")

        uc = cls.for_variant(variant_name)

        def _override_from_ci(prior_name, config_key, dist="gamma"):
            """Replace a prior with one derived from the wave config's {mean, ci95}."""
            val = cfg.get(config_key, {})
            if not isinstance(val, dict) or "mean" not in val:
                return
            mean = val["mean"]
            ci = val.get("ci95", [mean * 0.7, mean * 1.3])
            lo, hi = ci[0], ci[1]

            if dist == "gamma" and mean > 0:
                std_approx = (hi - lo) / 3.92
                std_approx = max(std_approx, mean * 0.05)
                shape = (mean / std_approx) ** 2
                scale = std_approx ** 2 / mean
                uc.priors[prior_name] = ParameterPrior(
                    name=prior_name, distribution="gamma",
                    params={"shape": round(shape, 2), "scale": round(scale, 4)},
                    description=f"From wave config: {mean} [{lo}–{hi}]",
                    bounds=(lo * 0.8, hi * 1.2),
                )
            elif dist == "beta" and 0 < mean < 1:
                std_approx = (hi - lo) / 3.92
                std_approx = max(std_approx, 0.001)
                k = mean * (1 - mean) / (std_approx ** 2) - 1
                k = max(k, 2.0)
                a = mean * k
                b = (1 - mean) * k
                uc.priors[prior_name] = ParameterPrior(
                    name=prior_name, distribution="beta",
                    params={"a": round(a, 1), "b": round(b, 1)},
                    description=f"From wave config: {mean} [{lo}–{hi}]",
                    bounds=(max(lo * 0.8, 0.0001), min(hi * 1.2, 0.999)),
                )

        _override_from_ci("latent_period_mean", "latent_period", "gamma")
        _override_from_ci("infectious_period_mean", "infectious_period", "gamma")
        _override_from_ci("immunity_waning_mean", "immunity_waning", "gamma")
        _override_from_ci("H", "ihr", "beta")
        _override_from_ci("M", "ifr", "beta")

        return uc

    @classmethod
    def from_dict(cls, d: dict) -> "UncertaintyConfig":
        """Build from a dict (e.g., YAML or LLM-generated)."""
        mode = UncertaintyMode[d.get("mode", "PARAMETER").upper()]
        priors = {}
        for name, pdef in d.get("priors", {}).items():
            priors[name] = ParameterPrior(
                name=name,
                distribution=pdef["distribution"],
                params=pdef.get("params", {}),
                description=pdef.get("description", ""),
                bounds=tuple(pdef["bounds"]) if "bounds" in pdef else None,
            )
        structural = []
        for sdef in d.get("structural_assumptions", []):
            structural.append(StructuralAssumption(
                name=sdef["name"],
                description=sdef.get("description", ""),
                overrides=sdef.get("overrides", {}),
            ))
        return cls(mode=mode, priors=priors, structural_assumptions=structural)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        d = {"mode": self.mode.name}
        d["priors"] = {}
        for name, prior in self.priors.items():
            entry = {"distribution": prior.distribution, "params": prior.params}
            if prior.description:
                entry["description"] = prior.description
            if prior.bounds:
                entry["bounds"] = list(prior.bounds)
            d["priors"][name] = entry
        d["structural_assumptions"] = [
            {"name": s.name, "description": s.description, "overrides": s.overrides}
            for s in self.structural_assumptions
        ]
        return d


    def describe(self) -> str:
        lines = [f"UncertaintyConfig (mode={self.mode.name})"]
        lines.append(f"  {len(self.priors)} parameter priors:")
        for prior in self.priors.values():
            lines.append(f"    {prior.describe()}")
        if self.structural_assumptions:
            lines.append(f"  {len(self.structural_assumptions)} structural assumptions:")
            for sa in self.structural_assumptions:
                lines.append(f"    {sa.name}: {sa.description}")
                for k, v in sa.overrides.items():
                    lines.append(f"      {k} = {v}")
        return "\n".join(lines)


def decompose_variance(
    results: List[dict],
    samples: List[ParameterSample],
    metric: str = "total_infections",
) -> Dict[str, float]:
    """Decompose total variance into parameter and stochastic components."""
    by_sample = {}
    for r in results:
        idx = r.get("sample_idx", 0)
        by_sample.setdefault(idx, []).append(r[metric])

    all_values = [r[metric] for r in results]
    total_var = float(np.var(all_values))

    within_vars = []
    group_means = []
    for idx, vals in by_sample.items():
        if len(vals) > 1:
            within_vars.append(np.var(vals))
        group_means.append(np.mean(vals))

    stochastic_var = float(np.mean(within_vars)) if within_vars else 0.0
    param_var = float(np.var(group_means)) if len(group_means) > 1 else total_var

    return {
        "total_var": total_var,
        "param_var": param_var,
        "stochastic_var": stochastic_var,
        "param_fraction": param_var / max(total_var, 1e-12),
        "stochastic_fraction": stochastic_var / max(total_var, 1e-12),
    }


def parameter_sensitivity(
    results: List[dict],
    samples: List[ParameterSample],
    metric: str = "total_infections",
) -> Dict[str, float]:
    """Compute Spearman rank correlation between each parameter and an outcome."""
    from scipy.stats import spearmanr

    if len(results) < 5:
        return {}

    metric_vals = np.array([r[metric] for r in results])

    correlations = {}
    for param_name in (samples[0].values.keys() if samples else []):
        param_vals = np.array([s.values.get(param_name, 0) for s in samples[:len(results)]])
        if np.std(param_vals) < 1e-12:
            continue
        corr, pval = spearmanr(param_vals, metric_vals)
        correlations[param_name] = {
            "correlation": float(corr),
            "p_value": float(pval),
            "significant": pval < 0.05,
        }

    return dict(sorted(correlations.items(), key=lambda x: abs(x[1]["correlation"]), reverse=True))


def _quantile(prior: ParameterPrior, u: float) -> float:
    """Inverse CDF for a ParameterPrior at quantile u ∈ [0, 1]."""
    from scipy import stats

    if prior.distribution == "fixed":
        return prior.params.get("value", 0.0)

    if prior.distribution == "normal":
        return stats.norm.ppf(u, loc=prior.params["mean"], scale=prior.params["std"])

    if prior.distribution == "lognormal":
        mean = prior.params["mean"]
        cv = prior.params.get("cv", 0.15)
        sigma2 = np.log(1 + cv**2)
        mu = np.log(mean) - sigma2 / 2
        return stats.lognorm.ppf(u, s=np.sqrt(sigma2), scale=np.exp(mu))

    if prior.distribution == "beta":
        return stats.beta.ppf(u, prior.params["a"], prior.params["b"])

    if prior.distribution == "gamma":
        return stats.gamma.ppf(u, a=prior.params["shape"], scale=prior.params["scale"])

    if prior.distribution == "uniform":
        return prior.params["low"] + u * (prior.params["high"] - prior.params["low"])

    raise ValueError(f"No quantile function for distribution: {prior.distribution}")
