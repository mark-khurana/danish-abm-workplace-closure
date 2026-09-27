"""Multi-strain support for the Danish ABM."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np


@dataclass
class StrainDef:
    """A single pathogen strain / variant / lineage."""
    name: str
    transmissibility: float = 1.0
    severity: float = 1.0

    def describe(self) -> str:
        parts = [self.name]
        if self.transmissibility != 1.0:
            parts.append(f"R×{self.transmissibility:.2f}")
        if self.severity != 1.0:
            parts.append(f"sev×{self.severity:.2f}")
        return " | ".join(parts)


@dataclass
class StrainConfig:
    """Multi-strain configuration."""
    strains: List[StrainDef] = field(default_factory=list)
    cross_immunity: Optional[torch.Tensor] = None
    introduction_days: Dict[str, int] = field(default_factory=dict)
    initial_proportions: Dict[str, float] = field(default_factory=dict)
    vaccine_antigen: Optional[str] = None
    vaccine_ve: Dict[str, float] = field(default_factory=dict)
    vaccine_ve_reference: Optional[str] = None

    def __post_init__(self):
        self.strain_names = [s.name for s in self.strains]
        self.strain_idx = {s.name: i for i, s in enumerate(self.strains)}
        self.n_strains = len(self.strains)

        if self.cross_immunity is None and self.n_strains > 0:
            self.cross_immunity = torch.eye(self.n_strains)

        self._validate_vaccine_antigen()

    def _validate_vaccine_antigen(self):
        """Fail loudly if the vaccine antigen is declared but not usable."""
        if self.vaccine_antigen is None:
            if self.vaccine_ve:
                raise ValueError(
                    "vaccine_ve was given without vaccine_antigen; the VE values "
                    "would never reach the immunity tracker."
                )
            return
        if self.vaccine_antigen in self.strain_idx:
            raise ValueError(
                f"vaccine_antigen {self.vaccine_antigen!r} collides with a strain "
                f"name; antigen names must be unique."
            )
        missing = [n for n in self.strain_names if n not in self.vaccine_ve]
        if missing:
            raise ValueError(
                f"vaccine_ve is missing peak VE for strains {missing}. Every strain "
                f"needs a cited value, otherwise the vaccine silently offers no "
                f"protection against those strains."
            )
        if self.vaccine_ve_reference is None:
            raise ValueError(
                "vaccine_ve_reference must name the strain the config's cited "
                "`vaccination.efficacy` refers to, so a perturbed efficacy scales "
                "the whole VE row coherently."
            )
        if self.vaccine_ve_reference not in self.vaccine_ve:
            raise ValueError(
                f"vaccine_ve_reference {self.vaccine_ve_reference!r} is not in "
                f"vaccine_ve (have {sorted(self.vaccine_ve)})."
            )
        for name, ve in self.vaccine_ve.items():
            if not 0.0 <= float(ve) <= 1.0:
                raise ValueError(f"vaccine_ve[{name!r}] = {ve} is not a VE in [0, 1].")

    @property
    def antigen_names(self) -> List[str]:
        """Tracker antigen order: strains first, then the vaccine antigen."""
        names = list(self.strain_names)
        if self.vaccine_antigen is not None:
            names.append(self.vaccine_antigen)
        return names

    def resolve_vaccine_ve(self, config_efficacy: Optional[float] = None):
        """Resolve the vaccine antigen into (peak_efficacy, relative_row)."""
        if self.vaccine_antigen is None:
            return None, None

        scale = 1.0
        if config_efficacy is not None:
            ref_ve = float(self.vaccine_ve[self.vaccine_ve_reference])
            if ref_ve <= 0:
                raise ValueError(
                    f"vaccine_ve[{self.vaccine_ve_reference!r}] must be > 0 to scale "
                    f"the config efficacy against it."
                )
            scale = float(config_efficacy) / ref_ve

        absolute = [min(1.0, float(self.vaccine_ve[n]) * scale) for n in self.strain_names]
        peak = max(absolute)
        if peak <= 0:
            return 0.0, torch.zeros(self.n_strains)
        row = torch.tensor([a / peak for a in absolute], dtype=torch.float32)
        return peak, row

    def antigen_cross_immunity(self, config_efficacy: Optional[float] = None) -> torch.Tensor:
        """Full (n_antigens, n_strains) cross-immunity, vaccine row appended."""
        ci = self.cross_immunity
        if ci is None:
            ci = torch.eye(self.n_strains)
        ci = ci.float()
        if ci.shape != (self.n_strains, self.n_strains):
            raise ValueError(
                f"cross_immunity must be ({self.n_strains}, {self.n_strains}), "
                f"got {tuple(ci.shape)}."
            )
        if self.vaccine_antigen is None:
            return ci
        _, row = self.resolve_vaccine_ve(config_efficacy)
        return torch.cat([ci, row.to(ci.dtype).unsqueeze(0)], dim=0)

    def unreachable_strains(self) -> List[str]:
        """Strains that can never appear in a run of this config."""
        return [s.name for s in self.strains
                if not self.initial_proportions.get(s.name, 0.0)]

    def get_strain(self, name: str) -> StrainDef:
        return self.strains[self.strain_idx[name]]

    def is_active(self, strain_name: str, day: int) -> bool:
        """Check if a strain has been introduced by this day."""
        intro_day = self.introduction_days.get(strain_name, 0)
        return day >= intro_day

    def active_strains(self, day: int) -> List[StrainDef]:
        return [s for s in self.strains if self.is_active(s.name, day)]

    def describe(self) -> str:
        lines = [f"StrainConfig: {self.n_strains} strains"]
        for s in self.strains:
            intro = self.introduction_days.get(s.name, 0)
            lines.append(f"  {s.describe()} (intro day {intro})")
        if self.cross_immunity is not None:
            lines.append(f"  Cross-immunity matrix: {self.cross_immunity.shape}")
        return "\n".join(lines)


    VARIANT_PRESETS = {
        "wild_type": "covid_2020",
        "wildtype": "covid_2020",
        "alpha": "covid_2021",
        "delta": "covid_2021",
        "omicron_ba1": "covid_2022",
        "omicron_ba2": "covid_2022",
        "ba1": "covid_2022",
        "ba2": "covid_2022",
    }

    @classmethod
    def preset_for_variant(cls, variant_name, default="covid_2021"):
        """Preset name for a wave's `variant.name`, or `default` if unrecognised."""
        if not variant_name:
            return default
        key = str(variant_name).strip().lower()
        preset = cls.VARIANT_PRESETS.get(key)
        if preset is None:
            print(f"  [strains] variant {variant_name!r} is not in VARIANT_PRESETS; "
                  f"falling back to preset {default!r}. Add it to "
                  f"StrainConfig.VARIANT_PRESETS if this wave needs its own strains.",
                  flush=True)
            return default
        return preset

    @classmethod
    def single_strain(cls) -> "StrainConfig":
        """Default single-strain config (backward compatible)."""
        return cls(
            strains=[StrainDef("default", transmissibility=1.0)],
            cross_immunity=torch.tensor([[1.0]]),
        )

    @classmethod
    def covid_2020(cls) -> "StrainConfig":
        """SARS-CoV-2 in Denmark, the 2020-12-16 to 2021-02-28 window: wildtype."""
        return cls(
            strains=[
                StrainDef("wildtype", transmissibility=1.0, severity=1.0),
                StrainDef("alpha", transmissibility=1.5, severity=1.4),
            ],
            cross_immunity=torch.tensor([
                [1.0,   0.85],
                [0.85,  1.0],
            ]),
            introduction_days={"wildtype": 0, "alpha": 0},
            initial_proportions={"wildtype": 1.0},
            vaccine_antigen="vaccine",
            vaccine_ve={"wildtype": 0.95, "alpha": 0.937},
            vaccine_ve_reference="wildtype",
        )

    @classmethod
    def covid_2021(cls) -> "StrainConfig":
        """SARS-CoV-2 in Denmark 2021: Wildtype → Alpha → Delta."""
        return cls(
            strains=[
                StrainDef("wildtype", transmissibility=1.0, severity=1.0),
                StrainDef("alpha", transmissibility=1.5, severity=1.4),
                StrainDef("delta", transmissibility=2.2, severity=1.6),
            ],
            cross_immunity=torch.tensor([
                [1.0,   0.85,  0.60],
                [0.85,  1.0,   0.70],
                [0.60,  0.70,  1.0],
            ]),
            introduction_days={
                "wildtype": 0,
                "alpha": 0,
                "delta": 150,
            },
            initial_proportions={"wildtype": 0.2, "alpha": 0.8},
            vaccine_antigen="vaccine",
            vaccine_ve={"wildtype": 0.95, "alpha": 0.937, "delta": 0.88},
            vaccine_ve_reference="alpha",
        )

    @classmethod
    def covid_2022(cls) -> "StrainConfig":
        """SARS-CoV-2 in Denmark 2022: Delta → Omicron BA.1 → BA.2 → BA.5."""
        return cls(
            strains=[
                StrainDef("delta", transmissibility=2.2, severity=1.6),
                StrainDef("ba1", transmissibility=3.5, severity=0.5),
                StrainDef("ba2", transmissibility=4.0, severity=0.4),
                StrainDef("ba5", transmissibility=4.2, severity=0.3),
            ],
            cross_immunity=torch.tensor([
                [1.0,   0.40, 0.35, 0.30],
                [0.40,  1.0,  0.85, 0.50],
                [0.35,  0.85, 1.0,  0.60],
                [0.30,  0.50, 0.60, 1.0],
            ]),
            introduction_days={"delta": 0, "ba1": 0, "ba2": 45, "ba5": 180},
            initial_proportions={"delta": 0.97, "ba1": 0.03},
            vaccine_antigen="vaccine",
            vaccine_ve={"delta": 0.93, "ba1": 0.672, "ba2": 0.662, "ba5": 0.60},
            vaccine_ve_reference="ba1",
        )

    @classmethod
    def influenza_season(cls) -> "StrainConfig":
        """Typical influenza season with A/H1N1, A/H3N2, B co-circulating."""
        return cls(
            strains=[
                StrainDef("A_H1N1", transmissibility=1.0, severity=1.0),
                StrainDef("A_H3N2", transmissibility=1.2, severity=1.3),
                StrainDef("B", transmissibility=0.8, severity=0.7),
            ],
            cross_immunity=torch.tensor([
                [1.0,   0.4,  0.05],
                [0.4,   1.0,  0.05],
                [0.05,  0.05, 1.0],
            ]),
            introduction_days={"A_H1N1": 0, "A_H3N2": 0, "B": 30},
        )


class StrainManager:
    """Tracks per-agent strain identity during simulation."""

    def __init__(self, config: StrainConfig, n_agents: int,
                 device: torch.device = None):
        self.config = config
        self.n_agents = n_agents
        self.device = device or torch.device("cpu")

        self.current_strain = torch.full(
            (n_agents,), -1, dtype=torch.long, device=self.device
        )

        self.daily_by_strain: Dict[str, List[float]] = {
            s.name: [] for s in config.strains
        }
        self._crossed_50: set = set()
        self._seeded_composition = None

    def assign_initial_strains(self, infected_mask: torch.Tensor):
        """Assign strains to initially infected agents based on initial_proportions."""
        infected_idx = infected_mask.squeeze().nonzero(as_tuple=True)[0]
        n_infected = len(infected_idx)
        if n_infected == 0:
            return

        probs = []
        for s in self.config.strains:
            p = self.config.initial_proportions.get(s.name, 0.0)
            probs.append(p)
        probs = torch.tensor(probs, dtype=torch.float32, device=self.device)
        if probs.sum() == 0:
            probs[0] = 1.0
        probs = probs / probs.sum()

        strain_assignments = torch.multinomial(
            probs.unsqueeze(0).expand(n_infected, -1), 1
        ).squeeze(-1)
        self.current_strain[infected_idx] = strain_assignments

    def get_infector_strains(self, infector_indices: torch.Tensor) -> torch.Tensor:
        """Look up which strain each infector carries."""
        return self.current_strain[infector_indices]

    def assign_new_infections_network(self, newly_exposed: torch.Tensor,
                                       infector_strains: torch.Tensor):
        """Assign strains to newly exposed agents based on their infectors."""
        exposed_idx = newly_exposed.squeeze().nonzero(as_tuple=True)[0]
        if len(exposed_idx) == 0:
            return
        self.current_strain[exposed_idx] = infector_strains[:len(exposed_idx)].clamp(min=0)

    def assign_new_infections_community(self, newly_exposed: torch.Tensor,
                                         day: int):
        """Assign strains for community-acquired infections."""
        exposed_idx = newly_exposed.squeeze().nonzero(as_tuple=True)[0]
        if len(exposed_idx) == 0:
            return

        weights = []
        for s in self.config.strains:
            if not self.config.is_active(s.name, day):
                weights.append(0.0)
                continue
            idx = self.config.strain_idx[s.name]
            n_infected = (self.current_strain == idx).sum().float().item()
            weights.append(n_infected * s.transmissibility)

        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        if weights.sum() == 0:
            for _i, _s in enumerate(self.config.strains):
                if self.config.is_active(_s.name, day):
                    weights[_i] = self.config.initial_proportions.get(_s.name, 0.0)
            if weights.sum() == 0:
                weights[0] = 1.0
            self._seeded_composition = {
                _s.name: round(float(weights[_i]) / float(weights.sum()), 4)
                for _i, _s in enumerate(self.config.strains) if weights[_i] > 0
            }
            print(f"  [strain seed] day {day}: {len(exposed_idx):,} first infections "
                  f"drawn from {self._seeded_composition}", flush=True)
        weights = weights / weights.sum()

        strain_assignments = torch.multinomial(
            weights.unsqueeze(0).expand(len(exposed_idx), -1), 1
        ).squeeze(-1)
        self.current_strain[exposed_idx] = strain_assignments

    def clear_recovered(self, recovered_mask: torch.Tensor):
        """Reset strain for agents who have recovered or died."""
        mask = recovered_mask.bool().squeeze()
        self.current_strain[mask] = -1

    def get_transmissibility_modifier(self) -> torch.Tensor:
        """Get per-agent transmissibility modifier based on their strain."""
        modifier = torch.ones(self.n_agents, device=self.device)
        for s in self.config.strains:
            idx = self.config.strain_idx[s.name]
            mask = self.current_strain == idx
            modifier[mask] = s.transmissibility
        return modifier

    def get_severity_modifier(self) -> torch.Tensor:
        """Get per-agent severity modifier based on their strain."""
        modifier = torch.ones(self.n_agents, device=self.device)
        for s in self.config.strains:
            idx = self.config.strain_idx[s.name]
            mask = self.current_strain == idx
            modifier[mask] = s.severity
        return modifier

    def track_daily(self, day: int):
        """Record daily infection counts by strain, and report takeovers."""
        counts = {}
        for s in self.config.strains:
            idx = self.config.strain_idx[s.name]
            count = (self.current_strain == idx).sum().item()
            self.daily_by_strain[s.name].append(count)
            counts[s.name] = count

        total = sum(counts.values())
        if total <= 0:
            return
        for name, count in counts.items():
            if count / total > 0.5 and name not in self._crossed_50:
                self._crossed_50.add(name)
                print(f"  [strain takeover] {name} crossed 50% of prevalence on "
                      f"day {day} ({count:,}/{total:,})", flush=True)

    def describe(self) -> str:
        lines = [f"StrainManager: {self.n_agents} agents"]
        total_infected = (self.current_strain >= 0).sum().item()
        lines.append(f"  Total infected: {total_infected}")
        for s in self.config.strains:
            idx = self.config.strain_idx[s.name]
            n = (self.current_strain == idx).sum().item()
            lines.append(f"  {s.name}: {n} ({n/max(total_infected,1)*100:.1f}%)")
        return "\n".join(lines)
