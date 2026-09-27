"""Configurable compartment graph engine for the Danish ABM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np

from .disease_params import ParamSpec


@dataclass
class Transition:
    """A single edge in the compartment graph."""
    from_state: str
    to_state: str
    mechanism: str
    duration: Optional[ParamSpec] = None
    probability: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "Transition":
        dur = None
        if "duration" in d and d["duration"] is not None:
            dur_d = d["duration"]
            dur = ParamSpec(
                mean=dur_d["mean"],
                ci95=tuple(dur_d["ci95"]) if "ci95" in dur_d else None,
            )
        return cls(
            from_state=d["from"],
            to_state=d["to"],
            mechanism=d.get("mechanism", "progression"),
            duration=dur,
            probability=d.get("probability", 1.0),
        )


class CompartmentModel:
    """Directed graph of disease compartments with configurable transitions."""

    def __init__(
        self,
        compartments: List[str],
        transitions: List[Transition],
        infectious_states: List[str],
        susceptible_states: List[str],
        removed_states: Optional[List[str]] = None,
        infectiousness_by_state: Optional[Dict[str, float]] = None,
        transmission_mode: str = "respiratory",
        network_layers: Optional[List[str]] = None,
        transmission_layers: Optional[List[str]] = None,
        seasonal_forcing: Optional[str] = None,
        generation_time: Optional[Dict] = None,
        notes: Optional[str] = None,
    ):
        self.compartment_names = list(compartments)
        self.state_to_idx = {name: i for i, name in enumerate(self.compartment_names)}
        self.idx_to_state = {i: name for i, name in enumerate(self.compartment_names)}
        self.num_states = len(self.compartment_names)

        self.transitions = list(transitions)
        self.infectious_states = list(infectious_states)
        self.susceptible_states = list(susceptible_states)
        self.removed_states = list(removed_states or [])

        self.infectiousness_by_state = {}
        for s in self.infectious_states:
            self.infectiousness_by_state[s] = 1.0
        if infectiousness_by_state:
            self.infectiousness_by_state.update(infectiousness_by_state)

        self.transmission_mode = transmission_mode
        self.network_layers = network_layers or [
            "family", "household", "neighborhood", "colleague", "classmate"
        ]
        self.transmission_layers = transmission_layers or self.network_layers
        self.seasonal_forcing = seasonal_forcing
        self.generation_time = generation_time
        self.notes = notes

        self._build_transition_graph()

    def _build_transition_graph(self):
        """Build efficient lookup tables from the transition list."""
        self._outgoing: Dict[str, List[Transition]] = {}
        for t in self.transitions:
            self._outgoing.setdefault(t.from_state, []).append(t)

        self._infection_transitions: List[Transition] = [
            t for t in self.transitions if t.mechanism == "infection"
        ]

        self._progression_transitions: List[Transition] = [
            t for t in self.transitions
            if t.mechanism in ("progression", "mortality", "waning")
        ]

        self._branching: Dict[str, List[Tuple[Transition, float]]] = {}
        for state_name in self.compartment_names:
            outgoing = [
                t for t in self._outgoing.get(state_name, [])
                if t.mechanism in ("progression", "mortality", "waning")
            ]
            if not outgoing:
                continue
            total_prob = sum(t.probability for t in outgoing)
            self._branching[state_name] = [
                (t, t.probability / total_prob) for t in outgoing
            ]


    def idx(self, state_name: str) -> int:
        return self.state_to_idx[state_name]

    def name(self, state_idx: int) -> str:
        return self.idx_to_state[state_idx]

    @property
    def infectious_indices(self) -> List[int]:
        return [self.state_to_idx[s] for s in self.infectious_states]

    @property
    def susceptible_indices(self) -> List[int]:
        return [self.state_to_idx[s] for s in self.susceptible_states]

    @property
    def removed_indices(self) -> List[int]:
        return [self.state_to_idx[s] for s in self.removed_states]


    def is_infectious(self, state_idx: int) -> bool:
        return self.idx_to_state.get(state_idx, "") in self.infectious_states

    def is_susceptible(self, state_idx: int) -> bool:
        return self.idx_to_state.get(state_idx, "") in self.susceptible_states

    def is_removed(self, state_idx: int) -> bool:
        return self.idx_to_state.get(state_idx, "") in self.removed_states

    def get_infectiousness_weight(self, state_idx: int) -> float:
        state_name = self.idx_to_state.get(state_idx, "")
        return self.infectiousness_by_state.get(state_name, 0.0)

    def infectious_mask(self, stages: torch.Tensor) -> torch.Tensor:
        """Float mask: 1.0 where agent is in an infectious state, 0.0 otherwise."""
        mask = torch.zeros_like(stages, dtype=torch.float32)
        for idx in self.infectious_indices:
            mask = mask + (stages == idx).float()
        return mask.clamp(max=1.0)

    def susceptible_mask(self, stages: torch.Tensor) -> torch.Tensor:
        """Float mask: 1.0 where agent is in a susceptible state."""
        mask = torch.zeros_like(stages, dtype=torch.float32)
        for idx in self.susceptible_indices:
            mask = mask + (stages == idx).float()
        return mask.clamp(max=1.0)

    def infectiousness_weights(self, stages: torch.Tensor) -> torch.Tensor:
        """Per-agent infectiousness weight based on current state."""
        weights = torch.zeros_like(stages, dtype=torch.float32)
        for state_name, w in self.infectiousness_by_state.items():
            idx = self.state_to_idx[state_name]
            weights = weights + (stages == idx).float() * w
        return weights


    def get_transition_times(self, n_agents: int, device=None) -> Dict[str, torch.Tensor]:
        """Sample per-agent durations for each timed transition."""
        result = {}
        for t in self._progression_transitions:
            key = f"{t.from_state}->{t.to_state}"
            if t.duration is not None:
                result[key] = t.duration.sample(n_agents, device=device)
            else:
                result[key] = torch.ones(n_agents, device=device)
        return result

    def get_next_state(self, current_state_name: str) -> Tuple[str, float]:
        """Get the next state from a deterministic (non-branching) transition."""
        branches = self._branching.get(current_state_name, [])
        if not branches:
            return current_state_name, 1.0
        if len(branches) == 1:
            return branches[0][0].to_state, 1.0
        best = max(branches, key=lambda x: x[1])
        return best[0].to_state, best[1]

    def get_branching_outcome(
        self, current_state_name: str, n_agents: int, device=None
    ) -> Tuple[List[str], torch.Tensor]:
        """For agents in current_state_name, sample which branch each takes."""
        branches = self._branching.get(current_state_name, [])
        if not branches:
            return [current_state_name], torch.zeros(n_agents, dtype=torch.long, device=device)

        to_states = [t.to_state for t, _ in branches]
        probs = torch.tensor([p for _, p in branches], device=device)

        if len(to_states) == 1:
            return to_states, torch.zeros(n_agents, dtype=torch.long, device=device)

        assignments = torch.multinomial(
            probs.unsqueeze(0).expand(n_agents, -1), 1
        ).squeeze(1)
        return to_states, assignments

    def get_infection_targets(self) -> List[Tuple[str, str, float]]:
        """Get (from_state, to_state, probability) for infection transitions."""
        results = []
        for t in self._infection_transitions:
            results.append((t.from_state, t.to_state, t.probability))
        return results

    def has_waning(self) -> bool:
        return any(t.mechanism == "waning" for t in self.transitions)

    def has_mortality(self) -> bool:
        return any(t.mechanism == "mortality" for t in self.transitions)

    def get_waning_transitions(self) -> List[Transition]:
        return [t for t in self.transitions if t.mechanism == "waning"]

    def get_mortality_transitions(self) -> List[Transition]:
        return [t for t in self.transitions if t.mechanism == "mortality"]


    @property
    def is_seirm_compatible(self) -> bool:
        """Check if this model has the standard S, E, I, R (and optionally M) states at
        the expected indices for backward compatibility.
        """
        expected = {"S": 0, "E": 1, "I": 2, "R": 3}
        for name, idx in expected.items():
            if self.state_to_idx.get(name) != idx:
                return False
        return True

    def to_seirm_vars(self) -> Dict[str, int]:
        """Return SEIRM variable mappings for backward-compatible config injection."""
        return {
            "SUSCEPTIBLE_VAR": self.idx("S") if "S" in self.state_to_idx else 0,
            "EXPOSED_VAR": self.idx("E") if "E" in self.state_to_idx else 1,
            "INFECTED_VAR": self.idx("I") if "I" in self.state_to_idx else 2,
            "RECOVERED_VAR": self.idx("R") if "R" in self.state_to_idx else 3,
            "MORTALITY_VAR": self.idx("M") if "M" in self.state_to_idx else 4,
        }


    def build_sf_infector(self) -> List[float]:
        """Build the SFInfector array: infectiousness weight per state index."""
        sf = []
        for i in range(self.num_states):
            state_name = self.idx_to_state[i]
            sf.append(self.infectiousness_by_state.get(state_name, 0.0))
        return sf


    def to_config_overrides(self) -> Dict:
        """Generate simulation_metadata overrides for this compartment model."""
        overrides = {}

        seirm = self.to_seirm_vars()
        overrides.update(seirm)

        overrides["num_network_layers"] = len(self.network_layers)
        overrides["network_layer_names"] = self.network_layers

        for t in self.transitions:
            if t.duration is None:
                continue
            if t.from_state == "E" and t.to_state == "I":
                overrides["EXPOSED_TO_INFECTED_TIME"] = round(t.duration.mean)
            elif t.from_state == "I" and t.to_state == "R":
                overrides["INFECTED_TO_RECOVERED_TIME"] = round(t.duration.mean)
            elif t.mechanism == "waning":
                overrides["RECOVERED_TO_SUSCEPTIBLE_TIME"] = round(t.duration.mean)

        if self.generation_time:
            gt = self.generation_time
            overrides.setdefault("disease_params", {})["generation_time"] = {
                "mean": gt["mean"],
            }
            if "ci95" in gt:
                overrides["disease_params"]["generation_time"]["ci95"] = gt["ci95"]

        return overrides


    @classmethod
    def from_dict(cls, d: dict) -> "CompartmentModel":
        """Build from a preset-style dict."""
        transitions = [Transition.from_dict(t) for t in d["transitions"]]
        return cls(
            compartments=d["compartments"],
            transitions=transitions,
            infectious_states=d["infectious_states"],
            susceptible_states=d["susceptible_states"],
            removed_states=d.get("removed_states", []),
            infectiousness_by_state=d.get("infectiousness_by_state"),
            transmission_mode=d.get("transmission_mode", "respiratory"),
            network_layers=d.get("network_layers"),
            transmission_layers=d.get("transmission_layers"),
            seasonal_forcing=d.get("seasonal_forcing"),
            generation_time=d.get("generation_time"),
            notes=d.get("notes"),
        )

    @classmethod
    def from_preset(cls, name: str) -> "CompartmentModel":
        """Build from a named pathogen preset."""
        from .pathogen_presets import PATHOGEN_PRESETS
        if name not in PATHOGEN_PRESETS:
            available = ", ".join(sorted(PATHOGEN_PRESETS.keys()))
            raise KeyError(f"Unknown pathogen preset '{name}'. Available: {available}")
        return cls.from_dict(PATHOGEN_PRESETS[name])

    @classmethod
    def seirm_default(cls) -> "CompartmentModel":
        """The standard SEIRM model used for COVID-19 (backward compatible)."""
        return cls.from_preset("sars_cov_2")

    def describe(self) -> str:
        lines = [f"CompartmentModel: {' -> '.join(self.compartment_names)}"]
        lines.append(f"  States ({self.num_states}): {self.compartment_names}")
        lines.append(f"  Infectious: {self.infectious_states}")
        lines.append(f"  Susceptible: {self.susceptible_states}")
        lines.append(f"  Removed: {self.removed_states}")
        lines.append(f"  Transmission mode: {self.transmission_mode}")
        lines.append(f"  Network layers: {self.network_layers}")
        lines.append(f"  Transmission layers: {self.transmission_layers}")
        lines.append(f"  Transitions ({len(self.transitions)}):")
        for t in self.transitions:
            dur_str = f" ({t.duration.mean:.0f}d)" if t.duration else ""
            prob_str = f" p={t.probability}" if t.probability < 1.0 else ""
            lines.append(f"    {t.from_state} -> {t.to_state} [{t.mechanism}]{dur_str}{prob_str}")
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)
