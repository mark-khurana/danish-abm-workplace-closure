"""Agent property registry — load, track, and stratify agent-level attributes."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


def read_tabular(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read a tabular file, auto-detecting format from extension."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path, **kwargs)
    elif ext == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    else:
        return pd.read_csv(path, **kwargs)


def find_file(directory: Union[str, Path], stem: str) -> Optional[Path]:
    """Find a file by stem, trying common extensions."""
    directory = Path(directory)
    for ext in [".parquet", ".pq", ".csv", ".tsv"]:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_tabular_files(directory: Union[str, Path]) -> List[Path]:
    """Find all tabular files in a directory (CSV/Parquet/TSV)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    exts = {".csv", ".tsv", ".parquet", ".pq"}
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)


class AgentPropertyRegistry:
    """Load and hold per-agent attributes for post-hoc outcome analysis."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self._properties: Dict[str, np.ndarray] = {}
        self._labels: Dict[str, Dict[int, str]] = {}
        self._dtypes: Dict[str, str] = {}


    def add(
        self,
        name: str,
        values: Union[np.ndarray, torch.Tensor, list],
        dtype: str = "categorical",
        labels: Optional[Dict[int, str]] = None,
    ) -> "AgentPropertyRegistry":
        """Register a property."""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values).squeeze()
        if len(values) != self.n_agents:
            raise ValueError(f"Property '{name}' has {len(values)} values, expected {self.n_agents}")
        self._properties[name] = values
        self._dtypes[name] = dtype
        if labels:
            self._labels[name] = labels
        return self

    def add_from_pickle(
        self,
        name: str,
        path: str,
        dtype: str = "categorical",
        labels: Optional[Dict[int, str]] = None,
    ) -> "AgentPropertyRegistry":
        """Load a property from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        if hasattr(data, "values"):
            arr = data.values.squeeze()
        else:
            arr = np.asarray(data).squeeze()
        return self.add(name, arr, dtype, labels)

    def add_from_csv(
        self,
        name: str,
        path: str,
        column: Optional[str] = None,
        dtype: str = "categorical",
        labels: Optional[Dict[int, str]] = None,
    ) -> "AgentPropertyRegistry":
        """Load a property from a CSV/Parquet file (one row per agent)."""
        df = read_tabular(path)
        if column:
            arr = df[column].values
        else:
            arr = df.iloc[:, 0].values
        return self.add(name, arr, dtype, labels)

    @classmethod
    def from_agents_csv(
        cls,
        csv_path: str,
        labels: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> "AgentPropertyRegistry":
        """Load all agent properties from a single file (CSV or Parquet)."""
        df = read_tabular(csv_path)

        id_col = None
        for candidate in ["agent_id", "id", "person_id", "cprnr"]:
            if candidate in df.columns:
                id_col = candidate
                break

        n_agents = len(df)
        registry = cls(n_agents)

        BUILTIN_LABELS = {
            "age": {0: "0-9", 1: "10-19", 2: "20-29", 3: "30-39",
                    4: "40-49", 5: "50-59", 6: "60-69", 7: "70+"},
            "sex": {0: "Female", 1: "Male"},
        }
        all_labels = {**BUILTIN_LABELS, **(labels or {})}

        for col in df.columns:
            if col == id_col:
                continue
            values = df[col].values

            if values.dtype.kind == "f":
                dtype = "continuous"
            else:
                try:
                    n_unique = len(np.unique(values))
                except TypeError:
                    continue
                if n_unique <= 2:
                    dtype = "binary"
                else:
                    dtype = "categorical"

            col_labels = all_labels.get(col)
            if col_labels and isinstance(col_labels, dict):
                col_labels = {int(k): v for k, v in col_labels.items()}

            registry.add(col, values, dtype, col_labels)

        return registry

    @classmethod
    def from_population_dir(
        cls,
        pop_dir: str,
        n_agents: Optional[int] = None,
        extra_pickles: Optional[Dict[str, dict]] = None,
    ) -> "AgentPropertyRegistry":
        """Auto-discover and load agent properties from a population directory."""
        pop_dir = str(pop_dir)

        agents_file = find_file(pop_dir, "agents")
        if agents_file is not None:
            csv_path = str(agents_file)
            yaml_labels = None
            yaml_path = os.path.join(pop_dir, "agents.yaml")
            if os.path.exists(yaml_path):
                try:
                    from omegaconf import OmegaConf
                    meta = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True) or {}
                    yaml_labels = {k: v.get("labels") for k, v in meta.items()
                                   if isinstance(v, dict) and "labels" in v}
                except Exception:
                    pass
            registry = cls.from_agents_csv(csv_path, labels=yaml_labels)
            if n_agents is not None and registry.n_agents != n_agents:
                raise ValueError(
                    f"agents.csv has {registry.n_agents} rows but model expects {n_agents} agents"
                )
            return registry


        SKIP_FILES = {
            "disease_stages", "ground_truth", "intervention_schedule",
            "contact_matrix", "transmission_matrices", "vaccination_schedule",
        }

        metadata: Dict[str, dict] = {}
        yaml_path = os.path.join(pop_dir, "agents.yaml")
        if os.path.exists(yaml_path):
            try:
                import yaml
                with open(yaml_path) as f:
                    metadata = yaml.safe_load(f) or {}
            except ImportError:
                try:
                    from omegaconf import OmegaConf
                    metadata = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True) or {}
                except Exception:
                    pass

        BUILTIN_LABELS = {
            "age": {0: "0-9", 1: "10-19", 2: "20-29", 3: "30-39",
                    4: "40-49", 5: "50-59", 6: "60-69", 7: "70+"},
            "sex": {0: "Female", 1: "Male"},
        }

        if n_agents is None:
            for fname in sorted(os.listdir(pop_dir)):
                if fname.endswith(".pickle"):
                    stem = fname.replace(".pickle", "")
                    if stem in SKIP_FILES:
                        continue
                    with open(os.path.join(pop_dir, fname), "rb") as f:
                        data = pickle.load(f)
                    arr = data.values.squeeze() if hasattr(data, "values") else np.asarray(data).squeeze()
                    n_agents = len(arr)
                    break

        if n_agents is None:
            raise ValueError("Cannot determine n_agents — no .pickle files found in " + pop_dir)

        registry = cls(n_agents)

        for fname in sorted(os.listdir(pop_dir)):
            if not fname.endswith(".pickle"):
                continue
            stem = fname.replace(".pickle", "")
            if stem in SKIP_FILES:
                continue

            path = os.path.join(pop_dir, fname)
            prop_meta = metadata.get(stem, {})
            labels = prop_meta.get("labels", BUILTIN_LABELS.get(stem))
            dtype = prop_meta.get("dtype", "categorical")

            if labels and isinstance(labels, dict):
                labels = {int(k): v for k, v in labels.items()}

            try:
                registry.add_from_pickle(stem, path, dtype, labels)
            except (ValueError, Exception):
                pass

        if extra_pickles:
            for name, spec in extra_pickles.items():
                if name in registry:
                    continue
                path = spec.get("path", os.path.join(pop_dir, f"{name}.pickle"))
                if os.path.exists(path):
                    labels = spec.get("labels")
                    if labels and isinstance(labels, dict):
                        labels = {int(k): v for k, v in labels.items()}
                    try:
                        registry.add_from_pickle(name, path, spec.get("dtype", "categorical"), labels)
                    except (ValueError, Exception):
                        pass

        return registry


    def get(self, name: str) -> np.ndarray:
        """Get a property array by name."""
        return self._properties[name]

    def get_labeled(self, name: str) -> np.ndarray:
        """Get property values mapped to human-readable labels (string array)."""
        values = self._properties[name]
        labels = self._labels.get(name)
        if labels:
            return np.array([labels.get(int(v), str(int(v))) if pd.notna(v) else "Unknown"
                             for v in values])
        return values.astype(str)

    @property
    def property_names(self) -> List[str]:
        return list(self._properties.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._properties

    def to_dataframe(self) -> pd.DataFrame:
        """Export all properties as a DataFrame (one row per agent)."""
        df = pd.DataFrame({"agent_id": np.arange(self.n_agents)})
        for name, values in self._properties.items():
            labels = self._labels.get(name)
            if labels:
                df[name] = [labels.get(int(v), str(int(v))) if pd.notna(v) else "Unknown"
                            for v in values]
            else:
                df[name] = values
        return df


@dataclass
class AgentEventLog:
    """Per-agent record of epidemic events."""

    n_agents: int

    infection_day: Optional[np.ndarray] = None
    recovery_day: Optional[np.ndarray] = None
    death_day: Optional[np.ndarray] = None
    hospitalization_day: Optional[np.ndarray] = None

    vaccination_day: Optional[np.ndarray] = None
    vaccination_dose: Optional[np.ndarray] = None

    first_positive_day: Optional[np.ndarray] = None
    total_tests: Optional[np.ndarray] = None

    n_infections: Optional[np.ndarray] = None

    def __post_init__(self):
        N = self.n_agents
        self.infection_day = np.full(N, -1, dtype=np.int32)
        self.recovery_day = np.full(N, -1, dtype=np.int32)
        self.death_day = np.full(N, -1, dtype=np.int32)
        self.hospitalization_day = np.full(N, -1, dtype=np.int32)
        self.vaccination_day = np.full(N, -1, dtype=np.int32)
        self.vaccination_dose = np.zeros(N, dtype=np.int32)
        self.first_positive_day = np.full(N, -1, dtype=np.int32)
        self.total_tests = np.zeros(N, dtype=np.int32)
        self.n_infections = np.zeros(N, dtype=np.int32)

    @classmethod
    def from_runner(cls, runner, n_days: int) -> "AgentEventLog":
        """Extract event log from a completed simulation runner."""
        n_agents = runner.state["agents"]["citizens"]["disease_stage"].shape[0]
        log = cls(n_agents)

        disease_stage = runner.state["agents"]["citizens"]["disease_stage"].detach().cpu().numpy().squeeze()
        infected_time = runner.state["agents"]["citizens"]["infected_time"].detach().cpu().numpy().squeeze()

        SUSCEPTIBLE = 0
        EXPOSED = 1
        INFECTED = 2
        RECOVERED = 3
        MORTALITY = 4

        was_infected = infected_time < n_days + 10
        log.infection_day[was_infected] = infected_time[was_infected].astype(np.int32)
        log.n_infections[was_infected] = 1

        is_dead = (disease_stage == MORTALITY) & was_infected
        log.death_day[is_dead] = infected_time[is_dead].astype(np.int32) + 10

        is_recovered = disease_stage == RECOVERED
        log.recovery_day[is_recovered] = infected_time[is_recovered].astype(np.int32) + 5

        for name, mod in runner.named_modules():
            cls_name = type(mod).__name__

            if cls_name == "NewTransmission":
                tracking = mod.tracking
                if hasattr(mod, "_trajectory") and mod._trajectory is not None:
                    traj = mod._trajectory.cpu().numpy()
                    for agent_id in range(n_agents):
                        stages = traj[:, agent_id]
                        exposed_days = np.where(stages == EXPOSED)[0]
                        if len(exposed_days) > 0:
                            log.infection_day[agent_id] = int(exposed_days[0])
                        infected_days = np.where(stages == INFECTED)[0]
                        recovered_days = np.where(stages == RECOVERED)[0]
                        if len(recovered_days) > 0:
                            log.recovery_day[agent_id] = int(recovered_days[0])
                        dead_days = np.where(stages == MORTALITY)[0]
                        if len(dead_days) > 0 and log.infection_day[agent_id] >= 0:
                            log.death_day[agent_id] = int(dead_days[0])

            elif cls_name == "VaccinationRollout":
                vax_tracking = getattr(mod, "_agent_vaccinated_day", None)
                if vax_tracking is None:
                    vax_tracking = getattr(mod, "_last_dose_day", None)
                if vax_tracking is not None:
                    vax_arr = vax_tracking.cpu().numpy().squeeze()
                    vaccinated = (vax_arr >= 0) & (vax_arr < n_days + 10)
                    log.vaccination_day[vaccinated] = vax_arr[vaccinated].astype(np.int32)
                vax_doses = getattr(mod, "_agent_doses", None)
                if vax_doses is None:
                    vax_doses = getattr(mod, "_dose_count", None)
                if vax_doses is not None:
                    log.vaccination_dose = vax_doses.cpu().numpy().squeeze().astype(np.int32)

            elif cls_name == "TestingAndIsolation":
                detected = getattr(mod, "_is_detected", None)
                if detected is not None:
                    det_arr = detected.cpu().numpy().squeeze()
                    log.first_positive_day[det_arr] = 0

        return log


    @property
    def was_infected(self) -> np.ndarray:
        """(N,) bool — agent was infected at any point."""
        return self.infection_day >= 0

    @property
    def died(self) -> np.ndarray:
        """(N,) bool — agent died."""
        return self.death_day >= 0

    @property
    def was_hospitalized(self) -> np.ndarray:
        return self.hospitalization_day >= 0

    @property
    def was_vaccinated(self) -> np.ndarray:
        return self.vaccination_day >= 0

    @property
    def was_detected(self) -> np.ndarray:
        return self.first_positive_day >= 0

    def to_dataframe(self) -> pd.DataFrame:
        """Export event log as DataFrame."""
        return pd.DataFrame({
            "agent_id": np.arange(self.n_agents),
            "infected": self.was_infected,
            "infection_day": self.infection_day,
            "recovered": self.recovery_day >= 0,
            "recovery_day": self.recovery_day,
            "died": self.died,
            "death_day": self.death_day,
            "hospitalized": self.was_hospitalized,
            "hospitalization_day": self.hospitalization_day,
            "vaccinated": self.was_vaccinated,
            "vaccination_day": self.vaccination_day,
            "vaccination_doses": self.vaccination_dose,
            "detected": self.was_detected,
            "n_infections": self.n_infections,
            "total_tests": self.total_tests,
        })


class OutcomeAnalysis:
    """Stratified outcome analysis by agent properties."""

    def __init__(
        self,
        registry: AgentPropertyRegistry,
        event_log: AgentEventLog,
        scale_factor: float = 1.0,
        scenario_name: str = "",
    ):
        self.registry = registry
        self.log = event_log
        self.scale_factor = scale_factor
        self.scenario_name = scenario_name

        self._df = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        """Merge properties + outcomes into a single DataFrame."""
        props = self.registry.to_dataframe()
        events = self.log.to_dataframe()
        return props.merge(events, on="agent_id")

    @property
    def df(self) -> pd.DataFrame:
        """The full agent-level DataFrame."""
        return self._df


    def stratify_by(
        self,
        by: Union[str, List[str]],
        outcomes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Stratified outcome summary."""
        if isinstance(by, str):
            by = [by]

        if outcomes is None:
            outcomes = ["infected", "died", "hospitalized", "vaccinated", "detected"]

        df = self._df

        from .substeps.utils import COLUMN_ALIASES
        resolved = []
        for name in by:
            if name in df.columns:
                resolved.append(name)
                continue
            hit = next((c for c in COLUMN_ALIASES.get(name, []) if c in df.columns), None)
            if hit is None:
                raise KeyError(
                    f"cannot stratify by {name!r}: neither it nor any of its known "
                    f"aliases {COLUMN_ALIASES.get(name, [])} is present. Available "
                    f"columns: {sorted(df.columns)}")
            resolved.append(hit)
        rename = {r: n for r, n in zip(resolved, by) if r != n}
        by = resolved
        sf = self.scale_factor

        grouped = df.groupby(by[0] if len(by) == 1 else by)

        rows = []
        for group_key, group_df in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            n = len(group_df)
            row = {rename.get(k, k): v for k, v in zip(by, group_key)}
            row["n_agents"] = n
            row["n_real"] = int(n / sf) if sf > 0 else n

            if "infected" in outcomes:
                n_inf = group_df["infected"].sum()
                row["infections"] = int(n_inf)
                row["infections_real"] = int(n_inf / sf) if sf > 0 else int(n_inf)
                row["attack_rate"] = n_inf / max(n, 1)

            if "died" in outcomes:
                n_dead = group_df["died"].sum()
                row["deaths"] = int(n_dead)
                row["deaths_real"] = int(n_dead / sf) if sf > 0 else int(n_dead)
                row["mortality_rate"] = n_dead / max(n, 1)
                row["cfr"] = n_dead / max(group_df["infected"].sum(), 1)

            if "hospitalized" in outcomes:
                n_hosp = group_df["hospitalized"].sum()
                row["hospitalizations"] = int(n_hosp)
                row["hospitalizations_real"] = int(n_hosp / sf) if sf > 0 else int(n_hosp)
                row["hosp_rate"] = n_hosp / max(n, 1)

            if "vaccinated" in outcomes:
                n_vax = group_df["vaccinated"].sum()
                row["vaccinated"] = int(n_vax)
                row["vax_coverage"] = n_vax / max(n, 1)

            if "detected" in outcomes:
                n_det = group_df["detected"].sum()
                row["detected"] = int(n_det)
                n_inf = group_df["infected"].sum()
                row["detection_rate"] = n_det / max(n_inf, 1)

            if "infection_day" in df.columns:
                inf_days = group_df.loc[group_df["infected"], "infection_day"]
                if len(inf_days) > 0:
                    row["mean_infection_day"] = float(inf_days.mean())
                    row["median_infection_day"] = float(inf_days.median())

            rows.append(row)

        result = pd.DataFrame(rows)
        return result

    def summary(self) -> pd.DataFrame:
        """Full summary: stratify by every registered property individually."""
        frames = {}
        for prop in self.registry.property_names:
            frames[prop] = self.stratify_by(prop)
        return frames

    def equity_metrics(self, by: str = "ses") -> pd.DataFrame:
        """Compute equity/disparity metrics for a stratifying variable."""
        df = self.stratify_by(by)
        if "attack_rate" not in df.columns:
            return df

        ref_idx = df["attack_rate"].idxmin()
        ref_rate = df.loc[ref_idx, "attack_rate"]
        ref_cfr = df.loc[ref_idx, "cfr"] if "cfr" in df.columns else 0

        df["rr_infection"] = df["attack_rate"] / max(ref_rate, 1e-9)
        if "cfr" in df.columns:
            df["rr_cfr"] = df["cfr"] / max(ref_cfr, 1e-9)
        if "mortality_rate" in df.columns:
            ref_mort = df.loc[ref_idx, "mortality_rate"]
            df["rr_mortality"] = df["mortality_rate"] / max(ref_mort, 1e-9)

        return df

    def compare_scenarios(
        self,
        other: "OutcomeAnalysis",
        by: Union[str, List[str]] = "age",
    ) -> pd.DataFrame:
        """Compare stratified outcomes between two scenarios."""
        df_self = self.stratify_by(by)
        df_other = other.stratify_by(by)

        if isinstance(by, str):
            by = [by]

        merged = df_self.merge(
            df_other, on=by, suffixes=(f"_{self.scenario_name}", f"_{other.scenario_name}"),
        )

        sn = self.scenario_name
        on = other.scenario_name
        for metric in ["infections_real", "deaths_real", "hospitalizations_real"]:
            col_s = f"{metric}_{sn}"
            col_o = f"{metric}_{on}"
            if col_s in merged.columns and col_o in merged.columns:
                merged[f"delta_{metric}"] = merged[col_s] - merged[col_o]

        return merged


    def to_csv(self, path: str, by: Optional[Union[str, List[str]]] = None):
        """Save stratified results to CSV."""
        if by:
            df = self.stratify_by(by)
        else:
            df = self._df
        df.to_csv(path, index=False)

    def summary_text(self, by: str = "age") -> str:
        """Human-readable summary string."""
        df = self.stratify_by(by)
        lines = [f"Outcome Analysis: {self.scenario_name or 'unnamed'}", "=" * 60]
        lines.append(f"Stratified by: {by}")
        lines.append(f"Total agents: {self.log.n_agents:,} "
                      f"(representing {int(self.log.n_agents / max(self.scale_factor, 1e-9)):,})")
        lines.append(f"Total infected: {self.log.was_infected.sum():,}")
        lines.append(f"Total deaths: {self.log.died.sum():,}")
        lines.append("")

        for _, row in df.iterrows():
            group = row[by]
            lines.append(f"  {by}={group} (n={row['n_agents']:,}):")
            lines.append(f"    Attack rate:  {row.get('attack_rate', 0):.1%}")
            lines.append(f"    Deaths:       {row.get('deaths_real', 0):,} "
                          f"(CFR={row.get('cfr', 0):.2%})")
            if "hospitalizations_real" in row:
                lines.append(f"    Hosp:         {row['hospitalizations_real']:,}")
            if "vax_coverage" in row:
                lines.append(f"    Vax coverage: {row['vax_coverage']:.1%}")

        return "\n".join(lines)


def prepare_population_dir(
    agents_csv: str,
    edges_csv: str,
    output_dir: str,
    events_dir: Optional[str] = None,
):
    """Convert simple input files into the model's population directory."""
    import shutil

    os.makedirs(output_dir, exist_ok=True)

    df = read_tabular(agents_csv)
    n_agents = len(df)
    print(f"Agents: {n_agents:,}")
    print(f"Properties: {[c for c in df.columns if c != 'agent_id']}")

    id_col = None
    for candidate in ["agent_id", "id", "person_id", "cprnr"]:
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col:
        df = df.sort_values(id_col).reset_index(drop=True)

    for col in df.columns:
        if col == id_col:
            continue
        values = df[col].values
        pickle_path = os.path.join(output_dir, f"{col}.pickle")
        with open(pickle_path, "wb") as f:
            pickle.dump(values, f)
        print(f"  {col}.pickle ({len(np.unique(values))} unique values)")

    shutil.copy2(agents_csv, os.path.join(output_dir, "agents.csv"))

    stages = np.zeros((n_agents, 1), dtype=np.int32)
    stages_path = os.path.join(output_dir, "disease_stages.csv")
    np.savetxt(stages_path, stages, fmt="%d")
    print(f"  disease_stages.csv (all susceptible)")

    net_dir = os.path.join(output_dir, "mobility_networks")
    os.makedirs(net_dir, exist_ok=True)
    shutil.copy2(edges_csv, os.path.join(net_dir, "0.csv"))
    edge_df = read_tabular(edges_csv)
    print(f"  mobility_networks/0.csv ({len(edge_df):,} edges)")

    if events_dir and os.path.isdir(events_dir):
        for fname in os.listdir(events_dir):
            src = os.path.join(events_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(output_dir, fname))
                print(f"  {fname} (copied from events/)")

    if "municipality" in df.columns:
        codes = sorted(df["municipality"].unique().tolist())
        mapping = {str(c): str(c) for c in codes}
        import json
        with open(os.path.join(output_dir, "mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"  mapping.json ({len(codes)} municipalities)")

    print(f"\nPopulation directory ready: {output_dir}/")
    print(f"Point your config to: population_dir: {output_dir}")
