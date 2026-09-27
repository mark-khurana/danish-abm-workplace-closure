"""Unified data configuration loader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf


def scalar_mean(val):
    """A NUMBER, or None. Never a dict, never a string."""
    m = val.get("mean", val) if isinstance(val, dict) else val
    return (float(m) if isinstance(m, (int, float)) and not isinstance(m, bool)
            else None)


def _find_repo_root(start: str | Path | None = None) -> Path:
    """Walk up from *start* until we find the directory containing danish_abm/."""
    p = Path(start or os.getcwd()).resolve()
    for candidate in [p] + list(p.parents):
        if (candidate / "danish_abm" / "__init__.py").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate repository root (directory containing danish_abm/). "
        "Either run from the repo root or pass repo_root= explicitly."
    )


def ensure_vaccination_schedule(vax_csv, population_dir, start_date, n_days, n_agents):
    """<population_dir>/vaccination_schedule.pt, built from the register file if absent."""
    pop_dir = Path(population_dir)
    sched = pop_dir / "vaccination_schedule.pt"
    if not sched.exists():
        from .registry_data import build_vaccination_schedule_from_csv
        agents_file = pop_dir / "agents.parquet"
        print(f"  [vax] building the register schedule {sched} from {vax_csv}", flush=True)
        build_vaccination_schedule_from_csv(
            vax_csv, sched, start_date, n_days, n_agents,
            agents_file=agents_file if agents_file.exists() else None)
    return sched


@dataclass
class DataConfig:
    """Resolved, absolute-path data configuration."""

    repo_root: Path
    population_dir: Path
    network_file: Path
    ground_truth_dir: Path
    num_agents: int

    real_population: int

    start_date: str
    end_date: str
    start_doy: int

    precomputed: dict = field(default_factory=dict)

    individual_test_csv: Optional[Path] = None
    individual_vaccination_csv: Optional[Path] = None
    vaccination_schedule_pt: Optional[Path] = None

    municipality_cases_csv: Optional[Path] = None
    municipality_tested_csv: Optional[Path] = None

    prior_immune_fraction: float = 0.0

    prior_immunity_scale: Optional[float] = None


    EPI_GROUPS = ("disease_params", "transmission_cfg", "severity_cfg",
                  "vaccination_cfg", "testing_cfg", "seasonal_cfg", "reinfection",
                  "variant")

    @staticmethod
    def _ensure_ci(val):
        """Give a parameter value uncertainty bounds if it has none."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            m = float(val)
            return {"mean": m, "ci95": [round(m * 0.5, 2), round(m * 1.5, 2)]}
        if isinstance(val, dict):
            spec = dict(val)
            if "ci95" not in spec and "mean" in spec:
                m = float(spec["mean"])
                spec["ci95"] = [round(m * 0.5, 2), round(m * 1.5, 2)]
            return spec
        return val

    @classmethod
    def parse_epi_params(cls, cfg: dict) -> dict:
        """Parse a flat epidemiological config into the groups the model consumes."""
        _ci = cls._ensure_ci

        disease_params = {}
        for key in ("latent_period", "infectious_period", "generation_time",
                    "immunity_waning"):
            if key in cfg:
                disease_params[key] = _ci(cfg[key])
        if isinstance(cfg.get("disease_params"), dict):
            for key, val in cfg["disease_params"].items():
                disease_params.setdefault(key, _ci(val))

        transmission_cfg = {k: _ci(cfg[k]) for k in
                            ("R0", "asymptomatic_infectiousness",
                             "asymptomatic_fraction", "seasonal_amplitude")
                            if k in cfg}

        severity_cfg = {k: _ci(cfg[k]) for k in
                        ("ifr", "ihr", "hospitalization_delay", "death_delay")
                        if k in cfg}
        for key in ("age_ifr_multiplier", "age_ihr_multiplier"):
            if key in cfg:
                severity_cfg[key] = cfg[key]
        if isinstance(cfg.get("severity"), dict):
            for key, val in cfg["severity"].items():
                severity_cfg.setdefault(key, val)

        return {
            "disease_params":   disease_params or None,
            "transmission_cfg": transmission_cfg or None,
            "severity_cfg":     severity_cfg or None,
            "vaccination_cfg":  cfg.get("vaccination", None),
            "testing_cfg":      cfg.get("testing", None),
            "seasonal_cfg":     cfg.get("seasonal", None),
            "reinfection":      _ci(cfg.get("reinfection_susceptibility", None)),
            "variant":          cfg.get("variant", None),
        }

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, repo_root: str | Path | None = None,
                  model_config_path: str | Path | None = None,
                  require_epi_params: bool = False,
                  quiet: bool = False) -> "DataConfig":
        """Load a data config YAML and resolve all paths."""
        root = Path(repo_root) if repo_root else _find_repo_root()
        raw = OmegaConf.to_container(OmegaConf.load(str(yaml_path)), resolve=True)

        pop = raw["population"]
        gt = raw["ground_truth"]
        sim = raw["simulation"]
        pre = raw.get("precomputed", {}) or {}

        pop_dir = root / pop["dir"]
        net_file = pop_dir / pop.get("network_file", "mobility_networks/0.csv")
        gt_dir = root / gt["dir"]

        precomputed = {}
        for key, relpath in pre.items():
            if relpath:
                p = root / relpath
                if p.exists():
                    precomputed[key] = p

        ind = raw.get("individual_data", {}) or {}

        def _resolve_optional(relpath):
            if relpath:
                p = root / relpath
                return p if p.exists() else None
            return None

        muni = raw.get("municipality_ground_truth", {}) or {}

        num_agents = pop.get("num_agents", 0)
        if not num_agents or num_agents == 0:
            from .agent_properties import find_file, read_tabular
            agents_file = find_file(pop_dir, "agents")
            if agents_file is not None:
                num_agents = len(read_tabular(agents_file))
            else:
                raise ValueError(f"num_agents not set and no agents.parquet found in {pop_dir}")
        elif num_agents == "auto":
            from .agent_properties import find_file, read_tabular
            agents_file = find_file(pop_dir, "agents")
            if agents_file is not None:
                num_agents = len(read_tabular(agents_file))
            else:
                raise ValueError(f"num_agents='auto' but no agents.parquet found in {pop_dir}")

        dc = cls(
            repo_root=root,
            population_dir=pop_dir,
            network_file=net_file,
            ground_truth_dir=gt_dir,
            num_agents=num_agents,
            real_population=gt["real_population"],
            start_date=sim["start_date"],
            end_date=sim["end_date"],
            start_doy=sim.get("start_doy", 0),
            precomputed=precomputed,
            individual_test_csv=_resolve_optional(ind.get("test_data_csv")),
            individual_vaccination_csv=_resolve_optional(ind.get("vaccination_csv")),
            vaccination_schedule_pt=_resolve_optional(ind.get("vaccination_schedule_pt")),
            municipality_cases_csv=_resolve_optional(muni.get("cases_csv")),
            municipality_tested_csv=_resolve_optional(muni.get("tested_csv")),
            prior_immune_fraction=cls._resolve_prior_immune_fraction(sim, pop_dir),
            prior_immunity_scale=sim.get("prior_immunity_scale", None),
        )

        mc = Path(model_config_path) if model_config_path else \
            Path(yaml_path).parent / "config.yaml"
        epi, src = {g: None for g in cls.EPI_GROUPS}, None
        if mc.exists() and mc.resolve() != Path(yaml_path).resolve():
            epi = cls.parse_epi_params(
                OmegaConf.to_container(OmegaConf.load(str(mc)), resolve=True) or {})
            src = mc
        for group, val in epi.items():
            setattr(dc, f"_{group}", val)

        _live = sorted(g for g, v in epi.items() if v)
        if _live and not quiet:
            _dp = epi.get("disease_params") or {}
            print(f"  [DataConfig] epidemiological parameters from {src}: "
                  f"{', '.join(_live)}")
            if _dp:
                print("  [DataConfig]   " + ", ".join(
                    f"{k}={(v or {}).get('mean') if isinstance(v, dict) else v}"
                    for k, v in sorted(_dp.items())))
        elif not _live:
            msg = (f"NO epidemiological parameters found for this wave. "
                   f"Looked in {mc}. The model will fall back to the generic "
                   f"defaults in danish_abm/yamls/config.yaml — for an Omicron wave "
                   f"that means permanent immunity (immunity_waning 999) and "
                   f"generation_time 5.5. Every wave-specific cited value "
                   f"(ifr, ihr, age multipliers, reinfection_susceptibility, "
                   f"vaccination efficacy, test sensitivities) is ignored.")
            if require_epi_params:
                raise ValueError(msg)
            if not quiet:
                print(f"  [DataConfig] ** WARNING: {msg} **")
        return dc

    @classmethod
    def _resolve_prior_immune_fraction(cls, sim: dict, pop_dir: Path) -> float:
        """Return prior_immune_fraction: explicit YAML value, or auto-detect from
        disease_stage.csv.
        """
        explicit = sim.get("prior_immune_fraction", None)
        if explicit is not None:
            return float(explicit)
        stage_file = pop_dir / "initial_state" / "disease_stage.csv"
        if stage_file.exists():
            import pandas as _pd
            _df = _pd.read_csv(stage_file)
            col = _df.columns[0] if len(_df.columns) == 1 else "stage"
            n_recovered = _df[col].isin([3, "3", "R", "r"]).sum()
            if len(_df) > 0:
                frac = n_recovered / len(_df)
                print(f"  [DataConfig] Auto-detected prior_immune_fraction={frac:.4f} "
                      f"({n_recovered:,}/{len(_df):,} recovered) from {stage_file}")
                return frac
        return 0.0


    def delay_days(self, key: str):
        """A severity delay in whole days, exactly as the simulator will receive it."""
        sc = getattr(self, "_severity_cfg", None) or {}
        if key not in sc:
            return None
        v = scalar_mean(sc[key])
        return None if v is None else int(v)

    @property
    def scale_factor(self) -> float:
        return self.num_agents / self.real_population

    @property
    def n_days(self) -> int:
        d0 = datetime.strptime(self.start_date, "%Y-%m-%d")
        d1 = datetime.strptime(self.end_date, "%Y-%m-%d")
        return (d1 - d0).days + 1

    @property
    def population_dir_rel(self) -> str:
        """Population dir relative to repo root (for config.yaml)."""
        return str(self.population_dir.relative_to(self.repo_root))


    def load_ground_truth(self) -> dict:
        """Load SSI surveillance CSVs, filter to sim period, scale to model pop."""
        gt_dir = self.ground_truth_dir
        scale = self.scale_factor
        start = self.start_date
        end = self.end_date

        tp = pd.read_csv(gt_dir / "Test_pos_over_time.csv", sep=";")
        tp = tp[~tp["Date"].str.contains("alt|person", case=False, na=False)]
        tp["Date"] = pd.to_datetime(tp["Date"])
        for col in ["NewPositive", "Tested"]:
            tp[col] = tp[col].astype(str).str.strip().str.replace(".", "", regex=False)
            tp[col] = pd.to_numeric(tp[col], errors="coerce").fillna(0)
        tp = tp[(tp["Date"] >= start) & (tp["Date"] <= end)].sort_values("Date").reset_index(drop=True)

        d = pd.read_csv(gt_dir / "Deaths_over_time.csv", sep=";", encoding="utf-8-sig")
        d.columns = [c.strip() for c in d.columns]
        d = d[~d["Dato"].str.contains("alt|person", case=False, na=False)]
        d["Dato"] = pd.to_datetime(d["Dato"])
        d["Antal_døde"] = pd.to_numeric(d["Antal_døde"], errors="coerce").fillna(0)
        d_all = d.sort_values("Dato").reset_index(drop=True)
        d = d_all[(d_all["Dato"] >= start) & (d_all["Dato"] <= end)].reset_index(drop=True)
        _end_ts = pd.to_datetime(end)
        d_tail = d_all[(d_all["Dato"] > _end_ts)
                       & (d_all["Dato"] <= _end_ts + pd.Timedelta(days=45))]
        _window_days = len(d)

        h = pd.read_csv(gt_dir / "Newly_admitted_over_time.csv", sep=";", encoding="utf-8-sig")
        h.columns = [c.strip() for c in h.columns]
        h = h[~h["Dato"].str.contains("alt|person", case=False, na=False)]
        h["Dato"] = pd.to_datetime(h["Dato"])
        h["Total"] = pd.to_numeric(h["Total"], errors="coerce").fillna(0)
        h = h[(h["Dato"] >= start) & (h["Dato"] <= end)].sort_values("Dato").reset_index(drop=True)

        n = min(len(tp), len(d), len(h), self.n_days)

        return {
            "daily_cases": torch.tensor(tp["NewPositive"].values[:n], dtype=torch.float32) * scale,
            "daily_deaths": torch.tensor(d["Antal_døde"].values[:n], dtype=torch.float32) * scale,
            "daily_hosp": torch.tensor(h["Total"].values[:n], dtype=torch.float32) * scale,
            "daily_tested": torch.tensor(tp["Tested"].values[:n], dtype=torch.float32) * scale,
            "daily_deaths_tail": (
                torch.tensor(d_tail["Antal_døde"].values, dtype=torch.float32) * scale
                if n == _window_days
                else torch.zeros(0, dtype=torch.float32)),
            "n_days": n,
            "dates": tp["Date"].values[:n],
            "scale_factor": scale,
        }


    def patch_model_config(self, config: dict) -> dict:
        """Update a loaded model config dict with paths/sizes from this DataConfig."""
        meta = config["simulation_metadata"]

        _variant = getattr(self, "_variant", None)
        if isinstance(_variant, dict) and _variant.get("name"):
            meta["variant_name"] = str(_variant["name"])

        meta["population_dir"] = self.population_dir_rel
        meta["num_agents"] = self.num_agents
        meta["infection_network_file"] = str(self.network_file.relative_to(self.repo_root))
        meta["age_group_file"] = str((self.population_dir / "age.pickle").relative_to(self.repo_root))
        meta["disease_stage_file"] = str((self.population_dir / "disease_stages.csv").relative_to(self.repo_root))

        meta["num_steps_per_episode"] = self.n_days
        meta["NUM_WEEKS"] = (self.n_days + 6) // 7
        meta["INFINITY_TIME"] = self.n_days + 50
        meta["sim_start_doy"] = self.start_doy
        meta["start_date"] = self.start_date
        meta["end_date"] = self.end_date
        meta["prior_immune_fraction"] = self.prior_immune_fraction
        meta["prior_immunity_scale"] = self.prior_immunity_scale

        for env_key in ["daily_deaths", "daily_hosp", "daily_infected"]:
            if env_key in config["state"]["environment"]:
                config["state"]["environment"][env_key]["shape"] = [self.n_days]

        lgi = config["state"]["environment"].get("lam_gamma_integrals")
        if lgi and "initialization_function" in lgi:
            lgi["initialization_function"]["arguments"]["t"]["value"] = self.n_days
            if hasattr(self, "_disease_params") and self._disease_params:
                gt_spec = self._disease_params.get("generation_time")
                if gt_spec:
                    gt_mean = gt_spec.get("mean", gt_spec) if isinstance(gt_spec, dict) else float(gt_spec)
                    lgi["initialization_function"]["arguments"]["scale"]["value"] = float(gt_mean)

        r2 = config.get("substeps", {}).get("0", {}).get("transition", {}).get(
            "new_transmission", {}).get("arguments", {}).get("R2")
        if r2:
            r2["shape"] = [meta["NUM_WEEKS"], 1]

        config["state"]["agents"]["citizens"]["number"] = self.num_agents

        if self.individual_test_csv:
            meta["individual_test_data_path"] = str(self.individual_test_csv)
            meta["record_trajectory"] = True
        if self.vaccination_schedule_pt:
            meta["vaccination_schedule_path"] = str(self.vaccination_schedule_pt)
            meta["vaccination_strategy"] = "registry"
        elif self.individual_vaccination_csv:
            meta["vaccination_schedule_path"] = str(
                ensure_vaccination_schedule(self.individual_vaccination_csv,
                                            self.population_dir, self.start_date,
                                            self.n_days, self.num_agents))
            meta["vaccination_strategy"] = "registry"

        if self.municipality_cases_csv:
            meta["municipality_mixing"] = True


        def _mean_of(val):
            """Extract mean from a {mean, ci95} dict or scalar."""
            if val is None:
                return None
            if isinstance(val, dict):
                return val.get("mean", val)
            return val

        def _scalar_mean(val):
            """A NUMBER, or None. Never a dict. Delegates to the module-level
            `scalar_mean`, which `delay_days` also uses -- see the note there.
            """
            return scalar_mean(val)

        if hasattr(self, "_disease") and self._disease:
            meta["pathogen_preset"] = self._disease

        if hasattr(self, "_layer_names") and self._layer_names:
            meta["network_layer_names"] = self._layer_names
            meta["num_network_layers"] = len(self._layer_names)

        if hasattr(self, "_disease_params") and self._disease_params:
            meta["disease_params_distributional"] = True
            meta["disease_params"] = {}
            for key in ["latent_period", "infectious_period", "generation_time", "immunity_waning"]:
                if key in self._disease_params:
                    meta["disease_params"][key] = self._disease_params[key]

        if hasattr(self, "_transmission_cfg") and self._transmission_cfg:
            tc = self._transmission_cfg
            if "R0" in tc:
                meta["R0"] = tc["R0"]
            if "asymptomatic_infectiousness" in tc:
                meta["asymptomatic_infectiousness"] = tc["asymptomatic_infectiousness"]
            if "asymptomatic_fraction" in tc:
                meta["asymptomatic_fraction"] = tc["asymptomatic_fraction"]
            if "seasonal_amplitude" in tc:
                amp = tc["seasonal_amplitude"]
                m = _scalar_mean(amp)
                if m is not None:
                    meta["seasonal"] = {"kind": "sinusoidal",
                                        "amplitude": float(m), "peak_day": 15}
                    meta["_seasonal_amplitude_ci"] = amp

        if hasattr(self, "_severity_cfg") and self._severity_cfg:
            sc = self._severity_cfg
            if "ifr" in sc:
                meta["ifr"] = sc["ifr"]
            if "ihr" in sc:
                meta["ihr"] = sc["ihr"]
            if "hospitalization_delay" in sc:
                hd = sc["hospitalization_delay"]
                _hd = _scalar_mean(hd)
                if _hd is not None:
                    meta["hospitalization_delay"] = int(_hd)
                meta["_hospitalization_delay_ci"] = hd
            if "death_delay" in sc:
                dd = sc["death_delay"]
                _dd = _scalar_mean(dd)
                if _dd is not None:
                    meta["death_delay"] = int(_dd)
                meta["_death_delay_ci"] = dd
            if "age_ifr_multiplier" in sc:
                meta["age_relative_ifr"] = sc["age_ifr_multiplier"]
            if "age_ihr_multiplier" in sc:
                meta["age_relative_ihr"] = sc["age_ihr_multiplier"]

        if hasattr(self, "_vaccination_cfg") and self._vaccination_cfg:
            vc = self._vaccination_cfg
            meta["vaccination_enabled"] = vc.get("enabled", True)
            if "vaccine_type" in vc:
                meta["vaccination_vaccine_type"] = vc["vaccine_type"]
            if "daily_capacity" in vc:
                meta["vaccination_daily_capacity"] = vc["daily_capacity"]
            if "start_day" in vc:
                meta["vaccination_start_day"] = vc["start_day"]
            if "strategy" in vc:
                meta["vaccination_strategy"] = vc["strategy"]
            if "efficacy" in vc:
                meta["vaccination_efficacy"] = vc["efficacy"]
            if "age_efficacy" in vc:
                meta["vaccination_age_efficacy"] = {
                    str(k): float(v) for k, v in vc["age_efficacy"].items()}
            if "ve_death" in vc:
                meta["vaccination_ve_death"] = vc["ve_death"]
            if "waning_half_life_death" in vc:
                meta["vaccination_waning_half_life_death"] = vc["waning_half_life_death"]
            if "efficacy" in vc or "waning_half_life" in vc or "ve_death" in vc:
                meta["_vaccination_overrides"] = vc

        if hasattr(self, "_testing_cfg") and self._testing_cfg:
            tc = self._testing_cfg
            meta["testing_enabled"] = tc.get("enabled", True)
            if "daily_rate" in tc:
                meta["base_testing_rate"] = tc["daily_rate"]
            for _name, _meta_key in (("sensitivity", "test_sensitivity"),
                                     ("specificity", "test_specificity")):
                if _name not in tc:
                    continue
                meta[f"_test_{_name}_ci"] = tc[_name]
                _s = _scalar_mean(tc[_name])
                if _s is not None:
                    meta[_meta_key] = _s
                else:
                    print(f"  [DataConfig] testing.{_name} is per-arm "
                          f"({sorted(tc[_name])}); not applied as a scalar "
                          f"{_meta_key}. The testing substep keeps its configured "
                          f"value; the anchor takes its arm accuracy from "
                          f"--anchor-test-types / --antigen-sensitivity.")
            if "symptomatic_multiplier" in tc:
                meta["symptomatic_testing_multiplier"] = tc["symptomatic_multiplier"]
            if "isolation_compliance" in tc:
                _ic = _scalar_mean(tc["isolation_compliance"])
                if _ic is not None:
                    meta["isolation_compliance"] = _ic
                meta["_isolation_compliance_ci"] = tc["isolation_compliance"]
            if "isolation_duration" in tc:
                meta["isolation_duration"] = tc["isolation_duration"]
            if "isolation_reduction" in tc:
                _ir = _scalar_mean(tc["isolation_reduction"])
                if _ir is not None:
                    meta["isolation_reduction"] = _ir
            if "isolation_in_transmission" in tc:
                meta["isolation_in_transmission"] = bool(tc["isolation_in_transmission"])
            if "isolation_effectiveness" in tc:
                meta["isolation_effectiveness"] = tc["isolation_effectiveness"]
            if "strategy" in tc:
                meta["testing_strategy"] = tc["strategy"]

        if hasattr(self, "_seasonal_cfg") and self._seasonal_cfg is not None:
            meta["seasonal"] = self._seasonal_cfg

        if hasattr(self, "_reinfection") and self._reinfection is not None:
            _rs = _scalar_mean(self._reinfection)
            if _rs is not None:
                meta["REINFECTION_SUSCEPTIBILITY"] = _rs
            meta["_reinfection_ci"] = self._reinfection

        if hasattr(self, "_npis") and self._npis is not None:
            npis = self._npis
            remove = getattr(self, "_remove_npis", None) or []
            delay = getattr(self, "_delay_npis_days", None) or 0

            if npis == "denmark_historical":
                try:
                    from .data.oxcgrt.prep_oxcgrt import save_intervention_schedule
                    from .substeps.utils import interv_schedule_path
                    pop_dir = meta["population_dir"]
                    interv_path = interv_schedule_path(pop_dir)
                    save_intervention_schedule(
                        self.start_date, self.end_date, interv_path,
                    )
                    if remove or delay:
                        data = torch.load(interv_path, weights_only=False)
                        mult = data["multipliers"]
                        npi_col_map = {
                            "school_closure": 4,
                            "workplace_closure": 3,
                            "mask_mandate": 5,
                            "lockdown": [2, 3, 4, 5],
                            "social_distancing": [2, 5],
                        }
                        for npi_name in remove:
                            cols = npi_col_map.get(npi_name, [])
                            if isinstance(cols, int):
                                cols = [cols]
                            for c in cols:
                                mult[:, c] = 1.0
                        if delay and delay != 0:
                            shifted = torch.ones_like(mult)
                            if delay > 0:
                                shifted[delay:] = mult[:-delay]
                            else:
                                shifted[:delay] = mult[-delay:]
                            mult = shifted
                        data["multipliers"] = mult
                        torch.save(data, interv_path)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Could not load OxCGRT NPI data: {e}. Continuing without NPIs.")

            elif isinstance(npis, list):
                meta["_npis_from_config"] = npis

        return config

    def create_sample(self, frac: float, seed: int = 42) -> "DataConfig":
        """Create a stratified subsample of the population for calibration."""
        import shutil

        rng = np.random.RandomState(seed)
        sample_dir = self.population_dir / "_sample"
        sample_dir.mkdir(exist_ok=True)

        from .agent_properties import find_file, read_tabular
        agents_file = find_file(self.population_dir, "agents")
        agents_df = read_tabular(agents_file)
        n_full = len(agents_df)
        n_sample = max(1, int(n_full * frac))

        agents_df["_orig_idx"] = np.arange(n_full)

        age_col = None
        for c in ["age", "age_group", "Age", "AGE", "alder"]:
            if c in agents_df.columns:
                age_col = c
                break

        if age_col:
            agents_df["_strat"] = pd.qcut(agents_df[age_col], q=10, labels=False, duplicates="drop")
            sampled = agents_df.groupby("_strat", group_keys=False).apply(
                lambda g: g.sample(frac=frac, random_state=rng)
            ).reset_index(drop=True)
            sampled.drop(columns=["_strat"], inplace=True)
        else:
            sampled = agents_df.sample(n=n_sample, random_state=rng).reset_index(drop=True)

        orig_ids = sampled["_orig_idx"].values
        id_map = {int(old): new for new, old in enumerate(orig_ids)}
        sampled = sampled.drop(columns=["_orig_idx"])
        sampled.to_parquet(sample_dir / "agents.parquet", index=False)

        n_sampled = len(sampled)
        print(f"  Sampled {n_sampled:,} / {n_full:,} agents ({100*frac:.0f}%)")

        net_dir = sample_dir / "networks"
        net_dir.mkdir(exist_ok=True)
        orig_set = set(orig_ids)

        def _sample_network_file(net_path, out_path, orig_set, id_map, fmt="csv"):
            """Filter and remap a single network file."""
            if fmt == "parquet":
                net_df = pd.read_parquet(net_path)
            else:
                net_df = pd.read_csv(net_path)
            cols = net_df.columns.tolist()
            src_col = next((c for c in cols if "source" in c.lower() or c == "src"), cols[0])
            tgt_col = next((c for c in cols if "target" in c.lower() or c in ("dst", "tgt")), cols[1])

            mask = net_df[src_col].isin(orig_set) & net_df[tgt_col].isin(orig_set)
            net_sample = net_df[mask].copy()
            net_sample[src_col] = net_sample[src_col].map(id_map)
            net_sample[tgt_col] = net_sample[tgt_col].map(id_map)
            if str(out_path).endswith(".parquet"):
                net_sample.to_parquet(out_path, index=False)
            else:
                net_sample.to_csv(out_path, index=False)
            return len(net_sample), len(net_df)

        total_kept, total_orig = 0, 0
        if self.network_file.exists():
            kept, orig = _sample_network_file(
                self.network_file, net_dir / "0.csv", orig_set, id_map
            )
            total_kept += kept
            total_orig += orig
        else:
            orig_net_dir = self.network_file.parent
            if not orig_net_dir.exists():
                orig_net_dir = self.population_dir / "networks"
            layer_files = []
            if orig_net_dir.exists():
                for f in sorted(orig_net_dir.iterdir()):
                    if f.suffix in (".parquet", ".pq", ".csv") and f.stem != "0":
                        layer_files.append(f)
            if layer_files:
                for lf in layer_files:
                    out_ext = lf.suffix
                    out_path = net_dir / f"{lf.stem}{out_ext}"
                    fmt = "parquet" if lf.suffix in (".parquet", ".pq") else "csv"
                    kept, orig = _sample_network_file(lf, out_path, orig_set, id_map, fmt)
                    total_kept += kept
                    total_orig += orig
                    print(f"    {lf.stem}: {kept:,} / {orig:,} edges")
            else:
                print(f"  WARNING: no network files found, skipping network sampling")

        if total_orig > 0:
            print(f"  Network total: {total_kept:,} / {total_orig:,} edges retained")

        events_dir = self.population_dir / "events"
        sample_events_dir = sample_dir / "events"
        sample_events_dir.mkdir(exist_ok=True)
        if events_dir.exists():
            sample_events_dir.mkdir(exist_ok=True)
            for event_file in list(events_dir.glob("*.csv")) + list(events_dir.glob("*.parquet")):
                ev_df = read_tabular(event_file)
                id_col = None
                for c in ["agent_id", "person_id", "id", "agent"]:
                    if c in ev_df.columns:
                        id_col = c
                        break
                if id_col:
                    ev_sample = ev_df[ev_df[id_col].isin(orig_set)].copy()
                    ev_sample[id_col] = ev_sample[id_col].map(id_map)
                    if event_file.suffix == ".parquet":
                        ev_sample.to_parquet(sample_events_dir / event_file.name, index=False)
                    else:
                        ev_sample.to_csv(sample_events_dir / event_file.name, index=False)
                else:
                    shutil.copy2(event_file, sample_events_dir / event_file.name)

        import pickle
        _pkl_to_col = {
            "age": ["age", "age_group", "Age", "AGE", "alder"],
            "sex": ["sex", "Sex", "SEX", "koen", "køn"],
            "municipality": ["municipality", "Municipality", "kommune", "muni_id"],
        }
        for pkl_name, candidates in _pkl_to_col.items():
            matched_col = None
            for c in candidates:
                if c in sampled.columns:
                    matched_col = c
                    break
            if matched_col:
                data = sampled[matched_col].values
                with open(sample_dir / f"{pkl_name}.pickle", "wb") as f:
                    pickle.dump(data, f)

        for pkl_file in self.population_dir.glob("*.pickle"):
            if pkl_file.stem in _pkl_to_col:
                continue
            if (sample_dir / pkl_file.name).exists():
                continue
            with open(pkl_file, "rb") as f:
                orig_data = pickle.load(f)
            if hasattr(orig_data, '__len__') and len(orig_data) == n_full:
                if hasattr(orig_data, "iloc"):
                    data = orig_data.iloc[orig_ids]
                    if hasattr(data, "reset_index"):
                        data = data.reset_index(drop=True)
                elif isinstance(orig_data, np.ndarray):
                    data = orig_data[orig_ids]
                else:
                    data = [orig_data[int(i)] for i in orig_ids]
                with open(sample_dir / pkl_file.name, "wb") as f:
                    pickle.dump(data, f)

        np.savetxt(sample_dir / "disease_stages.csv", np.zeros((n_sampled, 1), dtype=int), fmt="%d")

        for f in self.population_dir.iterdir():
            if f.is_file() and f.suffix not in (".parquet", ".csv", ".pickle"):
                if not (sample_dir / f.name).exists() and f.stat().st_size < 10_000_000:
                    shutil.copy2(f, sample_dir / f.name)

        from dataclasses import replace
        sample_net = net_dir / "0.csv"
        sample_test_csv = next(
            (p for p in [sample_events_dir / "tests.csv", sample_events_dir / "tests.parquet"] if p.exists()),
            None,
        )
        sample_vacc_csv = next(
            (p for p in [sample_events_dir / "vaccinations.csv", sample_events_dir / "vaccinations.parquet"] if p.exists()),
            None,
        )

        return replace(
            self,
            population_dir=sample_dir,
            network_file=sample_net,
            num_agents=n_sampled,
            individual_test_csv=sample_test_csv if self.individual_test_csv else None,
            individual_vaccination_csv=sample_vacc_csv if self.individual_vaccination_csv else None,
        )

    @property
    def has_individual_test_data(self) -> bool:
        return self.individual_test_csv is not None

    @property
    def has_individual_vaccination_data(self) -> bool:
        return self.individual_vaccination_csv is not None or self.vaccination_schedule_pt is not None

    @property
    def has_municipality_data(self) -> bool:
        return self.municipality_cases_csv is not None


    @classmethod
    def from_folder(
        cls,
        study_dir: str | Path,
        repo_root: str | Path | None = None,
    ) -> "DataConfig":
        """Create a DataConfig from a simple study folder."""
        study_dir = Path(study_dir).resolve()
        root = Path(repo_root).resolve() if repo_root else _find_repo_root()

        cfg_path = study_dir / "config.yaml"
        if cfg_path.exists():
            cfg = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)
        else:
            cfg = {}

        start_date = cfg.get("start_date", "2021-01-01")
        end_date = cfg.get("end_date", "2021-06-30")
        disease = cfg.get("disease", None)
        real_population = cfg.get("real_population", None)


        _epi = cls.parse_epi_params(cfg)
        disease_params = _epi["disease_params"]
        transmission_cfg = _epi["transmission_cfg"]
        severity_cfg = _epi["severity_cfg"]
        vaccination_cfg = _epi["vaccination_cfg"]
        testing_cfg = _epi["testing_cfg"]
        reinfection = _epi["reinfection"]
        seasonal_cfg = _epi["seasonal_cfg"]

        npis_cfg = cfg.get("npis", None)
        remove_npis = cfg.get("remove_npis", None)
        delay_npis_days = cfg.get("delay_npis_days", None)

        from .agent_properties import find_file, read_tabular, find_tabular_files
        agents_file = find_file(study_dir, "agents")
        if agents_file is None:
            raise FileNotFoundError(
                f"No agents file found in {study_dir}. "
                f"Expected agents.csv or agents.parquet"
            )

        agents_df = read_tabular(agents_file)
        n_agents = len(agents_df)

        if real_population is None:
            real_population = n_agents

        pop_dir = study_dir / "_population"
        pop_dir.mkdir(exist_ok=True)

        import pickle as _pkl

        id_col = None
        for candidate in ["agent_id", "id", "person_id"]:
            if candidate in agents_df.columns:
                id_col = candidate
                break
        if id_col:
            agents_df = agents_df.sort_values(id_col).reset_index(drop=True)

        for col in agents_df.columns:
            if col == id_col:
                continue
            with open(pop_dir / f"{col}.pickle", "wb") as f:
                _pkl.dump(agents_df[col].values, f)

        import shutil
        shutil.copy2(agents_file, pop_dir / agents_file.name)

        stages = np.zeros((n_agents, 1), dtype=np.int32)
        np.savetxt(pop_dir / "disease_stages.csv", stages, fmt="%d")

        if "municipality" in agents_df.columns:
            codes = sorted(agents_df["municipality"].unique().tolist())
            import json
            with open(pop_dir / "mapping.json", "w") as f:
                json.dump({str(c): str(c) for c in codes}, f)

        net_dir = study_dir / "networks"
        merged_net_dir = pop_dir / "mobility_networks"
        merged_net_dir.mkdir(exist_ok=True)

        layer_names = []
        all_edges = []
        if net_dir.exists():
            net_files = find_tabular_files(net_dir)
            seen_stems = set()
            unique_files = []
            for f in net_files:
                if f.stem not in seen_stems:
                    seen_stems.add(f.stem)
                    unique_files.append(f)

            for net_file in unique_files:
                layer_name = net_file.stem
                layer_idx = len(layer_names)
                layer_names.append(layer_name)

                edf = read_tabular(net_file)
                cols = [c.lower().strip() for c in edf.columns]
                edf.columns = cols
                if "source" in cols and "target" in cols:
                    src = edf["source"].values
                    tgt = edf["target"].values
                elif len(cols) >= 2:
                    src = edf.iloc[:, 0].values
                    tgt = edf.iloc[:, 1].values
                else:
                    continue

                layer_df = pd.DataFrame({
                    "source": src, "target": tgt,
                    "edge_type": layer_idx,
                })
                all_edges.append(layer_df)

        if all_edges:
            merged = pd.concat(all_edges, ignore_index=True)
            merged.to_csv(merged_net_dir / "0.csv", index=False)
        else:
            pd.DataFrame(columns=["source", "target", "edge_type"]).to_csv(
                merged_net_dir / "0.csv", index=False
            )

        net_file = merged_net_dir / "0.csv"

        events_dir = study_dir / "events"
        individual_test_csv = None
        individual_vaccination_csv = None
        vaccination_schedule_pt = None

        if events_dir.exists():
            tests_file = find_file(events_dir, "tests")
            if tests_file is not None:
                individual_test_csv = tests_file

            vax_file = find_file(events_dir, "vaccinations")
            if vax_file is not None:
                individual_vaccination_csv = vax_file
                try:
                    from .registry_data import vaccination_csv_to_schedule, save_vaccination_schedule
                    n_days = (datetime.strptime(end_date, "%Y-%m-%d") -
                              datetime.strptime(start_date, "%Y-%m-%d")).days + 1
                    vax_df = read_tabular(vax_file)
                    _agents_pq = pop_dir / "agents.parquet"
                    schedule = vaccination_csv_to_schedule(
                        vax_df, start_date, n_days, n_agents,
                        agents_file=_agents_pq if _agents_pq.exists() else None,
                    )
                    sched_pt = pop_dir / "vaccination_schedule.pt"
                    save_vaccination_schedule(schedule, sched_pt)
                    vaccination_schedule_pt = sched_pt
                except Exception:
                    pass

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        start_doy = start_dt.timetuple().tm_yday

        prior_immune_fraction = 0.0
        initial_state_dir = study_dir / "initial_state"
        if initial_state_dir.exists():
            stage_file = initial_state_dir / "disease_stage.csv"
            if stage_file.exists():
                import pandas as _pd
                _stage_df = _pd.read_csv(stage_file)
                if "stage" in _stage_df.columns:
                    n_recovered = _stage_df["stage"].isin([3, "3", "R", "r"]).sum()
                    if n_agents > 0:
                        prior_immune_fraction = n_recovered / n_agents

        dc = cls(
            repo_root=root,
            population_dir=pop_dir,
            network_file=net_file,
            ground_truth_dir=study_dir,
            num_agents=n_agents,
            real_population=real_population,
            start_date=start_date,
            end_date=end_date,
            start_doy=start_doy,
            individual_test_csv=individual_test_csv,
            individual_vaccination_csv=individual_vaccination_csv,
            vaccination_schedule_pt=vaccination_schedule_pt,
            prior_immune_fraction=prior_immune_fraction,
        )

        dc._disease = disease
        dc._layer_names = layer_names
        dc._study_dir = study_dir

        dc._disease_params = disease_params
        dc._transmission_cfg = transmission_cfg
        dc._vaccination_cfg = vaccination_cfg
        dc._testing_cfg = testing_cfg
        dc._seasonal_cfg = seasonal_cfg
        dc._severity_cfg = severity_cfg
        dc._reinfection = reinfection
        dc._npis = npis_cfg
        dc._remove_npis = remove_npis
        dc._delay_npis_days = delay_npis_days

        return dc

    def __repr__(self) -> str:
        extras = []
        if self.has_individual_test_data:
            extras.append("individual tests")
        if self.has_individual_vaccination_data:
            extras.append("individual vaccination")
        if self.has_municipality_data:
            extras.append("municipality GT")
        extra_str = f"\n  registry_data=[{', '.join(extras)}]" if extras else ""
        return (
            f"DataConfig(\n"
            f"  population_dir={self.population_dir}\n"
            f"  ground_truth_dir={self.ground_truth_dir}\n"
            f"  num_agents={self.num_agents:,}\n"
            f"  period={self.start_date} → {self.end_date} ({self.n_days} days)\n"
            f"  scale={self.scale_factor:.5f}{extra_str}\n"
            f"  prior_immune_fraction={self.prior_immune_fraction:.3f}\n"
            f")"
        )
