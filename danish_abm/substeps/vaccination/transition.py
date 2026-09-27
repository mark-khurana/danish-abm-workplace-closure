"""Vaccination substep — administers vaccines to agents during simulation."""

import os
import torch
import re

from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_by_path


class VaccinationRollout(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.n_agents = self.config["simulation_metadata"]["num_agents"]

        self.enabled = self.config["simulation_metadata"].get("vaccination_enabled", False)
        self.strategy = self.config["simulation_metadata"].get("vaccination_strategy", "age_priority")
        self.daily_capacity = self.config["simulation_metadata"].get("vaccination_daily_capacity", 0)
        self.start_day = self.config["simulation_metadata"].get("vaccination_start_day", 0)
        self.vaccine_type = self.config["simulation_metadata"].get("vaccination_vaccine_type", "mrna_covid")

        self._dose_count = None
        self._last_dose_day = None
        self._schedule = None

        self._immunity_tracker = None
        self._strain_config_ref = None
        self._vaccine_profile = None
        self._vaccine_susc_fallback = None
        self._vaccine_death_factor = None

        vax_overrides = self.config["simulation_metadata"].get("_vaccination_overrides", {})
        eff_val = self.config["simulation_metadata"].get("vaccination_efficacy", None)
        if eff_val is not None:
            if isinstance(eff_val, dict):
                self._config_efficacy = float(eff_val.get("mean", eff_val.get("value", 0.95)))
            else:
                self._config_efficacy = float(eff_val)
        else:
            self._config_efficacy = None

        ve_death_val = vax_overrides.get("ve_death", None)
        if ve_death_val is None:
            ve_death_val = self.config["simulation_metadata"].get("vaccination_ve_death", None)
        if ve_death_val is not None:
            if isinstance(ve_death_val, dict):
                self._ve_death = float(ve_death_val.get("mean", ve_death_val.get("value", 0.90)))
            else:
                self._ve_death = float(ve_death_val)
        else:
            self._ve_death = 0.90

        whl = vax_overrides.get("waning_half_life", None)
        if whl is not None:
            self._waning_half_life = float(whl["mean"] if isinstance(whl, dict) else whl)
        else:
            self._waning_half_life = None
        self._waning_residual = float(vax_overrides.get("waning_residual", 0.0))
        bm = vax_overrides.get("booster_multiplier", None)
        self._booster_multiplier = float(bm) if bm is not None else None

        whl_death = vax_overrides.get("waning_half_life_death", None)
        if whl_death is None:
            whl_death = self.config["simulation_metadata"].get("vaccination_waning_half_life_death", None)
        if whl_death is not None:
            self._waning_half_life_death = float(whl_death["mean"] if isinstance(whl_death, dict) else whl_death)
        elif self._waning_half_life is not None:
            self._waning_half_life_death = self._waning_half_life * 2.0
        else:
            self._waning_half_life_death = None

    def _resolve_refs(self):
        """Find the transmission substep's immunity tracker and build the profile."""
        from danish_abm import runtime_get

        if self._immunity_tracker is None:
            try:
                self._immunity_tracker = runtime_get(self.config, "immunity_tracker")
            except Exception:
                pass
        if self._strain_config_ref is None:
            try:
                self._strain_config_ref = runtime_get(self.config, "strain_config")
            except Exception:
                pass

        self._age_efficacy_mult = self.config["simulation_metadata"].get(
            "vaccination_age_efficacy", None
        )

        self._build_vaccine_profile()

    def _build_vaccine_profile(self):
        """Build the VaccineProfile from the wave config and the StrainConfig."""
        from danish_abm.immunity import VACCINE_PROFILES, exponential_waning, VaccineProfile

        sc = self._strain_config_ref
        antigen = getattr(sc, "vaccine_antigen", None) if sc is not None else None
        if antigen is None:
            self._vaccine_profile = VACCINE_PROFILES.get(self.vaccine_type)
            return

        peak, _row = sc.resolve_vaccine_ve(self._config_efficacy)
        if peak is None or peak <= 0:
            self._vaccine_profile = VACCINE_PROFILES.get(self.vaccine_type)
            return

        base = VACCINE_PROFILES.get(self.vaccine_type)

        half_life = self._waning_half_life
        if half_life is None or half_life <= 0:
            half_life = base.waning.half_life if base is not None else 150.0
            residual_frac = base.waning.residual if base is not None else 0.0
        else:
            residual_frac = min(1.0, max(0.0, float(self._waning_residual) / peak))

        booster = self._booster_multiplier
        if booster is None:
            booster = base.booster_multiplier if base is not None else 1.0

        self._vaccine_profile = VaccineProfile(
            name=(base.name if base is not None else self.vaccine_type),
            target_immunity={antigen: float(peak)},
            waning=exponential_waning(half_life=float(half_life),
                                      residual=float(residual_frac)),
            doses=(base.doses if base is not None else 2),
            booster_multiplier=float(booster),
        )

    def _resolve_death_factor(self):
        """Resolve vaccine_death_factor tensor from runtime (works in ALL modes)."""
        if self._vaccine_death_factor is not None:
            return
        try:
            from danish_abm import runtime_get
            self._vaccine_death_factor = runtime_get(self.config, "vaccine_death_factor")
        except Exception:
            pass

    def _init_state(self):
        """Initialize per-agent vaccination tracking."""
        self._dose_count = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)
        self._last_dose_day = torch.full((self.n_agents,), -999, dtype=torch.long, device=self.device)
        self._pre_sim_data = None

        pop_dir = self.config["simulation_metadata"]["population_dir"]
        sched_path = self.config["simulation_metadata"].get(
            "vaccination_schedule_path", None
        )
        if sched_path is None:
            sched_path = os.path.join(pop_dir, "vaccination_schedule.pt")
        if os.path.exists(sched_path):
            data = torch.load(sched_path, weights_only=False)
            self._pre_sim_data = data.pop("_pre_sim", None)
            self._schedule = data
        else:
            self._schedule = None

    def _tracker_immunity_str(self) -> str:
        """Report the immunity tracker's vaccine-antigen column."""
        tr = self._immunity_tracker
        sc = self._strain_config_ref
        antigen = getattr(sc, "vaccine_antigen", None) if sc is not None else None
        if tr is None or antigen is None or antigen not in tr.antigen_idx:
            return ""
        idx = tr.antigen_idx[antigen]
        col = tr.immunity_initial[:, idx]
        n_imm = int((col > 0).sum().item())
        return (f", vax_antigen={antigen}, n_vax_immune={n_imm:,}, "
                f"peak_ve={float(col.max().item()):.4f}")

    def _assert_vaccination_bites(self, n_vaccinated, mask):
        """Fail the run if today's doses conferred no infection protection."""
        if n_vaccinated <= 0:
            return

        tr = self._immunity_tracker
        if tr is not None:
            antigen = getattr(self._strain_config_ref, "vaccine_antigen", None)
            if antigen is None:
                raise ValueError(
                    "Multi-strain run with an immunity tracker but no vaccine "
                    "antigen declared on the StrainConfig. Doses would confer "
                    "VE_death only. Add vaccine_antigen / vaccine_ve to the preset."
                )
            if antigen not in tr.antigen_idx:
                raise ValueError(
                    f"Vaccine antigen {antigen!r} is not among the tracker's "
                    f"antigens {tr.antigen_names}; the tracker and the StrainConfig "
                    f"were built from different objects."
                )
            col = tr.immunity_initial[:, tr.antigen_idx[antigen]]
            n_immune = int((col[mask] > 0).sum().item())
            if n_immune == 0:
                raise ValueError(
                    f"{n_vaccinated:,} agents were vaccinated but none of them has "
                    f"any immunity recorded against antigen {antigen!r}. "
                    f"Vaccination is reducing deaths only."
                )
            return

        if self._vaccine_susc_fallback is not None:
            if float(self._vaccine_susc_fallback[mask].max().item()) >= 1.0 - 1e-9:
                raise ValueError(
                    f"{n_vaccinated:,} agents were vaccinated but every one of them "
                    f"still has susceptibility 1.0. Single-strain vaccination is "
                    f"reducing deaths only."
                )
            return

        raise ValueError(
            "Vaccination fired with neither an immunity tracker nor a "
            "vaccine-susceptibility tensor: nothing can carry infection protection."
        )

    def _apply_pre_sim_vaccinations(self):
        """Apply pre-simulation vaccinations with waning."""
        ps = self._pre_sim_data
        if ps is None or ps.get("agent_ids") is None:
            return

        agent_ids = ps["agent_ids"].to(self.device)
        doses = ps["doses"].to(self.device)
        days_before = ps["days_before"].to(self.device)

        valid = agent_ids < self.n_agents
        agent_ids = agent_ids[valid]
        doses = doses[valid]
        days_before = days_before[valid]

        if len(agent_ids) == 0:
            return

        self._dose_count[agent_ids] = doses
        self._last_dose_day[agent_ids] = days_before

        if self._vaccine_susc_fallback is not None:
            base_eff = self._config_efficacy
            if base_eff is None and self._vaccine_profile is not None:
                base_eff = sum(self._vaccine_profile.target_immunity.values()) / max(
                    len(self._vaccine_profile.target_immunity), 1)
            if base_eff is None:
                base_eff = 0.95

            eff = torch.full((len(agent_ids),), base_eff, device=self.device)
            if self._booster_multiplier is not None:
                boosted = doses >= 3
                eff[boosted] = min(1.0, base_eff * self._booster_multiplier)

            if self._waning_half_life is not None and self._waning_half_life > 0:
                days_elapsed = (-days_before).float()
                decay = torch.pow(
                    torch.tensor(0.5, device=self.device),
                    days_elapsed / self._waning_half_life,
                )
                eff = self._waning_residual + (eff - self._waning_residual) * decay

            new_susc = (1.0 - eff).clamp(min=0.0)
            self._vaccine_susc_fallback[agent_ids] = torch.minimum(
                self._vaccine_susc_fallback[agent_ids], new_susc,
            )

        if self._immunity_tracker is not None and self._vaccine_profile is not None:
            boosted = doses >= 3
            for is_booster in (False, True):
                grp = boosted if is_booster else ~boosted
                if not grp.any():
                    continue
                grp_ids = agent_ids[grp]
                grp_days = days_before[grp]
                for d in grp_days.unique():
                    sel = grp_ids[grp_days == d]
                    m = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
                    m[sel] = True
                    self._immunity_tracker.record_vaccination(
                        m, self._vaccine_profile, time=int(d.item()),
                        is_booster=bool(is_booster),
                    )

        if self._vaccine_death_factor is not None and self._ve_death > 0:
            ve_d = self._ve_death
            eff_death = torch.full((len(agent_ids),), ve_d, device=self.device)
            if self._booster_multiplier is not None:
                boosted = doses >= 3
                eff_death[boosted] = min(1.0, ve_d * self._booster_multiplier)
            if self._waning_half_life_death is not None and self._waning_half_life_death > 0:
                days_elapsed = (-days_before).float()
                decay_death = torch.pow(
                    torch.tensor(0.5, device=self.device),
                    days_elapsed / self._waning_half_life_death,
                )
                eff_death = self._waning_residual + (eff_death - self._waning_residual) * decay_death
            death_factor = (1.0 - eff_death).clamp(min=0.0)
            self._vaccine_death_factor[agent_ids] = torch.minimum(
                self._vaccine_death_factor[agent_ids], death_factor,
            )

        n_pre = len(agent_ids)
        dose_counts = {int(d): int((doses == d).sum()) for d in doses.unique()}
        if self._vaccine_susc_fallback is not None:
            mean_susc = self._vaccine_susc_fallback.mean().item()
            n_protected = (self._vaccine_susc_fallback < 1.0).sum().item()
        else:
            mean_susc = n_protected = 0
        death_str = ""
        if self._vaccine_death_factor is not None:
            mean_df = self._vaccine_death_factor.mean().item()
            death_str = f", mean_death_factor={mean_df:.4f}"
        death_str += self._tracker_immunity_str()
        if self._immunity_tracker is not None:
            antigen = getattr(self._strain_config_ref, "vaccine_antigen", None)
            if antigen is not None and antigen in self._immunity_tracker.antigen_idx:
                col = self._immunity_tracker.immunity_initial[
                    :, self._immunity_tracker.antigen_idx[antigen]]
                if int((col > 0).sum().item()) == 0:
                    raise ValueError(
                        f"{n_pre:,} pre-sim vaccinations were applied but no agent has "
                        f"immunity recorded against {antigen!r}. Pre-wave doses are "
                        f"conferring VE_death only."
                    )
        elif self._vaccine_susc_fallback is not None:
            if float(self._vaccine_susc_fallback.min().item()) >= 1.0 - 1e-9:
                raise ValueError(
                    f"{n_pre:,} pre-sim vaccinations were applied but every agent still "
                    f"has susceptibility 1.0."
                )

        print(f"  [vax pre-sim] Applied {n_pre:,} pre-sim vaccinations "
              f"(doses: {dose_counts}), waning_hl={self._waning_half_life}, "
              f"waning_hl_death={self._waning_half_life_death}, "
              f"protected={n_protected:,}, mean_susc={mean_susc:.4f}{death_str}", flush=True)

        self._pre_sim_data = None

    def _select_agents_age_priority(self, t, ages):
        """Select agents for vaccination using age-priority strategy."""
        min_dose_interval = 21
        eligible_dose1 = (self._dose_count == 0)
        eligible_dose2 = (
            (self._dose_count == 1) &
            (t - self._last_dose_day >= min_dose_interval)
        )
        eligible = eligible_dose1 | eligible_dose2

        if not eligible.any():
            return torch.zeros(self.n_agents, dtype=torch.bool, device=self.device), None

        ages_flat = ages.squeeze().to(self.device)
        sort_key = torch.where(eligible, ages_flat, torch.tensor(-1, device=self.device))
        _, sorted_idx = sort_key.sort(descending=True)

        n_eligible = eligible.sum().item()
        if self.daily_capacity > 0:
            n_vaccinate = min(self.daily_capacity, n_eligible)
        else:
            n_vaccinate = n_eligible

        selected_idx = sorted_idx[:n_vaccinate]
        selected_idx = selected_idx[eligible[selected_idx]]

        mask = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        mask[selected_idx] = True

        dose_numbers = self._dose_count[mask] + 1

        return mask, dose_numbers

    def _select_agents_registry(self, t):
        """Select agents from pre-built registry schedule."""
        if self._schedule is None:
            return torch.zeros(self.n_agents, dtype=torch.bool, device=self.device), None

        day_key = int(t)
        if day_key not in self._schedule:
            return torch.zeros(self.n_agents, dtype=torch.bool, device=self.device), None

        agent_ids = self._schedule[day_key]
        if isinstance(agent_ids, torch.Tensor):
            agent_ids = agent_ids.to(self.device)
        else:
            agent_ids = torch.tensor(agent_ids, dtype=torch.long, device=self.device)

        mask = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        valid = agent_ids < self.n_agents
        mask[agent_ids[valid]] = True

        dose_numbers = self._dose_count[mask] + 1
        return mask, dose_numbers

    def _update_waning(self, t, vaccinated_mask):
        """Recompute waned VE for all vaccinated agents (single-strain mode)."""
        base_eff = self._config_efficacy
        if base_eff is None and self._vaccine_profile is not None:
            base_eff = sum(self._vaccine_profile.target_immunity.values()) / max(
                len(self._vaccine_profile.target_immunity), 1)
        if base_eff is None:
            base_eff = 0.95

        days_elapsed = (t - self._last_dose_day[vaccinated_mask]).float()

        if self._vaccine_susc_fallback is not None:
            decay = torch.pow(
                torch.tensor(0.5, device=self.device),
                days_elapsed / self._waning_half_life,
            )
            eff = self._waning_residual + (base_eff - self._waning_residual) * decay

            if self._booster_multiplier is not None:
                boosted = self._dose_count[vaccinated_mask] >= 3
                eff[boosted] = (self._waning_residual +
                                (min(1.0, base_eff * self._booster_multiplier) - self._waning_residual) *
                                decay[boosted])

            self._vaccine_susc_fallback[vaccinated_mask] = (1.0 - eff).clamp(min=0.0)

        if self._vaccine_death_factor is not None and self._ve_death > 0:
            hl_death = self._waning_half_life_death or self._waning_half_life
            if hl_death is not None and hl_death > 0:
                ve_d = self._ve_death
                decay_death = torch.pow(
                    torch.tensor(0.5, device=self.device),
                    days_elapsed / hl_death,
                )
                eff_death = self._waning_residual + (ve_d - self._waning_residual) * decay_death
                if self._booster_multiplier is not None:
                    boosted = self._dose_count[vaccinated_mask] >= 3
                    eff_death[boosted] = (self._waning_residual +
                                           (min(1.0, ve_d * self._booster_multiplier) - self._waning_residual) *
                                           decay_death[boosted])
                self._vaccine_death_factor[vaccinated_mask] = (1.0 - eff_death).clamp(min=0.0)

    def _apply_ve_death(self, mask, dose_numbers):
        """Apply VE_death to newly vaccinated agents (works in ALL modes)."""
        if self._vaccine_death_factor is None or self._ve_death <= 0:
            return
        ve_d = self._ve_death
        is_booster = (dose_numbers is not None and dose_numbers.max() > 2) if dose_numbers is not None else False
        if is_booster and self._booster_multiplier is not None:
            ve_d = min(1.0, ve_d * self._booster_multiplier)
        death_f = 1.0 - ve_d
        self._vaccine_death_factor[mask] = torch.minimum(
            self._vaccine_death_factor[mask],
            torch.tensor(death_f, device=self._vaccine_death_factor.device),
        )

    def forward(self, state, action):
        t = int(state["current_step"])
        if t == 0:
            cfg_enabled = self.config['simulation_metadata'].get('vaccination_enabled', False)
            print(f"  [VAX-DBG] enabled={self.enabled}, config_enabled={cfg_enabled}, "
                  f"strategy={self.strategy}, start_day={self.start_day}", flush=True)
            if cfg_enabled and not self.enabled:
                print(f"  [VAX-FIX] Re-enabling vaccination from config!", flush=True)
                self.enabled = True
                self.strategy = self.config["simulation_metadata"].get("vaccination_strategy", self.strategy)
        if not self.enabled:
            return {}


        ages = get_by_path(state, re.split("/", self.input_variables["age"]))
        actual_n = ages.shape[0]
        if actual_n != self.n_agents:
            self.n_agents = actual_n

        _need_pre_sim = False
        if self._dose_count is None:
            self._init_state()
            self._resolve_refs()
            _need_pre_sim = True
        elif t == 0:
            self._dose_count.zero_()
            self._last_dose_day.fill_(-999)
            self._verified_bite = False
            self._fired_once = False
            self._resolve_refs()
            sched_path = self.config["simulation_metadata"].get(
                "vaccination_schedule_path", None)
            if sched_path is None:
                pop_dir = self.config["simulation_metadata"]["population_dir"]
                sched_path = os.path.join(pop_dir, "vaccination_schedule.pt")
            if sched_path and os.path.exists(sched_path):
                data = torch.load(sched_path, weights_only=False)
                self._pre_sim_data = data.get("_pre_sim", None)
            _need_pre_sim = True
        else:
            self._resolve_refs()

        self._resolve_death_factor()

        if self._immunity_tracker is None:
            self._vaccine_susc_fallback = None
            try:
                from danish_abm import runtime_get
                self._vaccine_susc_fallback = runtime_get(self.config, "vaccine_susceptibility")
            except Exception:
                pass
            if self._vaccine_susc_fallback is None:
                self._vaccine_susc_fallback = torch.ones(self.n_agents, device=self.device)
                from danish_abm import runtime_set
                runtime_set(self.config, "vaccine_susceptibility", self._vaccine_susc_fallback)
                if t == 0:
                    print(f"  [VAX-DBG] Created vaccine_susc_fallback ourselves ({self.n_agents} agents)",
                          flush=True)
            if self._vaccine_death_factor is None:
                self._vaccine_death_factor = torch.ones(self.n_agents, device=self.device)
                from danish_abm import runtime_set
                runtime_set(self.config, "vaccine_death_factor", self._vaccine_death_factor)
                if t == 0:
                    print(f"  [VAX-DBG] Created vaccine_death_factor ourselves ({self.n_agents} agents)",
                          flush=True)

        if _need_pre_sim:
            self._apply_pre_sim_vaccinations()

        if (self._waning_half_life is not None and self._waning_half_life > 0) or \
           (self._waning_half_life_death is not None and self._waning_half_life_death > 0):
            vaccinated = self._last_dose_day > -999
            if vaccinated.any():
                self._update_waning(t, vaccinated)

        if self.strategy != "registry" and t < self.start_day:
            return {}
        if self.strategy == "registry" and t < self.start_day and not getattr(
                self, "_registry_start_noted", False):
            print(f"  [vax] registry schedule: doses follow their register dates from "
                  f"day {t}; start_day={self.start_day} applies to synthetic rollouts "
                  f"only", flush=True)
            self._registry_start_noted = True

        if self.strategy == "registry":
            mask, dose_numbers = self._select_agents_registry(t)
        else:
            ages = get_by_path(state, re.split("/", self.input_variables["age"]))
            mask, dose_numbers = self._select_agents_age_priority(t, ages)

        if not mask.any():
            return {}

        n_vaccinated = mask.sum().item()

        self._dose_count[mask] += 1
        self._last_dose_day[mask] = t

        if self._immunity_tracker is not None and self._vaccine_profile is not None:
            is_booster = (dose_numbers is not None and dose_numbers.max() > 2) if dose_numbers is not None else False
            self._immunity_tracker.record_vaccination(
                mask, self._vaccine_profile, time=t,
                is_booster=bool(is_booster)
            )
            if self._age_efficacy_mult is not None:
                ages = get_by_path(state, re.split("/", self.input_variables["age"]))
                ages_flat = ages.squeeze().long()
                for ag_str, mult in self._age_efficacy_mult.items():
                    ag = int(ag_str)
                    age_mask = mask & (ages_flat == ag)
                    if age_mask.any():
                        for antigen in self._vaccine_profile.target_immunity:
                            if antigen in self._immunity_tracker.antigen_idx:
                                idx = self._immunity_tracker.antigen_idx[antigen]
                                self._immunity_tracker.immunity_initial[age_mask, idx] *= mult
        elif self._vaccine_susc_fallback is not None:
            if self._waning_half_life is None or self._waning_half_life <= 0:
                if self._config_efficacy is not None:
                    avg_eff = self._config_efficacy
                elif self._vaccine_profile is not None:
                    avg_eff = sum(self._vaccine_profile.target_immunity.values()) / max(len(self._vaccine_profile.target_immunity), 1)
                else:
                    avg_eff = 0.95
                is_booster = (dose_numbers is not None and dose_numbers.max() > 2) if dose_numbers is not None else False
                if is_booster and self._booster_multiplier is not None:
                    avg_eff = min(1.0, avg_eff * self._booster_multiplier)
                new_susc = 1.0 - avg_eff
                self._vaccine_susc_fallback[mask] = torch.minimum(
                    self._vaccine_susc_fallback[mask],
                    torch.tensor(new_susc, device=self._vaccine_susc_fallback.device),
                )
            else:
                self._update_waning(t, mask)

        if not getattr(self, "_verified_bite", False):
            self._verified_bite = True
            self._assert_vaccination_bites(n_vaccinated, mask)

        self._apply_ve_death(mask, dose_numbers)

        _first_fire = not getattr(self, "_fired_once", False)
        if _first_fire:
            self._fired_once = True
        if _first_fire or t % 30 == 0:
            eff_used = self._config_efficacy if self._config_efficacy is not None else "profile-default"
            death_str = ""
            if self._vaccine_death_factor is not None:
                mean_df = self._vaccine_death_factor.mean().item()
                death_str = f", mean_death_factor={mean_df:.4f}"
            susc_str = ""
            if self._vaccine_susc_fallback is not None:
                n_protected = (self._vaccine_susc_fallback < 1.0).sum().item()
                mean_susc = self._vaccine_susc_fallback.mean().item()
                susc_str = f", protected={n_protected:,}, mean_susc={mean_susc:.4f}"
            susc_str += self._tracker_immunity_str()
            print(f"  [vax t={t}] vaccinated {n_vaccinated:,} agents, "
                  f"eff={eff_used}, ve_death={self._ve_death}"
                  f"{susc_str}{death_str}", flush=True)

        if not hasattr(self, '_daily_vaccinated'):
            self._daily_vaccinated = []
        self._daily_vaccinated.append(n_vaccinated)

        return {}
