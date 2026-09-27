import torch
import torch.nn.functional as F
import re

from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_by_path


class SEIRMProgression(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.num_timesteps = self.config["simulation_metadata"]["num_steps_per_episode"]

        self._compartment_model = None
        self._load_compartment_model()

        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]
        self.INFECTED_VAR = self.config["simulation_metadata"]["INFECTED_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]
        self.MORTALITY_VAR = self.config["simulation_metadata"]["MORTALITY_VAR"]
        self.RESET_VAR = -1 * self.RECOVERED_VAR

        self.STAGE_SAME_VAR = 0
        self.STAGE_UPDATE_VAR = 1

        self.calibration_mode = self.config["simulation_metadata"]["calibration"]

        self.INFINITY_TIME = (
            self.config["simulation_metadata"]["num_steps_per_episode"] + 20
        )
        self.INFECTED_TO_RECOVERED_TIME = self.config["simulation_metadata"][
            "INFECTED_TO_RECOVERED_TIME"
        ]

        self.RECOVERED_TO_SUSCEPTIBLE_TIME = self.config["simulation_metadata"][
            "RECOVERED_TO_SUSCEPTIBLE_TIME"
        ]
        self.reinfection_susceptibility = self.config["simulation_metadata"].get(
            "REINFECTION_SUSCEPTIBILITY", 1.0
        )

        self._previously_infected = None

        self._distributional = self.config["simulation_metadata"].get(
            "disease_params_distributional", False
        )
        self._per_agent_infectious = None
        self._per_agent_waning = None
        self._per_agent_transition_times = None

        self.hosp_delay = int(self.config["simulation_metadata"].get("hospitalization_delay", 0))
        self._pending_hosp = []
        self._death_delay = int(self.config["simulation_metadata"].get(
            "death_delay", self.INFECTED_TO_RECOVERED_TIME
        ))
        self._pending_deaths = []
        self._pending_seed = None
        self._pending_seed_missing = False
        self._primed_totals = None
        self._dropped_pending_deaths = 0
        self._dropped_pending_hosp = 0
        self._latest_die_mask = None

        self._multi_strain = self.config["simulation_metadata"].get("multi_strain", False)

        meta = self.config["simulation_metadata"]
        default_rel_ifr = [0.005, 0.002, 0.015, 0.06, 0.15, 0.45, 1.5, 7.5]
        default_rel_ihr = [0.001, 0.001, 0.007, 0.014, 0.024, 0.049, 0.103, 0.21]
        rel_ifr_raw = meta.get("age_relative_ifr", default_rel_ifr)
        rel_ihr_raw = meta.get("age_relative_ihr", default_rel_ihr)
        age_ifr_scale = meta.get("_age_ifr_scale", 1.0)
        age_ihr_scale = meta.get("_age_ihr_scale", 1.0)
        self._age_ifr_raw = torch.tensor(rel_ifr_raw, dtype=torch.float32, device=self.device) * age_ifr_scale
        self._age_ihr_raw = torch.tensor(rel_ihr_raw, dtype=torch.float32, device=self.device) * age_ihr_scale
        self._age_rel_ifr = None
        self._age_rel_ihr = None

        charlson_or = meta.get("charlson_mortality_or", {
            0: 1.0, 1: 2.13, 2: 2.13, 3: 3.00, 4: 3.00, 5: 3.85,
        })
        self._n_charlson_groups = len(charlson_or)
        or_vals = list(charlson_or.values())
        charlson_or_scale = meta.get("_charlson_or_scale", 1.0)
        self._charlson_raw = torch.tensor(
            [charlson_or.get(i, or_vals[-1]) for i in range(self._n_charlson_groups)],
            dtype=torch.float32, device=self.device,
        ) * charlson_or_scale
        self._charlson_rel = None

        self.tracking = {
            "deaths_by_age": [],
            "hosp_by_age": [],
        }
        self._agent_ages = None
        self._agent_charlson = None
        self._vaccine_death_factor = None
        self._total_transitioning = 0
        self._total_transitioning_by_age = None

        cm = self._compartment_model
        if cm is not None:
            self._severe_state_indices = cm.infectious_indices
            self._has_mortality = cm.has_mortality()
        else:
            self._severe_state_indices = [self.INFECTED_VAR]
            self._has_mortality = True

    def _load_compartment_model(self):
        """Load CompartmentModel from config if available."""
        meta = self.config["simulation_metadata"]
        preset = meta.get("pathogen_preset")
        compartment_config = meta.get("compartment_config")

        if preset is not None or compartment_config is not None:
            from danish_abm.compartments import CompartmentModel
            if compartment_config:
                self._compartment_model = CompartmentModel.from_dict(compartment_config)
            else:
                self._compartment_model = CompartmentModel.from_preset(preset)

            from danish_abm import runtime_set
            runtime_set(self.config, "compartment_model", self._compartment_model)

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        timestep_tensor = torch.tensor([timestep])
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)

        return one_hot_tensor.to(self.device)

    def _get_transitioning_infected_count(self, t, current_stages, current_transition_times):
        """Count agents in severe/infectious states transitioning out today."""
        mask = torch.zeros_like(current_stages, dtype=torch.float32)
        for idx in self._severe_state_indices:
            mask = mask + (current_stages == idx).float()
        mask = mask.clamp(max=1.0) * (current_transition_times <= t).float()
        return mask.sum()

    def _get_transitioning_mask(self, current_stages, current_transition_times, t):
        """Soft mask for agents transitioning out of severe/infectious states."""
        mask = torch.zeros_like(current_stages, dtype=torch.float32)
        for idx in self._severe_state_indices:
            mask = mask + (current_stages == idx).float()
        return (mask.clamp(max=1.0) * (current_transition_times <= t).float()).squeeze()

    def _load_agent_ages(self, state):
        if self._agent_ages is None:
            try:
                ages = get_by_path(state, ["agents", "citizens", "age"])
                self._agent_ages = ages.squeeze().long().to(self.device)
            except Exception:
                pass
            if self._agent_ages is not None and self._age_rel_ifr is None:
                n_age = len(self._age_ifr_raw)
                ages_clamped = self._agent_ages.clamp(0, n_age - 1)
                pop_counts = torch.zeros(n_age, device=self.device)
                pop_counts.scatter_add_(0, ages_clamped, torch.ones_like(ages_clamped, dtype=torch.float32))
                pop_frac = pop_counts / pop_counts.sum().clamp(min=1)
                ifr_pop_mean = (self._age_ifr_raw * pop_frac).sum()
                self._age_rel_ifr = self._age_ifr_raw / ifr_pop_mean.clamp(min=1e-9)
                self._age_rel_ihr = self._age_ihr_raw
                ihr_pop_mean = (self._age_ihr_raw * pop_frac).sum()
                print(f"[diag] age IFR pop-weighted mean={ifr_pop_mean:.4f} "
                      f"(unweighted={self._age_ifr_raw.mean():.4f}), "
                      f"age dist: {pop_frac.cpu().tolist()}")
                print(f"[diag] age IHR (absolute, no normalization) "
                      f"pop-weighted mean={ihr_pop_mean:.4f}, "
                      f"values: {self._age_rel_ihr.cpu().tolist()}")
        if self._agent_charlson is None:
            for key in ["charlson_comorbidity_index", "comorbidity_index", "comorbidity", "charlson"]:
                try:
                    comorb = get_by_path(state, ["agents", "citizens", key])
                    self._agent_charlson = comorb.squeeze().long().clamp(0, self._n_charlson_groups - 1).to(self.device)
                    break
                except Exception:
                    continue
            if self._agent_charlson is not None and self._charlson_rel is None:
                pop_counts = torch.zeros(self._n_charlson_groups, device=self.device)
                pop_counts.scatter_add_(0, self._agent_charlson,
                                        torch.ones_like(self._agent_charlson, dtype=torch.float32))
                pop_frac = pop_counts / pop_counts.sum().clamp(min=1)
                or_pop_mean = (self._charlson_raw * pop_frac).sum()
                self._charlson_rel = self._charlson_raw / or_pop_mean.clamp(min=1e-9)
                print(f"[diag] Charlson pop-weighted mean OR={or_pop_mean:.4f} "
                      f"(unweighted={self._charlson_raw.mean():.4f}), "
                      f"CCI dist: {pop_frac.cpu().tolist()}")
            elif self._agent_charlson is None and self._charlson_rel is None:
                self._charlson_rel = self._charlson_raw / self._charlson_raw.mean().clamp(min=1e-9)

    def update_daily_deaths(
        self, t, daily_dead, current_stages, current_transition_times, state=None
    ):
        """Count expected deaths from newly infected agents (E→I transitions)."""
        if not self._has_mortality:
            return daily_dead

        if self.calibration_mode:
            M_rate = self.calibrate_M
        else:
            M_rate = self.learnable_args["M"]

        sev = self._get_severity_modifier()

        newly_infected_mask = (
            (current_stages.squeeze() == self.EXPOSED_VAR)
            & (current_transition_times.squeeze() <= t)
        ).float()

        if sev is not None and newly_infected_mask.sum() > 0:
            mean_sev = (sev.squeeze() * newly_infected_mask).sum() / newly_infected_mask.sum().clamp(min=1)
            M_rate = M_rate * mean_sev

        if state is not None:
            self._load_agent_ages(state)

        if newly_infected_mask.sum() > 0:
            per_agent_rate = self._per_agent_severity_rate(M_rate, self._age_rel_ifr)
            if per_agent_rate is not None:
                num_dead_today = (per_agent_rate * newly_infected_mask).sum()
            else:
                num_dead_today = newly_infected_mask.sum() * M_rate

            death_day = t + self._death_delay
            if death_day <= self.num_timesteps - 1:
                self._pending_deaths.append((death_day, num_dead_today))
            else:
                self._dropped_pending_deaths = (
                    self._dropped_pending_deaths + float(num_dead_today))

        with torch.no_grad():
            n_newly_inf = newly_infected_mask.sum().item()
            self._total_transitioning += n_newly_inf
            if self._agent_ages is not None:
                if self._total_transitioning_by_age is None:
                    self._total_transitioning_by_age = torch.zeros(8, device=self.device)
                ages_t = self._agent_ages.clamp(0, 7)
                self._total_transitioning_by_age.scatter_add_(
                    0, ages_t, newly_infected_mask
                )

        still_pending = []
        for day, n_d in self._pending_deaths:
            if day <= t:
                daily_dead = (
                    daily_dead
                    + self._generate_one_hot_tensor(int(day), self.num_timesteps) * n_d
                )
            else:
                still_pending.append((day, n_d))
        self._pending_deaths = still_pending

        if t == self.num_timesteps - 1 and self._pending_deaths:
            self._dropped_pending_deaths = sum(n for _, n in self._pending_deaths)
            self._pending_deaths = []

        return daily_dead

    def update_daily_hosp(
        self, t, daily_hosp, current_stages, current_transition_times, state=None
    ):
        """Count expected hospitalizations from newly infected agents (E→I)."""
        if self.calibration_mode:
            H_rate = self.calibrate_H
        else:
            H_rate = self.learnable_args["H"]

        sev = self._get_severity_modifier()

        newly_infected_mask = (
            (current_stages.squeeze() == self.EXPOSED_VAR)
            & (current_transition_times.squeeze() <= t)
        ).float()

        if sev is not None and newly_infected_mask.sum() > 0:
            mean_sev = (sev.squeeze() * newly_infected_mask).sum() / newly_infected_mask.sum().clamp(min=1)
            H_rate = H_rate * mean_sev

        if state is not None:
            self._load_agent_ages(state)

        if newly_infected_mask.sum() > 0:
            per_agent_rate = self._per_agent_severity_rate(H_rate, self._age_rel_ihr)
            if per_agent_rate is not None:
                num_hosp_today = (per_agent_rate * newly_infected_mask).sum()
            else:
                num_hosp_today = newly_infected_mask.sum() * H_rate

            hosp_day = t + self.hosp_delay
            if hosp_day <= self.num_timesteps - 1:
                self._pending_hosp.append((hosp_day, num_hosp_today))
            else:
                self._dropped_pending_hosp = (
                    self._dropped_pending_hosp + float(num_hosp_today))

        still_pending = []
        for day, n_h in self._pending_hosp:
            if day <= t:
                daily_hosp = (
                    daily_hosp
                    + self._generate_one_hot_tensor(int(day), self.num_timesteps) * n_h
                )
            else:
                still_pending.append((day, n_h))
        self._pending_hosp = still_pending

        if t == self.num_timesteps - 1 and self._pending_hosp:
            self._dropped_pending_hosp = sum(n for _, n in self._pending_hosp)
            self._pending_hosp = []

        return daily_hosp

    def update_current_stages(self, t, current_stages, current_transition_times):
        cm = self._compartment_model

        if cm is not None and not cm.is_seirm_compatible:
            return self._update_stages_generic(t, current_stages, current_transition_times, cm)

        transit_agents = (current_transition_times <= t) * self.STAGE_UPDATE_VAR

        stage_transition = (current_stages == self.EXPOSED_VAR) * transit_agents + (
            current_stages == self.INFECTED_VAR
        ) * transit_agents

        new_stages = current_stages + stage_transition

        infected_exiting = (
            (current_stages.squeeze() == self.INFECTED_VAR)
            & (current_transition_times.squeeze() <= t)
        )
        self._latest_die_mask = None
        if infected_exiting.any() and self._has_mortality:
            if self.calibration_mode:
                M_rate_val = float(self.calibrate_M)
            else:
                M_rate_val = float(self.learnable_args["M"])
            exit_idx = infected_exiting.nonzero(as_tuple=True)[0]
            n_exit = exit_idx.shape[0]
            p_death_sparse = torch.full((n_exit,), M_rate_val, device=self.device)
            if self._agent_ages is not None and self._age_rel_ifr is not None:
                ages = self._agent_ages[exit_idx].clamp(0, len(self._age_rel_ifr) - 1)
                p_death_sparse = p_death_sparse * self._age_rel_ifr[ages]
            if self._agent_charlson is not None and self._charlson_rel is not None:
                charlson = self._agent_charlson[exit_idx].clamp(0, self._n_charlson_groups - 1)
                p_death_sparse = p_death_sparse * self._charlson_rel[charlson]
            if self._vaccine_death_factor is not None:
                p_death_sparse = p_death_sparse * self._vaccine_death_factor[exit_idx]
            p_death_sparse = p_death_sparse.clamp(0, 1)
            die_sparse = torch.rand(n_exit, device=self.device) < p_death_sparse
            die = torch.zeros_like(infected_exiting)
            die[exit_idx] = die_sparse
            self._latest_die_mask = die
            die_add = die.to(new_stages.dtype)
            if new_stages.dim() > die_add.dim():
                die_add = die_add.unsqueeze(-1)
            new_stages = new_stages + die_add

        waning_mask = (
            (current_stages == self.RECOVERED_VAR)
            * (current_transition_times <= t)
            * (current_transition_times < self.INFINITY_TIME)
        )
        new_stages = new_stages - waning_mask.float() * self.RECOVERED_VAR

        if self._previously_infected is None:
            self._previously_infected = torch.zeros_like(current_stages)
        self._previously_infected = torch.max(
            self._previously_infected,
            (current_stages >= self.RECOVERED_VAR).float() * (current_stages < self.MORTALITY_VAR).float()
        )

        return new_stages

    def _update_stages_generic(self, t, current_stages, current_transition_times, cm):
        """Generic compartment progression for arbitrary graph structures."""
        new_stages = current_stages.clone()
        transit_mask = (current_transition_times <= t).float()

        for state_name, branches in cm._branching.items():
            state_idx = cm.idx(state_name)
            in_state = (current_stages == state_idx).float()
            transitioning = in_state * transit_mask

            if len(branches) == 1:
                trans, _ = branches[0]
                to_idx = cm.idx(trans.to_state)
                delta = to_idx - state_idx
                new_stages = new_stages + transitioning * delta
            else:
                for trans, prob in branches:
                    to_idx = cm.idx(trans.to_state)
                    delta = to_idx - state_idx
                    new_stages = new_stages + transitioning * delta * prob

        if self._previously_infected is None:
            self._previously_infected = torch.zeros_like(current_stages)
        inf_mask = cm.infectious_mask(current_stages)
        self._previously_infected = torch.max(self._previously_infected, inf_mask)

        return new_stages

    def _init_distributional_params(self, n_agents):
        """Lazily initialise per-agent disease timing."""
        cm = self._compartment_model
        if cm is not None:
            self._per_agent_transition_times = cm.get_transition_times(
                n_agents, device=self.device
            )
            key_ir = "I->R"
            key_waning = None
            for k in self._per_agent_transition_times:
                if k.startswith("I->R") or k == "I->R":
                    key_ir = k
                if "waning" in k or k.startswith("R->S"):
                    key_waning = k
            self._per_agent_infectious = self._per_agent_transition_times.get(
                key_ir, torch.full((n_agents,), self.INFECTED_TO_RECOVERED_TIME, device=self.device)
            )
            if key_waning and key_waning in self._per_agent_transition_times:
                self._per_agent_waning = self._per_agent_transition_times[key_waning]
            else:
                self._per_agent_waning = torch.full(
                    (n_agents,), self.RECOVERED_TO_SUSCEPTIBLE_TIME, device=self.device
                )
        else:
            from danish_abm.disease_params import DiseaseParams
            dp_config = self.config["simulation_metadata"].get("disease_params", {})
            dp = DiseaseParams.from_ci(**dp_config)
            self._per_agent_infectious = dp.sample_infectious(n_agents, device=self.device)
            self._per_agent_waning = dp.sample_waning(n_agents, device=self.device)

    def update_next_transition_times(self, t, current_stages, current_transition_times):
        cm = self._compartment_model

        if cm is not None and not cm.is_seirm_compatible:
            return self._update_transition_times_generic(
                t, current_stages, current_transition_times, cm
            )

        new_transition_times = torch.clone(current_transition_times).to(
            current_transition_times.device
        )
        curr_stages = torch.clone(current_stages).to(current_stages.device)

        if self._distributional and self._per_agent_infectious is None:
            self._init_distributional_params(int(curr_stages.shape[0]))

        infected_transitioning = (curr_stages == self.INFECTED_VAR) * (current_transition_times <= t) * (current_transition_times > t - 1)
        if self._distributional:
            waning_times = self._per_agent_waning.view_as(new_transition_times)
            new_transition_times[infected_transitioning] = (
                t + waning_times[infected_transitioning]
            )
        else:
            new_transition_times[infected_transitioning] = (
                t + self.RECOVERED_TO_SUSCEPTIBLE_TIME
            )

        if self._latest_die_mask is not None and self._latest_die_mask.any():
            dm = self._latest_die_mask
            if dm.dim() < new_transition_times.dim():
                dm = dm.unsqueeze(-1)
            new_transition_times = torch.where(
                dm, torch.tensor(self.INFINITY_TIME, device=self.device,
                                 dtype=new_transition_times.dtype),
                new_transition_times
            )

        exposed_transitioning = (curr_stages == self.EXPOSED_VAR) * (current_transition_times <= t) * (current_transition_times > t - 1)
        if self._distributional:
            infectious_times = self._per_agent_infectious.view_as(new_transition_times)
            new_transition_times[exposed_transitioning] = (
                t + infectious_times[exposed_transitioning]
            )
        else:
            new_transition_times[exposed_transitioning] = (
                t + self.INFECTED_TO_RECOVERED_TIME
            )

        waned = (
            (curr_stages == self.RECOVERED_VAR)
            * (current_transition_times <= t)
            * (current_transition_times < self.INFINITY_TIME)
        )
        new_transition_times[waned] = self.INFINITY_TIME

        del current_transition_times
        del current_stages

        return new_transition_times

    def _update_transition_times_generic(self, t, current_stages, current_transition_times, cm):
        """Generic transition time updates for arbitrary compartment graphs."""
        new_transition_times = torch.clone(current_transition_times).to(
            current_transition_times.device
        )
        curr_stages = torch.clone(current_stages).to(current_stages.device)
        n_agents = int(curr_stages.shape[0])

        if self._distributional and self._per_agent_transition_times is None:
            self._init_distributional_params(n_agents)

        for state_name, branches in cm._branching.items():
            state_idx = cm.idx(state_name)
            transitioning = (curr_stages == state_idx) * (current_transition_times <= t) * (current_transition_times > t - 1)

            if not transitioning.any():
                continue

            for trans, prob in branches:
                to_state = trans.to_state
                to_branches = cm._branching.get(to_state, [])
                if to_branches:
                    next_trans = to_branches[0][0]
                    tkey = f"{to_state}->{next_trans.to_state}"
                    if (
                        self._distributional
                        and self._per_agent_transition_times is not None
                        and tkey in self._per_agent_transition_times
                    ):
                        dur = self._per_agent_transition_times[tkey].view_as(new_transition_times)
                        new_transition_times[transitioning] = t + dur[transitioning]
                    elif next_trans.duration is not None:
                        new_transition_times[transitioning] = t + round(next_trans.duration.mean)
                    else:
                        new_transition_times[transitioning] = t + 1
                else:
                    new_transition_times[transitioning] = self.INFINITY_TIME

        del current_transition_times
        del current_stages

        return new_transition_times

    def _resolve_strain_refs(self):
        """Cache the strain manager, immunity tracker and strain config."""
        if getattr(self, "_strain_refs_resolved", False):
            return
        self._strain_refs_resolved = True
        self._strain_manager_ref = None
        self._immunity_tracker_ref = None
        self._strain_config_ref = None

        try:
            from danish_abm import runtime_get
            self._strain_manager_ref = runtime_get(self.config, "strain_manager")
            self._immunity_tracker_ref = runtime_get(self.config, "immunity_tracker")
            self._strain_config_ref = runtime_get(self.config, "strain_config")
        except Exception:
            pass

    def _update_multi_strain(self, t, old_stages, new_stages, transition_times):
        self._resolve_strain_refs()

        sm = self._strain_manager_ref
        it = self._immunity_tracker_ref
        if sm is None or it is None:
            return

        cm = self._compartment_model
        if cm is not None:
            old_infectious = cm.infectious_mask(old_stages.squeeze()).bool()
            removed_mask = torch.zeros_like(old_stages.squeeze(), dtype=torch.bool)
            for idx in cm.removed_indices:
                removed_mask = removed_mask | (new_stages.squeeze() == idx)
            new_infectious = cm.infectious_mask(new_stages.squeeze()).bool()
            recovered_mask = old_infectious & ~new_infectious & ~removed_mask
        else:
            recovered_mask = (
                (old_stages.squeeze() == self.INFECTED_VAR)
                & ((new_stages.squeeze() == self.RECOVERED_VAR)
                   | (new_stages.squeeze() == self.MORTALITY_VAR))
            )

        if not recovered_mask.any():
            return

        for s in sm.config.strains:
            s_idx = sm.config.strain_idx[s.name]
            strain_mask = recovered_mask & (sm.current_strain == s_idx)
            if strain_mask.any():
                it.record_event(strain_mask, s.name, time=t, initial_efficacy=1.0)

        sm.clear_recovered(recovered_mask)

    def _get_severity_modifier(self):
        self._resolve_strain_refs()
        sm = self._strain_manager_ref
        if sm is not None:
            return sm.get_severity_modifier()
        return None

    def _track_age_stratified(self, t, current_stages, current_transition_times, state):
        if self._agent_ages is None:
            self._load_agent_ages(state)
        if self._agent_ages is None:
            return

        n_age_groups = 8
        transitioning_mask = (
            (current_stages.squeeze() == self.EXPOSED_VAR)
            & (current_transition_times.squeeze() <= t)
        )

        if not transitioning_mask.any():
            self.tracking["deaths_by_age"].append([0] * n_age_groups)
            self.tracking["hosp_by_age"].append([0] * n_age_groups)
            return

        if self.calibration_mode:
            M_rate = float(self.calibrate_M)
            H_rate = float(self.calibrate_H)
        else:
            M_rate = float(self.learnable_args["M"])
            H_rate = float(self.learnable_args["H"])

        ages_transitioning = self._agent_ages[transitioning_mask].clamp(0, n_age_groups - 1)
        age_counts = torch.zeros(n_age_groups, device=self.device)
        age_counts.scatter_add_(0, ages_transitioning,
                                torch.ones_like(ages_transitioning, dtype=torch.float32))

        vdf_by_age = torch.ones(n_age_groups, device=self.device)
        if self._vaccine_death_factor is not None:
            for ag in range(n_age_groups):
                ag_mask = transitioning_mask & (self._agent_ages.clamp(0, n_age_groups - 1) == ag)
                if ag_mask.any():
                    vdf_by_age[ag] = self._vaccine_death_factor[ag_mask].mean()

        if self._agent_charlson is not None and self._charlson_rel is not None:
            charlson_by_age = torch.ones(n_age_groups, device=self.device)
            for ag in range(n_age_groups):
                ag_mask = transitioning_mask & (self._agent_ages.clamp(0, n_age_groups - 1) == ag)
                if ag_mask.any():
                    charlson_by_age[ag] = self._charlson_rel[self._agent_charlson[ag_mask]].mean()
            deaths_by_age = (age_counts * M_rate * self._age_rel_ifr * charlson_by_age * vdf_by_age).cpu().tolist()
            hosp_by_age = (age_counts * H_rate * self._age_rel_ihr * charlson_by_age * vdf_by_age).cpu().tolist()
        else:
            deaths_by_age = (age_counts * M_rate * self._age_rel_ifr * vdf_by_age).cpu().tolist()
            hosp_by_age = (age_counts * H_rate * self._age_rel_ihr * vdf_by_age).cpu().tolist()

        self.tracking["deaths_by_age"].append(deaths_by_age)
        self.tracking["hosp_by_age"].append(hosp_by_age)

    def _resolve_vaccine_death_factor(self):
        """Per-agent VE against death/admission, resolved lazily once per episode."""
        if self._vaccine_death_factor is None:
            try:
                from danish_abm import runtime_get
                self._vaccine_death_factor = runtime_get(self.config, "vaccine_death_factor")
            except Exception:
                pass
        return self._vaccine_death_factor

    def _per_agent_severity_rate(self, base_rate, age_table):
        """Per-agent admission or death rate: the calibrated scalar times the agent's
        own age-relative risk, comorbidity odds ratio and vaccine protection.
        """
        vdf = self._resolve_vaccine_death_factor()
        if self._agent_ages is None or age_table is None:
            return None
        ages = self._agent_ages.clamp(0, len(age_table) - 1)
        rate = base_rate * age_table[ages]
        if self._agent_charlson is not None and self._charlson_rel is not None:
            charlson = self._agent_charlson.clamp(0, self._n_charlson_groups - 1)
            rate = rate * self._charlson_rel[charlson]
        if vdf is not None:
            rate = rate * vdf
        return rate

    def _seed_severity_modifier(self):
        """Strain severity for the PRE-WINDOW cohort: the day-0 strain mix."""
        self._resolve_strain_refs()
        cfg = self._strain_config_ref
        if cfg is None:
            return None
        try:
            props = dict(getattr(cfg, "initial_proportions", None) or {})
            if not props or not cfg.strains:
                return None
            rest = 1.0 - float(sum(props.values()))
            if rest > 1e-9:
                first = cfg.strains[0].name
                props[first] = props.get(first, 0.0) + rest
            tot = float(sum(props.values()))
            if tot <= 0:
                return None
            sev = sum(float(props.get(st.name, 0.0)) * float(st.severity)
                      for st in cfg.strains)
            return sev / tot
        except Exception as e:
            print(f"  [prime] could not read the day-0 strain mix ({type(e).__name__}: {e}); "
                  f"priming without a severity modifier")
            return None

    def _prime_pending_queues(self, state):
        """Fill the admission and death queues with the pre-window E->I cohort."""
        import os
        if getattr(self, "_pending_seed_missing", False):
            return
        meta = self.config["simulation_metadata"]
        pop_dir = meta.get("population_dir")
        path = os.path.join(str(pop_dir), "seed_pending.csv") if pop_dir else None
        if not path or not os.path.exists(path):
            self._pending_seed_missing = True
            print(f"  [prime] ** no seed_pending.csv under {pop_dir}: the admission and "
                  f"death queues start EMPTY, so the first "
                  f"{max(self.hosp_delay, self._death_delay)} days of the window are an "
                  f"artefact, not model output **")
            return
        if self._pending_seed is None:
            import pandas as pd
            df = pd.read_csv(path)
            n_ag = int(meta.get("num_agents") or 0)
            ids = df["agent_id"].astype(int).to_numpy()
            leads = df["lead_day"].astype(int).to_numpy()
            if n_ag > 0:
                keep = (ids >= 0) & (ids < n_ag)
                if not keep.all():
                    print(f"  [prime] dropped {int((~keep).sum()):,} of {len(ids):,} seeded "
                          f"ids outside [0, {n_ag}) -- the register and the population are "
                          f"not on the same id space")
                ids, leads = ids[keep], leads[keep]
            self._pending_seed = [
                (int(d), torch.as_tensor(ids[leads == d], dtype=torch.long, device=self.device))
                for d in sorted(set(leads.tolist()))
            ]
            print(f"  [prime] {len(ids):,} pre-window E->I agents over "
                  f"{len(self._pending_seed)} lead days from {path}")

        if not self._pending_seed:
            return
        if self._agent_ages is None:
            print(f"  [prime] ** the population's ages are not loaded, so the queues are "
                  f"NOT primed and the first {max(self.hosp_delay, self._death_delay)} "
                  f"days of the window are an artefact **")
            return
        M_rate = self.calibrate_M if self.calibration_mode else self.learnable_args["M"]
        H_rate = self.calibrate_H if self.calibration_mode else self.learnable_args["H"]
        sev_mean = self._seed_severity_modifier()
        if sev_mean is not None:
            M_rate = M_rate * sev_mean
            H_rate = H_rate * sev_mean
        rate_d = self._per_agent_severity_rate(M_rate, self._age_rel_ifr)
        rate_h = self._per_agent_severity_rate(H_rate, self._age_rel_ihr)
        n_primed_d = n_primed_h = 0.0
        n_agents_primed = 0
        for lead, idx in self._pending_seed:
            if idx.numel() == 0:
                continue
            n_agents_primed += int(idx.numel())
            if self._has_mortality:
                day = self._death_delay - lead
                if 0 <= day <= self.num_timesteps - 1:
                    n_d = (rate_d[idx].sum() if rate_d is not None
                           else idx.numel() * M_rate)
                    self._pending_deaths.append((day, n_d))
                    n_primed_d += float(n_d)
            day = self.hosp_delay - lead
            if 0 <= day <= self.num_timesteps - 1:
                n_h = (rate_h[idx].sum() if rate_h is not None
                       else idx.numel() * H_rate)
                self._pending_hosp.append((day, n_h))
                n_primed_h += float(n_h)
        self._primed_totals = {"deaths": n_primed_d, "hosp": n_primed_h,
                               "agents": n_agents_primed,
                               "severity_modifier": (float(sev_mean) if sev_mean else None)}
        print(f"  [prime] queues primed from {n_agents_primed:,} agents: "
              f"{n_primed_d:,.0f} deaths over days "
              f"{max(self._death_delay - max(d for d, _ in self._pending_seed), 0)}"
              f"-{self._death_delay - 1}, {n_primed_h:,.0f} admissions over days "
              f"{max(self.hosp_delay - max(d for d, _ in self._pending_seed), 0)}"
              f"-{self.hosp_delay - 1}"
              + (f", strain severity x{sev_mean:.3f}" if sev_mean else ""))

    def forward(self, state, action):
        """Update stage and transition times for already infected agents"""
        input_variables = self.input_variables
        t = state["current_step"]

        if t == 0:
            self._pending_deaths = []
            self._pending_hosp = []
            self._dropped_pending_deaths = 0
            self._dropped_pending_hosp = 0
            self._latest_die_mask = None
            self._total_transitioning = 0
            self._total_transitioning_by_age = None

        self._load_agent_ages(state)

        if t == 0:
            with torch.no_grad():
                self._prime_pending_queues(state)

        current_stages = get_by_path(
            state, re.split("/", input_variables["disease_stage"])
        ).detach()
        current_transition_times = get_by_path(
            state, re.split("/", input_variables["next_stage_time"])
        ).detach()
        daily_deaths = get_by_path(
            state, re.split("/", input_variables["daily_deaths"])
        ).detach()
        daily_hosp = get_by_path(
            state, re.split("/", input_variables["daily_hosp"])
        ).detach()

        if t % 10 == 0 or t <= 2:
            with torch.no_grad():
                cs = current_stages.squeeze()
                ct = current_transition_times.squeeze()
                n_S = (cs == self.SUSCEPTIBLE_VAR).sum().item()
                n_E = (cs == self.EXPOSED_VAR).sum().item()
                n_I = (cs == self.INFECTED_VAR).sum().item()
                n_R = (cs == self.RECOVERED_VAR).sum().item()
                n_D = (cs == self.MORTALITY_VAR).sum().item() if hasattr(self, 'MORTALITY_VAR') else 0
                e_mask = (cs == self.EXPOSED_VAR)
                e_ready = (e_mask & (ct <= t)).sum().item()
                e_future = (e_mask & (ct > t) & (ct < 500)).sum().item()
                e_inf = (e_mask & (ct >= 500)).sum().item()
                i_mask = (cs == self.INFECTED_VAR)
                i_ready = (i_mask & (ct <= t)).sum().item()
                n_total = n_S + n_E + n_I + n_R + n_D
                n_expected = cs.shape[0]
                n_missing = n_expected - n_total
                extra = ""
                if n_missing > 0:
                    known = {0, 1, 2, 3, 4}
                    for s in [-3, -2, -1, 5, 6, 7]:
                        c = (cs == s).sum().item()
                        if c > 0:
                            extra += f" stg{s}={c}"
                    int_cs = cs.round()
                    non_int = (cs != int_cs).sum().item()
                    if non_int > 0:
                        extra += f" non_int={non_int}"
                    nan_count = cs.isnan().sum().item()
                    if nan_count > 0:
                        extra += f" NaN={nan_count}"
                print(f"  [dp t={t:3d}] S={n_S} E={n_E}(ready={e_ready},future={e_future},inf={e_inf}) "
                      f"I={n_I}(exit={i_ready}) R={n_R} D={n_D} miss={n_missing}{extra} | cum_EI={self._total_transitioning:.0f}",
                      flush=True)
        if t == self.num_timesteps - 1 and (self._dropped_pending_deaths
                                            or self._dropped_pending_hosp):
            print(f"  [dp] window closed with {self._dropped_pending_deaths:.0f} deaths and "
                  f"{self._dropped_pending_hosp:.0f} admissions still in the delay queue, "
                  f"scheduled past day {self.num_timesteps - 1}. Not counted: they fall "
                  f"outside the observation window, as they do in the register.", flush=True)

        new_stages = self.update_current_stages(
            t, current_stages, current_transition_times
        )
        new_transition_times = self.update_next_transition_times(
            t, current_stages, current_transition_times
        )

        new_daily_deaths = self.update_daily_deaths(
            t, daily_deaths, current_stages, current_transition_times, state=state
        )
        new_daily_hosp = self.update_daily_hosp(
            t, daily_hosp, current_stages, current_transition_times, state=state
        )

        if self._multi_strain:
            self._update_multi_strain(t, current_stages, new_stages, current_transition_times)

        with torch.no_grad():
            self._track_age_stratified(t, current_stages, current_transition_times, state)

        return {
            self.output_variables[0]: new_stages.detach(),
            self.output_variables[1]: new_transition_times.detach(),
            self.output_variables[2]: new_daily_deaths.detach(),
            self.output_variables[3]: new_daily_hosp.detach(),
        }
