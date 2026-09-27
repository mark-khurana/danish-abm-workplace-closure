"""Testing and isolation substep — detects infections and triggers isolation."""

import torch
import re

from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_by_path


_TEST_SCHEDULE_CACHE = {}


class TestingAndIsolation(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.n_agents = self.config["simulation_metadata"]["num_agents"]
        self.num_timesteps = self.config["simulation_metadata"]["num_steps_per_episode"]

        self.enabled = self.config["simulation_metadata"].get("testing_enabled", False)

        self.test_sensitivity = self.config["simulation_metadata"].get("test_sensitivity", 0.8)
        self.test_specificity = self.config["simulation_metadata"].get("test_specificity", 0.99)

        self._daily_test_counts = None
        self._load_test_schedule()

        self.base_testing_rate = self.config["simulation_metadata"].get("base_testing_rate", 0.015)
        self.symptomatic_multiplier = self.config["simulation_metadata"].get(
            "symptomatic_testing_multiplier", 5.0
        )

        self.isolation_duration = self.config["simulation_metadata"].get("isolation_duration", 10)

        self.isolation_reduction = self.config["simulation_metadata"].get(
            "isolation_reduction", 0.70
        )
        self.isolation_compliance = self.config["simulation_metadata"].get("isolation_compliance", 0.8)
        self.isolation_effectiveness = self.config["simulation_metadata"].get(
            "isolation_effectiveness", 0.1
        )

        self.calibration_mode = self.config["simulation_metadata"].get("calibration", False)
        if self.calibration_mode:
            import torch.nn as nn
            init_logit = torch.log(torch.tensor(self.isolation_reduction / (1.0 - self.isolation_reduction)))
            self.calibrate_isolation_reduction = nn.Parameter(
                torch.tensor(init_logit, dtype=torch.float32)
            )

        self._weekly_testing_rate = None
        weekly_rates = self.config["simulation_metadata"].get("weekly_testing_rates", None)
        if weekly_rates is not None:
            self._weekly_testing_rate = torch.tensor(weekly_rates, dtype=torch.float32, device=self.device)

        self._surveillance_policy_name = self.config["simulation_metadata"].get(
            "surveillance_policy", None
        )
        self._surveillance_total_tests = self.config["simulation_metadata"].get(
            "surveillance_total_daily_tests", 5000
        )
        self._surveillance_threshold = self.config["simulation_metadata"].get(
            "surveillance_prevalence_threshold", 0.01
        )
        self._surveillance_update_freq = self.config["simulation_metadata"].get(
            "surveillance_surrogate_update_frequency", 7
        )
        self._surveillance_policy = None
        self._surveillance_graph = None
        self._surveillance_state = None
        self._surveillance_surrogate = None
        self._municipality_ids = None
        self._n_municipalities = 0
        self._municipality_pop = None

        if self._surveillance_policy_name:
            self._init_surveillance()

        self._is_detected = None
        self._is_isolating = None
        self._isolation_end_day = None
        self._daily_detected = None

        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.INFECTED_VAR = self.config["simulation_metadata"]["INFECTED_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]

        self.tracking = {"daily_tests": [], "daily_positives": [], "daily_isolating": []}

        from danish_abm import runtime_set
        runtime_set(self.config, "testing_tracking", self.tracking)

    def _load_test_schedule(self):
        """Load daily test counts from individual test records (events/tests.csv)."""
        import os
        import pandas as pd
        from datetime import datetime, timedelta

        test_path = self.config["simulation_metadata"].get("individual_test_data_path")
        if not test_path:
            _m = self.config["simulation_metadata"]
            print("  Testing: no individual_test_data_path configured — using ASSUMED "
                  f"base_testing_rate={_m.get('base_testing_rate', 0.015)} "
                  f"x{_m.get('symptomatic_testing_multiplier', 5.0)} for symptomatics, "
                  "NOT real SSI test volumes.")
            return

        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"individual_test_data_path is set to {test_path!r} but that file does "
                "not exist. Refusing to fall back to an assumed testing rate, which would "
                "silently change what the model is fitted to."
            )

        start_date = self.config["simulation_metadata"].get("start_date")
        if not start_date:
            raise KeyError(
                "simulation_metadata['start_date'] is missing, so the test schedule "
                f"in {test_path!r} cannot be placed on the simulation's day axis. "
                "DataConfig.patch_model_config must set it (it sets sim_start_doy "
                "beside it)."
            )

        scale_factor = self.config["simulation_metadata"].get("scale_factor", 1.0)
        cache_key = (str(test_path), str(start_date), int(self.num_timesteps),
                     float(scale_factor))
        cached = _TEST_SCHEDULE_CACHE.get(cache_key)
        if cached is not None:
            self._daily_test_counts = cached.to(self.device)
            return

        try:
            from danish_abm.agent_properties import read_tabular

            def _is_date_col(name):
                return name.strip().lower().replace(" ", "_") in ("date", "test_date", "dato")

            if str(test_path).lower().endswith((".parquet", ".pq")):
                import pyarrow.parquet as pq
                names = list(pq.ParquetFile(test_path).schema_arrow.names)
                date_cols = [c for c in names if _is_date_col(c)]
                if not date_cols:
                    raise KeyError(
                        f"{test_path!r} has no date column. Looked for date/test_date/"
                        f"dato among {names}. Widen the match rather than falling back "
                        "to an assumed testing rate."
                    )
                df = read_tabular(test_path, columns=[date_cols[0]])
            else:
                df = read_tabular(test_path)

            col_map = {}
            for c in df.columns:
                if _is_date_col(c):
                    col_map[c] = "date"
            df = df.rename(columns=col_map)
            if "date" not in df.columns:
                raise KeyError(
                    f"{test_path!r} has no date column among {list(df.columns)}"
                )
            df["date"] = pd.to_datetime(df["date"])

            start_dt = pd.Timestamp(start_date)
            end_dt = start_dt + timedelta(days=self.num_timesteps)

            mask = (df["date"] >= start_dt) & (df["date"] < end_dt)
            daily = df[mask].groupby(df[mask]["date"].dt.date).size()

            counts = torch.zeros(self.num_timesteps, dtype=torch.float32, device=self.device)
            for date_val, count in daily.items():
                day_idx = (pd.Timestamp(date_val) - start_dt).days
                if 0 <= day_idx < self.num_timesteps:
                    counts[day_idx] = float(count)

            if scale_factor > 0 and scale_factor < 1.0:
                counts = counts * scale_factor

            if counts.sum() <= 0:
                raise ValueError(
                    f"{test_path!r} yielded no tests inside the simulation window "
                    f"starting {start_date} ({self.num_timesteps} days). The file "
                    "covers every wave, so an empty window means the dates do not "
                    "overlap -- check start_date, not the file."
                )
            self._daily_test_counts = counts
            _TEST_SCHEDULE_CACHE[cache_key] = counts.detach().cpu()
            _nz = int((counts > 0).sum().item())
            print(f"  Testing: loaded {int(counts.sum().item()):,} test events "
                  f"from {test_path} ({self.num_timesteps} days, {_nz} non-empty, "
                  f"mean {counts.sum().item() / max(self.num_timesteps, 1):,.0f}/day)")
        except Exception as e:
            raise RuntimeError(
                f"could not load the test schedule from {test_path!r}: {e}"
            ) from e

    def _init_surveillance(self):
        """Initialise surveillance policy, graph, and surrogate model."""
        import os
        import numpy as np
        from danish_abm.surveillance import (
            MunicipalityGraph, SurveillanceState, SurrogateModel,
            get_surveillance_policy,
        )

        pop_dir = self.config["simulation_metadata"]["population_dir"]

        from danish_abm.substeps.transmission.transition import _load_municipality_array
        muni_arr = _load_municipality_array(pop_dir)
        if muni_arr is None:
            self._surveillance_policy_name = None
            return

        unique_codes = sorted(set(int(x) for x in muni_arr))
        code_to_idx = {c: i for i, c in enumerate(unique_codes)}
        self._n_municipalities = len(unique_codes)
        idx_arr = [code_to_idx[int(x)] for x in muni_arr]
        self._municipality_ids = torch.tensor(idx_arr, dtype=torch.long, device=self.device)

        self._municipality_pop = torch.zeros(self._n_municipalities, device=self.device)
        ones = torch.ones(self.n_agents, device=self.device)
        self._municipality_pop.scatter_add_(0, self._municipality_ids, ones)

        network_dir = os.path.join(pop_dir, "mobility_networks")
        network_csv = os.path.join(network_dir, "0.csv")
        muni_pickle_path = os.path.join(pop_dir, "municipality.pickle")
        if os.path.exists(network_csv) and os.path.exists(muni_pickle_path):
            self._surveillance_graph = MunicipalityGraph.from_pickle(muni_pickle_path, network_csv)
        else:
            adj = torch.ones(self._n_municipalities, self._n_municipalities)
            adj.fill_diagonal_(0)
            pop_cpu = self._municipality_pop.cpu()
            self._surveillance_graph = MunicipalityGraph(
                self._n_municipalities, adj, pop_cpu, unique_codes,
            )

        self._surveillance_policy = get_surveillance_policy(self._surveillance_policy_name)

        self._surveillance_state = SurveillanceState(
            n_municipalities=self._n_municipalities,
            prevalence_threshold=self._surveillance_threshold,
        )
        self._surveillance_surrogate = SurrogateModel(self._surveillance_graph)

    def _init_state(self):
        self._is_detected = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self._is_isolating = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self._isolation_end_day = torch.full(
            (self.n_agents,), -1, dtype=torch.long, device=self.device
        )
        self._daily_detected = torch.zeros(self.num_timesteps, device=self.device)
        self._seed_pre_window_detections()

    def _seed_pre_window_detections(self):
        """Carry the pre-window positives' detection and isolation into day 0."""
        import os
        pop_dir = self.config["simulation_metadata"].get("population_dir")
        path = os.path.join(str(pop_dir), "seed_pending.csv") if pop_dir else None
        if not path or not os.path.exists(path):
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            ids = df["agent_id"].astype(int).to_numpy()
            leads = df["lead_day"].astype(int).to_numpy()
            keep = (ids >= 0) & (ids < self.n_agents)
            ids, leads = ids[keep], leads[keep]
            if not len(ids):
                return
            idx = torch.as_tensor(ids, dtype=torch.long, device=self.device)
            end = torch.as_tensor(self.isolation_duration - leads, dtype=torch.long,
                                  device=self.device)
            self._is_detected[idx] = True
            still = end > 0
            self._is_isolating[idx[still]] = True
            self._isolation_end_day[idx[still]] = end[still]
            print(f"  [testing] {len(ids):,} pre-window positives start detected, "
                  f"{int(still.sum()):,} of them still isolating, so they are not "
                  f"re-tested and re-counted as new cases inside the window")
        except Exception as e:
            print(f"  [testing] ** could not seed pre-window detections from {path} "
                  f"({type(e).__name__}: {e}); pre-window positives will be re-counted as "
                  f"new cases **")

    def _get_testing_rate(self, t):
        """Get the testing rate for day t."""
        if self._weekly_testing_rate is not None:
            week = min(t // 7, len(self._weekly_testing_rate) - 1)
            return self._weekly_testing_rate[week].item()
        return self.base_testing_rate

    def _apply_surveillance_allocation(self, test_prob, t, tested_mask):
        """Modulate per-agent testing probability using surveillance policy allocation."""
        if self._surveillance_policy is None or self._municipality_ids is None:
            return test_prob

        allocation = self._surveillance_policy.allocate(
            self._surveillance_state,
            self._surveillance_total_tests,
            self._surveillance_graph,
        )

        muni_rates = torch.zeros(self._n_municipalities, device=self.device)
        for muni_id, n_tests in allocation.items():
            pop = self._municipality_pop[muni_id].item()
            if pop > 0:
                muni_rates[muni_id] = min(n_tests / pop, 1.0)

        surveillance_rate = muni_rates[self._municipality_ids]

        test_prob = surveillance_rate

        return test_prob

    def _update_surveillance_state(self, t, tested, test_positive):
        """Update the surveillance state and surrogate model with test results."""
        if self._surveillance_state is None or self._municipality_ids is None:
            return

        observations = {}
        for muni_id in range(self._n_municipalities):
            mask = self._municipality_ids == muni_id
            n_tests = int((tested & mask).sum().item())
            n_positives = int((test_positive & mask).sum().item())
            if n_tests > 0:
                observations[muni_id] = (n_tests, n_positives)

        self._surveillance_state.update(observations)

        if t % self._surveillance_update_freq == 0:
            self._surveillance_surrogate.update(observations)
            self._surveillance_state.update_beliefs(
                self._surveillance_surrogate.predict(),
                self._surveillance_surrogate.uncertainty(),
            )

    def forward(self, state, action):
        if not self.enabled:
            return {}

        t = int(state["current_step"])

        disease_stages = get_by_path(
            state, re.split("/", self.input_variables["disease_stage"])
        ).squeeze()

        actual_n = disease_stages.shape[0]
        if actual_n != self.n_agents:
            self.n_agents = actual_n

        if self._is_detected is None or t == 0:
            self._init_state()

        expired = self._is_isolating & (self._isolation_end_day <= t)
        self._is_isolating[expired] = False

        is_symptomatic = (disease_stages == self.INFECTED_VAR)

        if self._daily_test_counts is not None:
            n_target = int(self._daily_test_counts[min(t, len(self._daily_test_counts) - 1)].item())
            n_eligible = int((~self._is_isolating).sum().item())
            if n_target <= 0 or n_eligible <= 0:
                test_prob = torch.zeros(self.n_agents, device=self.device)
            else:
                eligible = ~self._is_isolating
                test_prob = eligible.float()
                test_prob[is_symptomatic & eligible] = 5.0
                total_weight = test_prob.sum()
                if total_weight > 0:
                    test_prob.mul_(min(n_target, n_eligible) / total_weight.item())
                    test_prob.clamp_(0.0, 1.0)
        else:
            testing_rate = self._get_testing_rate(t)
            test_prob = torch.full((self.n_agents,), testing_rate, device=self.device)

            if self._surveillance_policy is not None:
                test_prob = self._apply_surveillance_allocation(
                    test_prob, t, self._is_isolating,
                )

            test_prob[is_symptomatic] *= self.symptomatic_multiplier

            test_prob[self._is_isolating] = 0.0

        test_prob = test_prob.clamp(0.0, 1.0)

        tested = torch.bernoulli(test_prob).bool()
        n_tested = tested.sum().item()

        truly_infected = (disease_stages == self.INFECTED_VAR) | (disease_stages == self.EXPOSED_VAR)

        tp_candidates = tested & truly_infected
        n_tp_cand = tp_candidates.sum().item()
        if n_tp_cand > 0:
            tp_hits = torch.bernoulli(
                torch.full((n_tp_cand,), self.test_sensitivity, device=self.device)
            ).bool()
            true_positives = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
            true_positives[tp_candidates] = tp_hits
        else:
            true_positives = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)

        fp_candidates = tested & ~truly_infected
        n_fp_cand = fp_candidates.sum().item()
        if n_fp_cand > 0:
            fp_hits = torch.bernoulli(
                torch.full((n_fp_cand,), 1.0 - self.test_specificity, device=self.device)
            ).bool()
            false_positives = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
            false_positives[fp_candidates] = fp_hits
        else:
            false_positives = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)

        test_positive = true_positives | false_positives
        n_positive = test_positive.sum().item()

        if self.calibration_mode and hasattr(self, 'calibrate_isolation_reduction'):
            iso_reduction = torch.sigmoid(self.calibrate_isolation_reduction)
        else:
            iso_reduction = self.isolation_reduction

        will_isolate = test_positive & ~self._is_isolating

        self._is_isolating[will_isolate] = True
        self._isolation_end_day[will_isolate] = t + self.isolation_duration
        self._is_detected[test_positive] = True

        self._daily_detected[t] = float(n_positive)

        is_quarantined = state["agents"]["citizens"].get("is_quarantined")
        if is_quarantined is not None:
            iso_val = iso_reduction if isinstance(iso_reduction, float) else iso_reduction.item()
            is_quarantined = is_quarantined.view(-1)
            is_quarantined.zero_()
            is_quarantined[self._is_isolating] = iso_val
            state["agents"]["citizens"]["is_quarantined"] = is_quarantined.unsqueeze(1)

        self.tracking["daily_tests"].append(n_tested)
        self.tracking["daily_positives"].append(n_positive)
        self.tracking["daily_isolating"].append(int(self._is_isolating.sum().item()))

        if self._surveillance_policy is not None:
            self._update_surveillance_state(t, tested, test_positive)

        return {}

    @property
    def daily_detected(self):
        """Return daily detected cases tensor for comparison with SSI data."""
        if self._daily_detected is None:
            return torch.zeros(self.num_timesteps, device=self.device)
        return self._daily_detected
