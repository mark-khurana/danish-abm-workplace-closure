import os
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import re

from agent_torch.core.substep import SubstepTransitionMessagePassing
from agent_torch.core.helpers import get_by_path
from agent_torch.core.distributions import StraightThroughBernoulli
from danish_abm.substeps.utils import LAYER_NAMES


def _load_municipality_array(pop_dir):
    """Load municipality codes from agents.parquet or municipality.pickle."""
    import numpy as np
    import pandas as pd

    for ext in (".parquet", ".pq", ".csv"):
        agents_path = os.path.join(pop_dir, f"agents{ext}")
        if os.path.exists(agents_path):
            for col_name in ("municipality", "Municipality", "kommune", "muni_id"):
                try:
                    if ext in (".parquet", ".pq"):
                        df = pd.read_parquet(agents_path, columns=[col_name])
                    else:
                        df = pd.read_csv(agents_path, usecols=[col_name])
                    return df[col_name].values.squeeze()
                except (KeyError, ValueError):
                    continue

    import pickle
    muni_path = os.path.join(pop_dir, "municipality.pickle")
    if os.path.exists(muni_path):
        with open(muni_path, "rb") as f:
            muni_data = pickle.load(f)
        if hasattr(muni_data, "values"):
            return muni_data.values.squeeze()
        return np.asarray(muni_data).squeeze()

    return None


REGION_NAMES = ("Hovedstaden", "Sjaelland", "Syddanmark", "Midtjylland", "Nordjylland")
_REGION_MUNICIPALITIES = {
    0: (101, 147, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 173, 175, 183, 185,
        187, 190, 201, 210, 217, 219, 223, 230, 240, 250, 260, 270, 400, 411),
    1: (253, 259, 265, 269, 306, 316, 320, 326, 329, 330, 336, 340, 350, 360, 370, 376,
        390),
    2: (410, 420, 430, 440, 450, 461, 479, 480, 482, 492, 510, 530, 540, 550, 561, 563,
        573, 575, 580, 607, 621, 630),
    3: (615, 657, 661, 665, 671, 706, 707, 710, 727, 730, 740, 741, 746, 751, 756, 760,
        766, 779, 791),
    4: (773, 787, 810, 813, 820, 825, 840, 846, 849, 851, 860),
}
MUNICIPALITY_TO_REGION = {code: region
                          for region, codes in _REGION_MUNICIPALITIES.items()
                          for code in codes}


def municipality_region(code):
    """Region index (0-4, see REGION_NAMES) of a Danish municipality code."""
    try:
        return MUNICIPALITY_TO_REGION[int(code)]
    except KeyError:
        raise KeyError(
            f"municipality code {code!r} is not a Danish municipality; the community "
            f"term's region tier cannot place it") from None


class NewTransmission(SubstepTransitionMessagePassing):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]
        self.INFECTED_VAR = self.config["simulation_metadata"]["INFECTED_VAR"]

        self.num_timesteps = self.config["simulation_metadata"]["num_steps_per_episode"]
        self.num_weeks = self.config["simulation_metadata"]["NUM_WEEKS"]

        self.STAGE_UPDATE_VAR = 1
        self.INFINITY_TIME = self.config["simulation_metadata"]["INFINITY_TIME"]
        _sfia = self.config["simulation_metadata"].get("SFInfectorAge_value")
        if _sfia is None:
            _env = (self.config.get("state", {}) or {}).get("environment", {}) or {}
            _sfia = (_env.get("SFInfectorAge", {}) or {}).get("value")
        if _sfia is None:
            raise KeyError(
                "SFInfectorAge is missing from the config. It carries the age profile "
                "of infectiousness (Davies 2020 clinical fraction x Buitrago-Garcia "
                "2020 subclinical discount, levelled to Sah 2021). Running without it "
                "silently removes the term while the manuscript still cites it.")
        self._sf_infector_age = torch.as_tensor(
            _sfia, dtype=torch.float32, device=self.device).ravel()

        self.EXPOSED_TO_INFECTED_TIME = self.config["simulation_metadata"][
            "EXPOSED_TO_INFECTED_TIME"
        ]

        self._compartment_model = None
        self._load_compartment_model()

        self._distributional = self.config["simulation_metadata"].get(
            "disease_params_distributional", False
        )
        self._per_agent_latent = None

        self._multi_strain = self.config["simulation_metadata"].get("multi_strain", False)
        self._strain_manager = None
        self._immunity_tracker = None

        self.reinfection_susceptibility = self.config["simulation_metadata"].get(
            "REINFECTION_SUSCEPTIBILITY", 1.0
        )
        self._previously_infected = None

        self._transmission_mode = "respiratory"
        if self._compartment_model is not None:
            self._transmission_mode = self._compartment_model.transmission_mode

        seasonal_cfg = self.config["simulation_metadata"].get("seasonal", None)
        if seasonal_cfg is None and self._compartment_model is not None:
            sf = self._compartment_model.seasonal_forcing
            if sf is not None:
                seasonal_cfg = sf
        if seasonal_cfg:
            from danish_abm.seasonal import SeasonalForcing, get_seasonal
            if isinstance(seasonal_cfg, str):
                self._seasonal = get_seasonal(seasonal_cfg)
            elif isinstance(seasonal_cfg, dict):
                preset = seasonal_cfg.get("preset", None)
                if preset:
                    self._seasonal = get_seasonal(preset)
                else:
                    self._seasonal = SeasonalForcing(**seasonal_cfg)
            else:
                self._seasonal = None
            self._sim_start_doy = self.config["simulation_metadata"].get("sim_start_doy", 0)
        else:
            self._seasonal = None

        self.mode = self.config["simulation_metadata"]["EXECUTION_MODE"]
        self.st_bernoulli = StraightThroughBernoulli.apply

        self.calibration_mode = self.config["simulation_metadata"]["calibration"]

        self.num_layers = self.config["simulation_metadata"].get("num_network_layers", 3)

        self._layer_pressure = None
        self._step_layer_pressure = None

        self._edge_chunk_size = int(os.environ.get("ABM_EDGE_CHUNK_SIZE", 2_000_000))

        self.symmetric_transmission = bool(
            self.config["simulation_metadata"].get("symmetric_transmission", False))
        if not self.symmetric_transmission:
            print("  [transmission] symmetric_transmission=False: each stored edge "
                  "transmits in ONE direction only. Set symmetric_transmission: true "
                  "in simulation_metadata and recalibrate to make contacts "
                  "bidirectional.", flush=True)

        meta = self.config["simulation_metadata"]
        self.isolation_in_transmission = bool(meta.get("isolation_in_transmission", True))
        self.isolation_reduction = float(meta.get("isolation_reduction", 0.70))
        if not 0.0 <= self.isolation_reduction <= 1.0:
            raise ValueError(f"isolation_reduction must lie in [0, 1], got "
                             f"{self.isolation_reduction}")
        _layer_names = list(meta.get("network_layer_names", list(LAYER_NAMES)))
        self._household_layer_idx = (_layer_names.index("household")
                                     if "household" in _layer_names else -1)
        self._isolation_reported = False
        print(f"  [transmission] case isolation "
              f"{'ON' if self.isolation_in_transmission else 'OFF'}: reduction "
              f"{self.isolation_reduction:.2f} outside the household "
              f"(household layer index {self._household_layer_idx})", flush=True)

        self._transmission_layer_indices = None
        if self._compartment_model is not None:
            cm = self._compartment_model
            if cm.transmission_layers and cm.network_layers:
                layer_name_to_idx = {name: i for i, name in enumerate(cm.network_layers)}
                self._transmission_layer_indices = [
                    layer_name_to_idx[l] for l in cm.transmission_layers
                    if l in layer_name_to_idx
                ]

        self.network_years = self.config["simulation_metadata"].get("network_years", None)
        self.network_base_dir = self.config["simulation_metadata"].get("network_base_dir", None)
        self._current_network_year = None

        self.tracking = {"R_eff": [], "S": [], "E": [], "I": [], "R": [], "M": [], "new_infections": []}
        if self._compartment_model is not None:
            for name in self._compartment_model.compartment_names:
                if name not in self.tracking:
                    self.tracking[name] = []
        self.tracking["new_infections_by_age"] = []
        self.tracking["active_triggers"] = []

        from danish_abm import runtime_set
        runtime_set(self.config, "transmission_tracking", self.tracking)

        self._infectiousness_weights = None
        if self._compartment_model is not None:
            ibs = self._compartment_model.infectiousness_by_state
            if any(v != 1.0 for v in ibs.values()):
                self._infectiousness_weights = ibs

        self._record_trajectory = self.config["simulation_metadata"].get("record_trajectory", False)
        self._trajectory = None
        self._newly_exposed_ste = []

        self._test_ids_by_day = None
        self._tested_stages = {}

        self._time_one_hot = torch.eye(self.num_timesteps, device=self.device)
        self._week_one_hot = torch.eye(self.num_weeks, device=self.device)
        self._buffer_cache = {}

        pop_dir = self.config["simulation_metadata"]["population_dir"]
        contact_matrix_path = os.path.join(pop_dir, "contact_matrix.pt")
        if os.path.exists(contact_matrix_path):
            data = torch.load(contact_matrix_path, weights_only=False)
            self.C_comix = data["C_comix"].to(self.device)
            self.n_age_groups = self.C_comix.shape[0]
            self.has_community_mixing = True
            print(f"  [transmission] community mean-field ENABLED from "
                  f"{os.path.basename(contact_matrix_path)} "
                  f"({self.n_age_groups} age bands, "
                  f"{float(self.C_comix.sum(dim=1).mean()):.2f} contacts/person/day)",
                  flush=True)
        else:
            self.has_community_mixing = False
            print(f"  [transmission] *** community mean-field DISABLED: no "
                  f"{contact_matrix_path} ***\n"
                  f"      R_community is inert and the mean-field share of "
                  f"transmission will read 0.0%. If the structural q for this "
                  f"population allocates a share to the mean-field channel, that "
                  f"share is being absorbed by the network layers.\n"
                  f"      Build it: python decomposition/build_contact_matrix.py "
                  f"--population-dir {pop_dir}", flush=True)

        self._municipality_mixing = self.config["simulation_metadata"].get(
            "municipality_mixing", False
        )
        self._municipality_ids = None
        self._region_ids = None
        self._n_municipalities = 0
        self._n_regions = 0
        self._municipality_initialized = False

        from danish_abm.substeps.utils import interv_schedule_path
        intervention_path = interv_schedule_path(pop_dir)
        self.npi_active = None
        if os.path.exists(intervention_path):
            sched = torch.load(intervention_path, weights_only=False)
            raw = sched["multipliers"].to(self.device)
            self.intervention_schedule = self._migrate_intervention_tensor(raw)
            self.has_interventions = True
            if "npi_active" in sched:
                self.npi_active = sched["npi_active"].to(self.device)
            self.spillover_scale = float(sched.get("spillover_scale", 1.0))
            self._telework_setup(sched)
        else:
            self.has_interventions = False
            self._telework_mode = None

        from danish_abm.npi import build_spillover_matrix, SPILLOVER_NPI_NAMES
        layer_names_cfg = self.config["simulation_metadata"].get(
            "network_layer_names", list(LAYER_NAMES)
        )
        logits_init, prior_means, prior_stds = build_spillover_matrix(
            npi_names=SPILLOVER_NPI_NAMES, layer_names=list(layer_names_cfg)
        )
        self._spillover_prior_means = prior_means.to(self.device)
        self._spillover_prior_stds = prior_stds.to(self.device)
        self._spillover_logits_init = logits_init.to(self.device)
        self._spillover_initialized = False

    def _telework_setup(self, sched):
        """Read ABM_TELEWORK and the closure's colleague multiplier by day. Presence
        itself is solved lazily on the first forward pass, when the edges exist.
        """
        import os as _os
        self._telework_mode = None
        self._tw_mult_by_day = None
        self._tw_presence = {}
        mode = (_os.environ.get("ABM_TELEWORK") or "").strip().lower()
        if not mode:
            return
        from danish_abm.telework import MODES
        if mode not in MODES:
            raise ValueError(f"ABM_TELEWORK={mode!r}; expected one of {MODES}")
        layer_names = list(self.config["simulation_metadata"].get(
            "network_layer_names", list(LAYER_NAMES)))
        self._tw_colleague_idx = layer_names.index("colleague")
        n_days = int(self.intervention_schedule.shape[0])
        mult = torch.ones(n_days, dtype=torch.float32)
        for e in sched.get("entries") or []:
            if "closure" not in str(e.get("name", "")) or "workplace" not in str(e.get("name", "")):
                continue
            m = (e.get("affected_layers") or {}).get("colleague")
            if m is None:
                continue
            s0, s1 = max(0, int(e["day_start"])), min(n_days, int(e["day_end"]))
            if s0 < s1:
                mult[s0:s1] *= float(m)
        if not (mult < 1.0).any():
            print("  [telework] ABM_TELEWORK set but no workplace-closure entry with a "
                  "colleague multiplier in this schedule -- nothing to redistribute "
                  "(expected in the no-closure arm)", flush=True)
            self._telework_mode = mode
            self._tw_mult_by_day = mult.to(self.device)
            return
        self._telework_mode = mode
        self._tw_mult_by_day = mult.to(self.device)
        print(f"  [telework] mode={mode}: colleague links carry p_i*p_j while the "
              f"workplace closure is in force; closure multipliers by day: "
              f"{sorted(set(round(float(v), 4) for v in mult[mult < 1].tolist()))}", flush=True)

    def _telework_presence(self, m, edge_index, edge_attr):
        """Presence vector for closure multiplier m (cached per distinct m)."""
        key = round(float(m), 6)
        if key in self._tw_presence:
            return self._tw_presence[key]
        import os as _os, json as _json
        import numpy as _np
        import pandas as _pd
        from danish_abm import telework as TW
        pop_dir = self.config["simulation_metadata"]["population_dir"]
        n = int(edge_index.max().item()) + 1
        col_idx = self._tw_colleague_idx
        E = int(edge_index.shape[1])
        CH = 20_000_000

        def edge_chunks():
            for s0 in range(0, E, CH):
                s1 = min(E, s0 + CH)
                sel = edge_attr[0, s0:s1].to(torch.int16) == col_idx
                if not bool(sel.any()):
                    continue
                yield (edge_index[0, s0:s1][sel], edge_index[1, s0:s1][sel],
                       edge_attr[1, s0:s1][sel].float())

        expo = torch.zeros(n, device=edge_attr.device, dtype=torch.float32)
        for src, dst, w in edge_chunks():
            expo.scatter_add_(0, src.long(), w)
            expo.scatter_add_(0, dst.long(), w)
        codes = None
        for ext in (".parquet", ".pq"):
            f = _os.path.join(pop_dir, f"agents{ext}")
            if _os.path.exists(f):
                df = _pd.read_parquet(f)
                col = next((c for c in ("socio13", "SOCIO13", "socio") if c in df.columns), None)
                if col is None:
                    raise KeyError(f"ABM_TELEWORK needs a socio13 column in {f}")
                codes = _pd.to_numeric(df[col], errors="coerce").to_numpy()
                break
        if codes is None:
            raise FileNotFoundError(f"ABM_TELEWORK needs agents.parquet under {pop_dir}")
        if len(codes) < n:
            codes = _np.concatenate([codes, _np.full(n - len(codes), _np.nan)])
        codes = codes[:n]
        weights = TW.load_weights()
        tau_np, filled, mean_tau = TW.agent_teleworkability(codes, expo.cpu().numpy(), weights)
        tau = torch.as_tensor(tau_np, dtype=torch.float32, device=edge_attr.device)
        k, p, achieved = TW.solve_presence(tau, edge_chunks, float(m), mode=self._telework_mode)
        rep = TW.report(tau_np, p.cpu().numpy(), codes, k, float(m), achieved, mean_tau,
                        filled, expo.cpu().numpy())
        rep["mode"] = self._telework_mode
        rep["weights_table"] = weights
        print(f"  [telework] m={float(m):.4f} -> k={k:.4f}, achieved exposure-weighted "
              f"mean p_i*p_j={achieved:.4f}; {int(filled.sum()):,} agents filled with mean "
              f"tau={mean_tau:.3f}", flush=True)
        for r in rep["by_socio13"]:
            if r["n_with_colleagues"] >= 1000:
                print(f"      socio13 {r['socio13']:<4} n={r['n']:>9,}  tau={r['tau']:.2f}  "
                      f"presence={r['presence']:.3f}", flush=True)
        out = _os.environ.get("ABM_TELEWORK_REPORT")
        if out:
            try:
                _os.makedirs(_os.path.dirname(out) or ".", exist_ok=True)
                prev = {}
                if _os.path.exists(out):
                    with open(out) as fh:
                        prev = _json.load(fh)
                prev[str(key)] = rep
                with open(out, "w") as fh:
                    _json.dump(prev, fh, indent=1)
            except OSError as exc:
                print(f"  [telework] could not write report {out}: {exc}", flush=True)
        self._tw_presence[key] = p.to(torch.float16)
        return self._tw_presence[key]

    def _apply_telework(self, msg, ei, ea, t, edge_index, edge_attr):
        """Multiply colleague-edge messages by p_i*p_j/m on closure days."""
        if not getattr(self, "_telework_mode", None) or self._tw_mult_by_day is None:
            return msg
        ti = int(t)
        if ti >= self._tw_mult_by_day.shape[0]:
            return msg
        m = self._tw_mult_by_day[ti]
        if float(m) >= 1.0:
            return msg
        p = self._telework_presence(m, edge_index, edge_attr)
        is_col = ea[0].to(torch.int16) == self._tw_colleague_idx
        f = (p[ei[0].long()].float() * p[ei[1].long()].float()) / float(m)
        f = torch.where(is_col, f, torch.ones_like(f))
        return msg * f.view(-1, 1).to(msg.dtype)

    def _init_spillover_logits(self):
        """Override learnable spillover_logits with literature-prior initialization."""
        self._spillover_initialized = True
        if self.calibration_mode and hasattr(self, 'calibrate_spillover_logits'):
            tgt = self.calibrate_spillover_logits
        elif not self.calibration_mode and "spillover_logits" in self.learnable_args:
            tgt = self.learnable_args["spillover_logits"]
        else:
            return
        if tgt.shape == self._spillover_logits_init.shape and bool(torch.any(tgt != 0)):
            return
        with torch.no_grad():
            tgt.copy_(self._spillover_logits_init)

    def _init_municipality_data(self, n_agents):
        """Lazily initialize municipality/region data on first forward call."""
        self._municipality_initialized = True
        if not self._municipality_mixing:
            return

        pop_dir = self.config["simulation_metadata"]["population_dir"]
        muni_arr = _load_municipality_array(pop_dir)
        if muni_arr is None:
            self._municipality_mixing = False
            return

        if len(muni_arr) != n_agents:
            import os as _os
            sample_dir = _os.path.join(pop_dir, "_sample")
            if _os.path.isdir(sample_dir):
                sample_arr = _load_municipality_array(sample_dir)
                if sample_arr is not None and len(sample_arr) == n_agents:
                    muni_arr = sample_arr
            if len(muni_arr) != n_agents:
                if len(muni_arr) > n_agents:
                    muni_arr = muni_arr[:n_agents]
                else:
                    print(f"  WARNING: municipality data size {len(muni_arr)} "
                          f"< agent count {n_agents}, disabling municipality mixing")
                    self._municipality_mixing = False
                    return

        import numpy as _np
        unique_codes = sorted(set(int(x) for x in muni_arr))
        code_to_idx = {c: i for i, c in enumerate(unique_codes)}
        self._n_municipalities = len(unique_codes)
        idx_arr = [code_to_idx[int(x)] for x in muni_arr]
        self._municipality_ids = torch.tensor(idx_arr, dtype=torch.long, device=self.device)
        self._municipality_names = unique_codes

        code_region = {code: municipality_region(code) for code in unique_codes}
        region_arr = [code_region[int(x)] for x in muni_arr]
        _reg_counts = _np.bincount(_np.asarray(region_arr), minlength=len(REGION_NAMES))
        print("  [transmission] community region tier: "
              + ", ".join(f"{n}={c:,}" for n, c in zip(REGION_NAMES, _reg_counts)),
              flush=True)
        self._region_ids = torch.tensor(region_arr, dtype=torch.long, device=self.device)
        self._n_regions = len(set(region_arr))

        muni_idx_arr = _np.array(idx_arr)
        muni_pop = _np.bincount(muni_idx_arr, minlength=self._n_municipalities).astype(float)
        log_pop = _np.log(muni_pop.clip(min=1))
        log_pop_centered = log_pop - log_pop.mean()
        agent_log_pop = log_pop_centered[muni_idx_arr]
        self._agent_log_muni_pop = torch.tensor(
            agent_log_pop, dtype=torch.float32, device=self.device
        )

    def _load_compartment_model(self):
        """Load CompartmentModel from runtime registry or config."""
        from danish_abm import runtime_get
        cm = runtime_get(self.config, "compartment_model")
        if cm is not None:
            self._compartment_model = cm
            return

        meta = self.config["simulation_metadata"]
        preset = meta.get("pathogen_preset")
        compartment_config = meta.get("compartment_config")
        if preset is not None or compartment_config is not None:
            from danish_abm.compartments import CompartmentModel
            if compartment_config:
                self._compartment_model = CompartmentModel.from_dict(compartment_config)
            else:
                self._compartment_model = CompartmentModel.from_preset(preset)

    def _get_buffer(self, name, like_tensor):
        key = (name, tuple(like_tensor.shape), like_tensor.dtype, like_tensor.device)
        buf = self._buffer_cache.get(key)
        if (
            buf is None
            or buf.shape != like_tensor.shape
            or buf.dtype != like_tensor.dtype
            or buf.device != like_tensor.device
        ):
            buf = torch.empty_like(like_tensor)
            self._buffer_cache[key] = buf
        return buf

    def _migrate_intervention_tensor(self, raw):
        """Migrate legacy 5-column intervention tensors to 7-column format."""
        if raw.shape[1] == self.num_layers + 2:
            return raw
        if raw.shape[1] == 5:
            n_days = raw.shape[0]
            new = torch.ones(n_days, self.num_layers + 2, device=raw.device)
            new[:, 1] = raw[:, 0]
            new[:, 4] = raw[:, 1]
            new[:, 3] = raw[:, 2]
            new[:, 5] = raw[:, 3]
            new[:, 6] = raw[:, 4]
            return new
        return raw

    def _maybe_swap_network(self, state, t):
        """Swap to a different year's network at yearly boundaries (if configured)."""
        if self.network_years is None or self.network_base_dir is None:
            return
        year_idx = min(t // 365, len(self.network_years) - 1)
        year = self.network_years[year_idx]
        if year == self._current_network_year:
            return
        self._current_network_year = year
        from danish_abm.substeps.utils import load_network_for_year
        _, (edge_index, edge_attr) = load_network_for_year(
            {"network_base_dir": self.network_base_dir}, year
        )
        adj_path = "network/agent_agent/infection_network/adjacency_matrix"
        parts = adj_path.split("/")
        node = state
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = (edge_index.to(self.device), edge_attr.to(self.device))

    def _accumulate_layer_pressure(self, edge_attr, msg, msg_r=None):
        """Sum transmission pressure per network layer."""
        with torch.no_grad():
            if self._layer_pressure is None:
                self._layer_pressure = torch.zeros(self.num_layers + 1,
                                                   dtype=torch.float64,
                                                   device=msg.device)
            if self._step_layer_pressure is None:
                self._step_layer_pressure = torch.zeros(self.num_layers,
                                                        dtype=torch.float64,
                                                        device=msg.device)
            types = edge_attr[0].long().clamp_(0, self.num_layers - 1)
            _w = msg.detach().reshape(-1).float()
            if msg_r is not None:
                _w = _w + msg_r.detach().reshape(-1).float()
            sums = torch.bincount(
                types, weights=_w, minlength=self.num_layers,
            )[:self.num_layers].double()
            del types, _w
            self._layer_pressure[:self.num_layers] += sums
            self._step_layer_pressure += sums
            del sums

    def layer_pressure_shares(self):
        """Realised share of transmission pressure by layer, as a dict."""
        if self._layer_pressure is None:
            return {}
        lp = self._layer_pressure[:self.num_layers + 1].cpu()
        total = float(lp.sum())
        if total <= 0:
            return {}
        names = list(LAYER_NAMES)[:self.num_layers] + ["community(mean-field)"]
        return {n: float(v) / total for n, v in zip(names, lp)}

    def report_layer_pressure(self):
        """Print the realised share of transmission pressure by layer."""
        shares = self.layer_pressure_shares()
        if not shares:
            return
        parts = [f"{n}={100.0 * v:.1f}%" for n, v in shares.items()]
        print(f"  [transmission] realised pressure by layer: {', '.join(parts)}",
              flush=True)
        print("  [transmission] compare against the empirical literature on where "
              "transmission occurred; a workplace-dominated split under lockdown "
              "would indicate the layer weighting, not the epidemiology, is driving "
              "the NPI results.", flush=True)

    def _lam(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        R,
        q_layers,
        SFSusceptibility,
        SFInfector,
        lam_gamma_integrals,
    ):
        S_A_s = SFSusceptibility[x_i[:, 0].long()]
        A_s_i = SFInfector[x_j[:, 1].long()]
        B_n = edge_attr[1, :]
        integrals = torch.zeros_like(B_n)
        infected_idx = x_j[:, 2].bool()
        infected_times = (t - x_j[infected_idx, 3] - 1).clamp(min=0)

        n_integrals = lam_gamma_integrals.shape[0]
        integrals[infected_idx] = lam_gamma_integrals[
            infected_times.long().clamp(max=n_integrals - 1)].to(integrals.dtype)
        edge_network_numbers = edge_attr[0, :]

        I_bar = x_i[:, 4].clamp(min=1.0)

        edge_types = edge_network_numbers.long().clamp(0, q_layers.shape[0] - 1)
        q_per_edge = q_layers[edge_types]

        A_age_j = self._sf_infector_age[x_j[:, 0].long()]
        res = R * q_per_edge * S_A_s * A_s_i * A_age_j * B_n * integrals / I_bar

        if self.isolation_in_transmission:
            iso_j = x_j[:, 5].to(res.dtype)
            outside_home = (edge_types != self._household_layer_idx).to(res.dtype)
            res = res * (1 - iso_j * outside_home)

        if self._transmission_layer_indices is not None:
            allowed = torch.zeros_like(edge_types, dtype=torch.float32)
            for li in self._transmission_layer_indices:
                allowed = allowed + (edge_types == li).float()
            res = res * allowed.clamp(max=1.0)

        return res.view(-1, 1)

    def message(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        R,
        q_layers,
        SFSusceptibility,
        SFInfector,
        lam_gamma_integrals,
    ):
        return self._lam(
            x_i, x_j, edge_attr, t, R, q_layers, SFSusceptibility, SFInfector, lam_gamma_integrals
        )

    def _isolation_level_from_state(self, state, n_agents):
        """(n_agents, 1) isolation level: isolation_reduction for agents the testing
        substep has marked as isolating (agents/citizens/is_quarantined), else 0.
        """
        level = torch.zeros(n_agents, 1, device=self.device)
        if not self.isolation_in_transmission:
            return level
        q = state["agents"]["citizens"].get("is_quarantined")
        if q is None:
            return level
        return ((q.reshape(-1, 1) > 0).to(level.dtype)
                * self.isolation_reduction).to(self.device)

    def _community_lambda(self, ages, disease_stages, R_community, isolation=None):
        """Mean-field community mixing using the Prem et al. 2021 'others' matrix."""
        if self._transmission_mode in ("contact", "sexual"):
            return torch.zeros(disease_stages.shape[0], 1, device=self.device)

        ages_flat = ages.squeeze().long().clamp(0, self.n_age_groups - 1)
        stages_flat = disease_stages.squeeze()
        cm = self._compartment_model
        if cm is not None:
            is_infected = cm.infectious_mask(stages_flat)
            if self._infectiousness_weights is not None:
                is_infected = cm.infectiousness_weights(stages_flat)
        else:
            is_infected = (stages_flat == self.INFECTED_VAR).float()
        if isolation is not None:
            is_infected = is_infected * (1 - isolation.reshape(-1).to(is_infected.dtype))

        A = self.n_age_groups
        nat_inf = torch.zeros(A, device=self.device)
        nat_pop = torch.zeros(A, device=self.device)
        nat_inf.scatter_add_(0, ages_flat, is_infected)
        nat_pop.scatter_add_(0, ages_flat, torch.ones_like(is_infected))
        nat_prev = nat_inf / nat_pop.clamp(min=1)
        nat_foi = self.C_comix @ nat_prev

        if self._municipality_mixing and self._municipality_ids is not None:
            M = self._n_municipalities
            muni_joint = self._municipality_ids * A + ages_flat
            muni_inf = torch.zeros(M * A, device=self.device)
            muni_pop = torch.zeros(M * A, device=self.device)
            muni_inf.scatter_add_(0, muni_joint, is_infected)
            muni_pop.scatter_add_(0, muni_joint, torch.ones_like(is_infected))
            muni_prev = (muni_inf / muni_pop.clamp(min=1)).view(M, A)
            muni_foi = muni_prev @ self.C_comix.T

            if self._region_ids is not None and self._n_regions > 1:
                R = self._n_regions
                reg_joint = self._region_ids * A + ages_flat
                reg_inf = torch.zeros(R * A, device=self.device)
                reg_pop = torch.zeros(R * A, device=self.device)
                reg_inf.scatter_add_(0, reg_joint, is_infected)
                reg_pop.scatter_add_(0, reg_joint, torch.ones_like(is_infected))
                reg_prev = (reg_inf / reg_pop.clamp(min=1)).view(R, A)
                reg_foi = reg_prev @ self.C_comix.T
            else:
                reg_foi = None

            if self.calibration_mode:
                logits = self.calibrate_spatial_mixing_logits
                slope = self.calibrate_spatial_local_pop_slope
            else:
                logits = self.learnable_args["spatial_mixing_logits"]
                slope = self.learnable_args["spatial_local_pop_slope"]
            base_logits = logits.squeeze()

            pop_shift = slope.squeeze() * self._agent_log_muni_pop
            agent_logit_muni = base_logits[0] + pop_shift
            agent_logit_reg = base_logits[1].expand_as(pop_shift)
            agent_logit_nat = base_logits[2].expand_as(pop_shift)

            agent_logits = torch.stack([agent_logit_muni, agent_logit_reg, agent_logit_nat], dim=1)
            weights = torch.softmax(agent_logits, dim=1)

            per_agent_foi = weights[:, 0] * muni_foi[self._municipality_ids, ages_flat]
            if reg_foi is not None:
                per_agent_foi = per_agent_foi + weights[:, 1] * reg_foi[self._region_ids, ages_flat]
            else:
                per_agent_foi = per_agent_foi + weights[:, 1] * nat_foi[ages_flat]
            per_agent_foi = per_agent_foi + weights[:, 2] * nat_foi[ages_flat]

            community_lam = R_community * per_agent_foi
            return community_lam.unsqueeze(1)
        else:
            community_lam = R_community * nat_foi[ages_flat]
            return community_lam.unsqueeze(1)

    def update_stages(self, current_stages, newly_exposed_today):
        newly_exposed_today = (newly_exposed_today > 0.5).float()
        new_stages = current_stages + newly_exposed_today * self.STAGE_UPDATE_VAR
        return new_stages

    def update_transition_times(self, t, current_transition_times, newly_exposed_today):
        out = self._get_buffer("next_stage_time", current_transition_times)
        out.copy_(current_transition_times)
        mask = newly_exposed_today.bool().squeeze()
        if mask.numel() == 0 or not mask.any():
            return out
        n_newly = mask.sum().item()
        if self._distributional:
            if self._per_agent_latent is None:
                self._init_distributional_params(int(current_transition_times.shape[0]))
            latent_times = self._per_agent_latent[mask]
            out[mask] = t + 1 + latent_times.unsqueeze(-1) if latent_times.dim() == 1 and out.dim() == 2 else t + 1 + latent_times
        else:
            out[mask] = t + 1 + self.EXPOSED_TO_INFECTED_TIME
        return out

    def _init_distributional_params(self, n_agents):
        """Lazily initialise per-agent disease timing from DiseaseParams."""
        from danish_abm.disease_params import DiseaseParams
        dp_config = self.config["simulation_metadata"].get("disease_params", {})
        dp = DiseaseParams.from_ci(**dp_config)
        self._per_agent_latent = dp.sample_latent(n_agents, device=self.device)
        self._disease_params = dp

    def _init_multi_strain(self, n_agents, current_stages):
        """Lazily initialise multi-strain tracking."""
        from danish_abm.strains import StrainConfig, StrainManager
        from danish_abm.immunity import ImmunityTracker

        strain_cfg = self.config["simulation_metadata"].get("strain_config", {})
        preset = strain_cfg.get("preset", None)
        if preset == "covid_2020":
            self._strain_config = StrainConfig.covid_2020()
        elif preset == "covid_2021":
            self._strain_config = StrainConfig.covid_2021()
        elif preset == "covid_2022":
            self._strain_config = StrainConfig.covid_2022()
        elif preset == "influenza_season":
            self._strain_config = StrainConfig.influenza_season()
        elif not preset:
            self._strain_config = StrainConfig.single_strain()
        else:
            raise ValueError(
                f"strain_config preset {preset!r} is not recognised by this dispatch. "
                f"Known: covid_2020, covid_2021, covid_2022, influenza_season. "
                f"Add a branch for it rather than falling back to single_strain().")

        self._strain_manager = StrainManager(
            self._strain_config, n_agents, device=self.device
        )

        from danish_abm import runtime_set
        runtime_set(self.config, "strain_manager", self._strain_manager)
        runtime_set(self.config, "strain_config", self._strain_config)

        antigen_names = self._strain_config.antigen_names

        self._immunity_tracker = ImmunityTracker(
            n_agents, antigen_names, device=self.device
        )

        cfg_eff = self.config["simulation_metadata"].get("vaccination_efficacy", None)
        if isinstance(cfg_eff, dict):
            cfg_eff = cfg_eff.get("mean", cfg_eff.get("value", None))
        cfg_eff = float(cfg_eff) if cfg_eff is not None else None
        self._antigen_cross_immunity = self._strain_config.antigen_cross_immunity(
            cfg_eff).to(self.device)
        if self._antigen_cross_immunity.shape != (
                self._immunity_tracker.n_antigens, self._strain_config.n_strains):
            raise ValueError(
                f"antigen cross-immunity is "
                f"{tuple(self._antigen_cross_immunity.shape)}, expected "
                f"({self._immunity_tracker.n_antigens}, "
                f"{self._strain_config.n_strains})."
            )

        from danish_abm import runtime_set
        runtime_set(self.config, "immunity_tracker", self._immunity_tracker)


        cm_ref = self._compartment_model
        if cm_ref is not None:
            infected_mask = cm_ref.infectious_mask(current_stages).bool()
        else:
            infected_mask = (current_stages > self.SUSCEPTIBLE_VAR) & (
                current_stages < self.RECOVERED_VAR
            )
        self._strain_manager.assign_initial_strains(infected_mask)

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        if num_timesteps == self.num_timesteps:
            return self._time_one_hot[timestep]
        if num_timesteps == self.num_weeks:
            return self._week_one_hot[timestep]
        timestep_tensor = torch.tensor([timestep], device=self.device)
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)
        return one_hot_tensor.to(self.device)

    def update_infected_times(self, t, agents_infected_times, newly_exposed_today):
        out = self._get_buffer("infected_time", agents_infected_times)
        out.copy_(agents_infected_times)
        mask = newly_exposed_today.bool().squeeze()
        if mask.numel() > 0:
            out[mask] = t
        return out

    def forward(self, state, action=None):
        input_variables = self.input_variables
        t = int(state["current_step"])
        if self._step_layer_pressure is not None:
            self._step_layer_pressure.zero_()
        if t < 3 or t % 10 == 0:
            import psutil
            _mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
            _inf = sum(self.tracking.get("new_infections", [])) if self.tracking.get("new_infections") else 0
            _last7 = sum(self.tracking["new_infections"][-7:]) if len(self.tracking.get("new_infections", [])) >= 7 else _inf
            print(f"  [step {t:3d}/{self.num_timesteps}] mem={_mem:.1f}GB, "
                  f"cum_inf={_inf:.0f}, last7d={_last7:.0f}", flush=True)
            with torch.no_grad():
                _cs = get_by_path(state, re.split("/", input_variables["disease_stage"])).squeeze()
                _ct = get_by_path(state, re.split("/", input_variables["next_stage_time"])).squeeze()
                _nE = (_cs == self.EXPOSED_VAR).sum().item()
                _nI = ((_cs > self.SUSCEPTIBLE_VAR) & (_cs < self.RECOVERED_VAR)).sum().item()
                _eReady = ((_cs == self.EXPOSED_VAR) & (_ct <= t)).sum().item()
                print(f"  [tx t={t:3d}] E={_nE} I(E+I)={_nI} E_ready={_eReady}", flush=True)
        if not self._spillover_initialized:
            self._init_spillover_logits()

        self._maybe_swap_network(state, t)
        time_step_one_hot = self._generate_one_hot_tensor(t, self.num_timesteps)

        week_id = int(t / 7)
        week_one_hot = self._generate_one_hot_tensor(week_id, self.num_weeks)

        if self.calibration_mode:
            R_tensor = self.calibrate_R2
        else:
            R_tensor = self.learnable_args["R2"]
        R = (R_tensor * week_one_hot).sum()

        if self._seasonal is not None:
            seasonal_mult = self._seasonal(t, self._sim_start_doy)
            R = R * seasonal_mult

        SFSusceptibility = get_by_path(
            state, re.split("/", input_variables["SFSusceptibility"])
        ).detach()
        SFInfector = get_by_path(state, re.split("/", input_variables["SFInfector"])).detach()
        all_lam_gamma = get_by_path(
            state, re.split("/", input_variables["lam_gamma_integrals"])
        ).detach()

        agents_infected_time = get_by_path(
            state, re.split("/", input_variables["infected_time"])
        ).detach()
        agents_mean_interactions_split = get_by_path(
            state, re.split("/", input_variables["mean_interactions"])
        ).detach()
        agents_ages = get_by_path(state, re.split("/", input_variables["age"])).detach()
        if not hasattr(self, '_age_diag_done'):
            _nag = getattr(self, 'n_age_groups', 8); a = agents_ages.squeeze().long(); bad = ((a < 0) | (a >= _nag)).sum().item(); print(f"[diag] age groups: {_nag}, invalid ages: {bad}/{a.shape[0]} ({100*bad/a.shape[0]:.2f}%), unique invalid: {a[(a<0)|(a>=_nag)].unique().tolist()}"); self._age_diag_done = True

        if not self._municipality_initialized:
            self._init_municipality_data(agents_ages.shape[0])

        if not hasattr(self, '_vaccine_susc') or self._vaccine_susc is None:
            n_ag = agents_ages.shape[0]
            self._vaccine_susc = torch.ones(n_ag, device=self.device)
            self._vaccine_death_factor = torch.ones(n_ag, device=self.device)
            from danish_abm import runtime_set
            runtime_set(self.config, "vaccine_susceptibility", self._vaccine_susc)
            runtime_set(self.config, "vaccine_death_factor", self._vaccine_death_factor)
        elif t == 0:
            self._vaccine_susc.fill_(1.0)
            self._vaccine_death_factor.fill_(1.0)

        current_stages = get_by_path(
            state, re.split("/", input_variables["disease_stage"])
        ).detach()
        current_transition_times = get_by_path(
            state, re.split("/", input_variables["next_stage_time"])
        ).detach()

        all_edgelist, all_edgeattr = get_by_path(
            state, re.split("/", input_variables["adjacency_matrix"])
        )

        if self.device.type == 'cuda':
            if t == 0:
                _a = torch.cuda.memory_allocated() / 1e9
                print(f"  [GPU t=0] before edge compact: {_a:.2f}GB, "
                      f"edge_index={all_edgelist.dtype} {all_edgelist.shape} ({all_edgelist.nelement()*all_edgelist.element_size()/1e9:.2f}GB), "
                      f"edge_attr={all_edgeattr.dtype} {all_edgeattr.shape} ({all_edgeattr.nelement()*all_edgeattr.element_size()/1e9:.2f}GB)",
                      flush=True)
            adj_path = re.split("/", input_variables["adjacency_matrix"])
            def _set_adj(ei, ea):
                node = state
                for p in adj_path[:-1]:
                    node = node[p]
                node[adj_path[-1]] = (ei, ea)

            if all_edgelist.dtype == torch.int64:
                all_edgelist = all_edgelist.int()
                _set_adj(all_edgelist, all_edgeattr)
                torch.cuda.empty_cache()
            if all_edgeattr.dtype not in (torch.float16, torch.int32):
                all_edgeattr = all_edgeattr.half()
                _set_adj(all_edgelist, all_edgeattr)
                torch.cuda.empty_cache()
            if t == 0:
                _a = torch.cuda.memory_allocated() / 1e9
                print(f"  [GPU t=0] after edge compact: {_a:.2f}GB", flush=True)

        daily_infected = get_by_path(
            state, re.split("/", input_variables["daily_infected"])
        ).detach()

        cm = self._compartment_model
        if cm is not None:
            agents_infected_index = cm.infectious_mask(current_stages).bool()
        else:
            agents_infected_index = torch.logical_and(
                current_stages > self.SUSCEPTIBLE_VAR, current_stages < self.RECOVERED_VAR
            )

        if self.reinfection_susceptibility < 1.0:
            stages_flat = current_stages.squeeze()
            if cm is not None:
                ever_infected = cm.infectious_mask(stages_flat).bool()
                for idx in cm.removed_indices:
                    ever_infected = ever_infected | (stages_flat == idx)
                susc_set = set(cm.susceptible_indices)
                for i in range(cm.num_states):
                    if i not in susc_set and i not in cm.removed_indices:
                        ever_infected = ever_infected | (stages_flat == i)
            else:
                ever_infected = (stages_flat >= self.RECOVERED_VAR) & (stages_flat <= 4)
            if self._previously_infected is None:
                self._previously_infected = ever_infected.bool()
            else:
                self._previously_infected = self._previously_infected | ever_infected.bool()

        isolation_level = self._isolation_level_from_state(state, current_stages.shape[0])
        if self.isolation_in_transmission:
            n_iso = int((isolation_level > 0).sum().item())
            if n_iso > 0 and not self._isolation_reported:
                print(f"  [transmission] day {t}: {n_iso:,} agents isolating; their "
                      f"transmission outside the household is reduced by "
                      f"{self.isolation_reduction:.2f}", flush=True)
                self._isolation_reported = True
        self._isolation_level = isolation_level
        will_isolate = isolation_level

        _node_feats = (
            agents_ages,
            current_stages.detach(),
            agents_infected_index,
            agents_infected_time,
            agents_mean_interactions_split,
            will_isolate,
        )
        if self.device.type == 'cuda':
            all_node_attr = torch.stack([f.half().reshape(-1) for f in _node_feats], dim=1)
        else:
            all_node_attr = torch.stack(_node_feats).transpose(0, 1).squeeze()

        num_nodes = all_node_attr.size(0) if hasattr(all_node_attr, 'size') else None
        agents_data = Data(
            all_node_attr, edge_index=all_edgelist, edge_attr=all_edgeattr, t=t
        )
        if num_nodes is not None:
            agents_data.num_nodes = int(num_nodes)

        if self.calibration_mode:
            q_layers = self.calibrate_q_layers.clamp(min=0)
        else:
            q_layers = self.learnable_args["q_layers"].clamp(min=0)

        if self.has_interventions and t < self.intervention_schedule.shape[0]:
            mult = self.intervention_schedule[t]
            q_layers = q_layers.squeeze() * mult[:self.num_layers]

            if self.npi_active is not None and t < self.npi_active.shape[0]:
                active = self.npi_active[t]
                if active.any():
                    if self.calibration_mode and hasattr(self, 'calibrate_spillover_logits'):
                        spillover_logits = self.calibrate_spillover_logits
                    elif not self.calibration_mode and "spillover_logits" in self.learnable_args:
                        spillover_logits = self.learnable_args["spillover_logits"]
                    else:
                        spillover_logits = None

                    if spillover_logits is not None:
                        spillover = torch.exp(spillover_logits)
                        n_layers_spill = min(spillover.shape[1], self.num_layers)
                        active_spill = torch.where(
                            active.unsqueeze(1).expand_as(spillover) > 0.5,
                            spillover,
                            torch.ones_like(spillover),
                        )
                        layer_spillover = active_spill.prod(dim=0)
                        _ss = getattr(self, "spillover_scale", 1.0)
                        if _ss != 1.0:
                            layer_spillover = torch.exp(
                                _ss * torch.log(layer_spillover.clamp(min=1e-12)))
                        q_layers[:n_layers_spill] = q_layers[:n_layers_spill] * layer_spillover[:n_layers_spill]
        else:
            q_layers = q_layers.squeeze()

        from danish_abm import runtime_get
        trigger_mult = runtime_get(self.config, "trigger_dynamic_multiplier")
        if trigger_mult is not None:
            q_layers = q_layers * trigger_mult[:self.num_layers]

        R_clamped = R.clamp(min=0)
        lam_gamma = all_lam_gamma.squeeze()

        if self.device.type == 'cuda':
            sf_susc = SFSusceptibility.half()
            sf_inf = SFInfector.half()
            lam_gamma = lam_gamma.half()

            edge_index = all_edgelist
            edge_attr = all_edgeattr
            x = all_node_attr
            num_edges = edge_index.shape[1]
            chunk_size = self._edge_chunk_size

            del agents_data, all_node_attr, all_edgeattr, all_edgelist
            torch.cuda.empty_cache()

            new_transmission = torch.zeros(num_nodes, 1, device=self.device)

            for start in range(0, num_edges, chunk_size):
                end = min(start + chunk_size, num_edges)
                ei = edge_index[:, start:end]
                ea = edge_attr[:, start:end]

                x_i = x[ei[1]]
                x_j = x[ei[0]]

                msg = self._lam(
                    x_i, x_j, ea, t, R_clamped, q_layers,
                    sf_susc, sf_inf, lam_gamma,
                )
                msg = self._apply_telework(msg, ei, ea, t, edge_index, edge_attr)

                idx = ei[1].long().unsqueeze(1).expand_as(msg)
                new_transmission.scatter_add_(0, idx, msg.float())


                if self.symmetric_transmission:
                    msg_r = self._lam(
                        x_j, x_i, ea, t, R_clamped, q_layers,
                        sf_susc, sf_inf, lam_gamma,
                    )
                    msg_r = self._apply_telework(msg_r, ei, ea, t, edge_index, edge_attr)
                    idx_r = ei[0].long().unsqueeze(1).expand_as(msg_r)
                    new_transmission.scatter_add_(0, idx_r, msg_r.float())
                    self._accumulate_layer_pressure(ea, msg, msg_r)
                    del msg_r, idx_r
                else:
                    self._accumulate_layer_pressure(ea, msg)

                del x_i, x_j, msg, ei, ea, idx
                torch.cuda.empty_cache()

            del edge_attr, x
            torch.cuda.empty_cache()
        else:
            edge_index = all_edgelist
            edge_attr = all_edgeattr
            x = all_node_attr
            num_edges = edge_index.shape[1]
            chunk_size = self._edge_chunk_size

            del agents_data, all_node_attr, all_edgeattr, all_edgelist
            import gc; gc.collect()

            R_for_msg = R_clamped.detach()
            q_for_msg = q_layers.detach()

            with torch.no_grad():
                new_transmission = torch.zeros(num_nodes, 1, device=self.device)

                for start in range(0, num_edges, chunk_size):
                    end = min(start + chunk_size, num_edges)
                    ei = edge_index[:, start:end]
                    ea = edge_attr[:, start:end]

                    x_i = x[ei[1]]
                    x_j = x[ei[0]]

                    msg = self._lam(
                        x_i, x_j, ea, t, R_for_msg, q_for_msg,
                        SFSusceptibility, SFInfector, lam_gamma,
                    )
                    msg = self._apply_telework(msg, ei, ea, t, edge_index, edge_attr)

                    idx = ei[1].long().unsqueeze(1).expand_as(msg)
                    new_transmission.scatter_add_(0, idx, msg)

                    if self.symmetric_transmission:
                        msg_r = self._lam(
                            x_j, x_i, ea, t, R_for_msg, q_for_msg,
                            SFSusceptibility, SFInfector, lam_gamma,
                        )
                        msg_r = self._apply_telework(msg_r, ei, ea, t, edge_index, edge_attr)
                        idx_r = ei[0].long().unsqueeze(1).expand_as(msg_r)
                        new_transmission.scatter_add_(0, idx_r, msg_r)
                        self._accumulate_layer_pressure(ea, msg, msg_r)
                        del msg_r, idx_r
                    else:
                        self._accumulate_layer_pressure(ea, msg)

                    del x_i, x_j, msg, ei, ea, idx

            del edge_attr, x
            _q_ratio = torch.ones((), device=new_transmission.device)
            if self._step_layer_pressure is not None:
                _lp = self._step_layer_pressure
                _tot = float(_lp.sum())
                if _tot > 0:
                    _frac = (_lp / _tot).to(new_transmission.dtype)
                    _q_flat = q_layers.reshape(-1)
                    _n_l = min(_frac.numel(), _q_flat.numel())
                    _q_ratio = (
                        _frac[:_n_l]
                        * (_q_flat[:_n_l] / (_q_flat[:_n_l].detach() + 1e-12))
                    ).sum()

            new_transmission = new_transmission * (R_clamped / (R_for_msg + 1e-12)) * _q_ratio

        if self.has_community_mixing:
            if self.calibration_mode:
                R_comm = self.calibrate_R_community.clamp(min=0)
            else:
                R_comm = self.learnable_args["R_community"].clamp(min=0)
            if self.has_interventions and t < self.intervention_schedule.shape[0]:
                R_comm = R_comm * self.intervention_schedule[t, self.num_layers]
            if trigger_mult is not None:
                R_comm = R_comm * trigger_mult[self.num_layers]
            community_term = self._community_lambda(
                agents_ages, current_stages, (R_comm * R_clamped).squeeze(),
                isolation=(self._isolation_level if self.isolation_in_transmission
                           else None))
            _susc_age = SFSusceptibility[
                agents_ages.squeeze().long().clamp(0, SFSusceptibility.shape[0] - 1)
            ].reshape(-1, 1).to(community_term.dtype)
            new_transmission = new_transmission + community_term * _susc_age

            with torch.no_grad():
                if self._layer_pressure is None:
                    self._layer_pressure = torch.zeros(self.num_layers + 1,
                                                       dtype=torch.float64,
                                                       device=community_term.device)
                self._layer_pressure[self.num_layers] += float(
                    (community_term.detach() * _susc_age).sum().item()
                )

        if self.has_interventions and t < self.intervention_schedule.shape[0]:
            susc_mult = self.intervention_schedule[t, self.num_layers + 1]
            new_transmission = new_transmission * susc_mult
        if trigger_mult is not None:
            new_transmission = new_transmission * trigger_mult[self.num_layers + 1]

        if self._multi_strain:
            n_agents = int(new_transmission.shape[0])
            if self._strain_manager is None:
                self._init_multi_strain(n_agents, current_stages)
            active = self._strain_config.active_strains(t)
            if active:
                dominant = max(active, key=lambda s: (
                    self._strain_manager.current_strain == self._strain_config.strain_idx[s.name]
                ).sum().item())
                d_idx = self._strain_config.strain_idx[dominant.name]
                ci_col = self._antigen_cross_immunity[:, d_idx]
                susc = self._immunity_tracker.effective_susceptibility(
                    dominant.name, ci_col, t
                )
                new_transmission = new_transmission * susc.unsqueeze(1)
                new_transmission = new_transmission * dominant.transmissibility

        if self.reinfection_susceptibility < 1.0 and not self._multi_strain:
            if self._previously_infected is None:
                self._previously_infected = torch.zeros(
                    new_transmission.shape[0], device=self.device, dtype=torch.bool
                )
            if cm is not None:
                reinf_mask = self._previously_infected & cm.susceptible_mask(current_stages).squeeze().bool()
            else:
                reinf_mask = self._previously_infected & (current_stages.squeeze() == self.SUSCEPTIBLE_VAR)
            if reinf_mask.any():
                new_transmission[reinf_mask] *= self.reinfection_susceptibility

        if hasattr(self, '_vaccine_susc') and self._vaccine_susc is not None and not self._multi_strain:
            new_transmission = new_transmission * self._vaccine_susc.unsqueeze(1)

        new_transmission = new_transmission.clamp(min=0)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        prob_not_infected = torch.exp(-1 * new_transmission)
        p_infected = (1 - prob_not_infected).clamp(0, 1)
        del prob_not_infected

        potentially_exposed_today = self.st_bernoulli(p_infected).squeeze(-1).to(
            self.device
        )
        del p_infected

        if cm is not None:
            susc_mask = cm.susceptible_mask(current_stages).squeeze()
        else:
            susc_mask = (current_stages == self.SUSCEPTIBLE_VAR).squeeze()
        newly_exposed_today = susc_mask * potentially_exposed_today

        if self._record_trajectory and num_nodes <= 1_000_000:
            self._newly_exposed_ste.append(newly_exposed_today.squeeze().clone())

        import_rate = self.config["simulation_metadata"].get("daily_importation_rate", 0.0)
        if import_rate > 0:
            if cm is not None:
                susceptible_mask = cm.susceptible_mask(current_stages).squeeze().bool() & (newly_exposed_today == 0)
            else:
                susceptible_mask = (current_stages.squeeze() == self.SUSCEPTIBLE_VAR) & (newly_exposed_today == 0)
            n_susceptible = susceptible_mask.sum().item()
            if n_susceptible > 0:
                n_import = max(1, int(import_rate * n_susceptible))
                susc_idx = susceptible_mask.nonzero(as_tuple=True)[0]
                sel = torch.randint(len(susc_idx), (n_import,), device=self.device)
                newly_exposed_today[susc_idx[sel]] = 1.0

        if self._multi_strain and self._strain_manager is not None:
            self._strain_manager.assign_new_infections_community(newly_exposed_today, t)
            self._strain_manager.track_daily(t)

        daily_infected.add_(newly_exposed_today.sum() * time_step_one_hot)

        newly_exposed_today = newly_exposed_today.unsqueeze(1)

        with torch.no_grad():
            n_new = newly_exposed_today.sum().item()
            stages_flat = current_stages.squeeze()

            if cm is not None:
                n_infectious = cm.infectious_mask(stages_flat).sum().item()
            else:
                n_infectious = (stages_flat == self.INFECTED_VAR).sum().item()

            infectious_period = float(self.config["simulation_metadata"]["INFECTED_TO_RECOVERED_TIME"])
            r_eff = (n_new / max(n_infectious, 1)) * infectious_period
            self.tracking["R_eff"].append(r_eff)
            self.tracking["new_infections"].append(n_new)

            self.tracking["S"].append((stages_flat == self.SUSCEPTIBLE_VAR).sum().item())
            self.tracking["E"].append((stages_flat == self.EXPOSED_VAR).sum().item())
            self.tracking["I"].append(n_infectious)
            self.tracking["R"].append((stages_flat == self.RECOVERED_VAR).sum().item())
            self.tracking["M"].append((stages_flat == self.config["simulation_metadata"]["MORTALITY_VAR"]).sum().item())

            if cm is not None:
                for name in cm.compartment_names:
                    if name not in ("S", "E", "I", "R", "M") and name in self.tracking:
                        idx = cm.idx(name)
                        self.tracking[name].append((stages_flat == idx).sum().item())

            trigger_engine = runtime_get(self.config, "trigger_engine")
            if trigger_engine is not None:
                self.tracking["active_triggers"].append(trigger_engine.get_active_policies())
            else:
                self.tracking["active_triggers"].append([])

            newly_flat = newly_exposed_today.squeeze() if newly_exposed_today.dim() > 1 else newly_exposed_today
            ages_flat_track = agents_ages.squeeze().long()
            n_age_groups = 8
            age_counts = torch.zeros(n_age_groups, device=self.device)
            newly_mask = newly_flat > 0
            if newly_mask.any():
                valid_age = (ages_flat_track >= 0) & (ages_flat_track < n_age_groups)
                combined = newly_mask & valid_age
                if combined.any():
                    age_counts.scatter_add_(0, ages_flat_track[combined], newly_flat[combined].float())
            self.tracking["new_infections_by_age"].append(age_counts.cpu().tolist())

            if self._municipality_mixing and self._municipality_ids is not None:
                if "municipality_infections" not in self.tracking:
                    self.tracking["municipality_infections"] = []
                if "municipality_infections_diff" not in self.tracking:
                    self.tracking["municipality_infections_diff"] = []
                newly_flat = newly_exposed_today.squeeze() if newly_exposed_today.dim() > 1 else newly_exposed_today
                muni_new = torch.zeros(self._n_municipalities, device=self.device)
                muni_new.scatter_add_(0, self._municipality_ids, newly_flat.float())
                self.tracking["municipality_infections"].append(muni_new.cpu().tolist())
                self.tracking["municipality_infections_diff"].append(muni_new.detach())

        if self._test_ids_by_day is not None:
            _ids = self._test_ids_by_day.get(int(t))
            if _ids is not None and _ids.numel() > 0:
                self._tested_stages[int(t)] = (
                    current_stages.squeeze()[_ids.to(current_stages.device)]
                    .to(torch.int8).cpu()
                )

        if self._record_trajectory:
            n_agents = int(current_stages.shape[0])
            if n_agents <= 1_000_000:
                if self._trajectory is None:
                    self._trajectory = torch.zeros(
                        self.num_timesteps, n_agents, dtype=torch.int8, device='cpu'
                    )
                if t < self.num_timesteps:
                    self._trajectory[t] = current_stages.squeeze().to(torch.int8).cpu()

        updated_stages = self.update_stages(current_stages, newly_exposed_today)
        updated_next_stage_times = self.update_transition_times(
            t, current_transition_times, newly_exposed_today
        )
        updated_infected_times = self.update_infected_times(
            t, agents_infected_time, newly_exposed_today
        )

        if t < 3 or t % 10 == 0:
            with torch.no_grad():
                _us = updated_stages.squeeze()
                _ut = updated_next_stage_times.squeeze()
                _nE_out = (_us == self.EXPOSED_VAR).sum().item()
                _nE_new = newly_exposed_today.sum().item()
                _eReady_out = ((_us == self.EXPOSED_VAR) & (_ut <= t)).sum().item()
                _eFuture_out = ((_us == self.EXPOSED_VAR) & (_ut > t) & (_ut < 500)).sum().item()
                print(f"  [tx t={t:3d} OUT] E_total={_nE_out} newly_exposed={_nE_new:.0f} "
                      f"E_ready={_eReady_out} E_future={_eFuture_out}", flush=True)

        if t >= self.num_timesteps - 1:
            self.report_layer_pressure()

        return {
            self.output_variables[0]: updated_stages.detach(),
            self.output_variables[1]: updated_next_stage_times.detach(),
            self.output_variables[2]: updated_infected_times.detach(),
            self.output_variables[3]: daily_infected.detach(),
        }
