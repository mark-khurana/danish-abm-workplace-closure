import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import gamma
import networkx as nx
import torch.nn.functional as F
import pdb


class _ResilientStream:
    """Wraps a text stream so transient OSErrors on write/flush don't crash."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        try:
            return self._stream.write(data)
        except OSError:
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            return self._stream.flush()
        except OSError:
            return None

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install_resilient_stdio():
    """Make sys.stdout/sys.stderr tolerant of transient I/O errors."""
    if not isinstance(sys.stdout, _ResilientStream):
        sys.stdout = _ResilientStream(sys.stdout)
    if not isinstance(sys.stderr, _ResilientStream):
        sys.stderr = _ResilientStream(sys.stderr)

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

LAYER_NAMES = ["family", "household", "neighborhood", "colleague", "classmate"]
LAYER_NAME_TO_IDX = {name: i for i, name in enumerate(LAYER_NAMES)}
NUM_STRUCTURED_LAYERS = 5


def configure_layers(layer_names):
    """Reconfigure global layer constants from a pathogen preset's network_layers."""
    global LAYER_NAMES, LAYER_NAME_TO_IDX, NUM_STRUCTURED_LAYERS
    LAYER_NAMES = list(layer_names)
    LAYER_NAME_TO_IDX = {name: i for i, name in enumerate(LAYER_NAMES)}
    NUM_STRUCTURED_LAYERS = len(LAYER_NAMES)


def get_layer_config(config=None):
    """Get layer names from config (if available) or return module defaults."""
    if config is not None:
        meta = config.get("simulation_metadata", {})
        names = meta.get("network_layer_names")
        if names:
            return list(names), {name: i for i, name in enumerate(names)}, len(names)
    return LAYER_NAMES, LAYER_NAME_TO_IDX, NUM_STRUCTURED_LAYERS


def get_lam_gamma_integrals(shape, params):
    scale, rate = params["scale"], params["rate"]
    t = params["t"]
    b = rate * rate / scale
    a = scale / b

    a, b, t = a.cpu(), b.cpu(), t.cpu()

    res = [
        (gamma.cdf(t_i, a=a, loc=0, scale=b) - gamma.cdf(t_i - 1, a=a, loc=0, scale=b))
        for t_i in range(int(t.item()))
    ]
    res = np.array(res)
    return torch.tensor(res).float()


def _stage_fraction(n, device=None):
    """Deterministic per-agent uniform in [0, 1), reproducible across processes."""
    i = torch.arange(n, dtype=torch.int64, device=device)
    return ((i * 2654435761) % 4294967296).double().div(4294967296.0)


def _seeded_residual(params, n):
    """Per-agent time left in the current stage, read from a file beside the stage file."""
    fp = params.get("file_path")
    if not fp:
        return None
    rp = os.path.join(os.path.dirname(fp), "disease_stages_residual.csv")
    if not os.path.exists(rp):
        return None
    try:
        v = pd.read_csv(rp, header=None).to_numpy(dtype=float).reshape(-1)
    except Exception:
        return None
    if v.size < n:
        return None
    return torch.from_numpy(v[:n].copy())


def _durations(params, exposed_to_infected_times, infected_to_recovered_times):
    """Stage durations, preferring the wave's own values over the 3/5 fallback."""
    lat = params.get("latent_days", exposed_to_infected_times)
    inf = params.get("infectious_days", infected_to_recovered_times)
    return float(lat), float(inf)


def get_infected_time(shape, params, exposed_to_infected_times=3,
                      infected_to_recovered_times=5):
    agents_stages = read_from_file(shape, params)
    lat, inf = _durations(params, exposed_to_infected_times, infected_to_recovered_times)

    agents_infected_time = (500) * torch.ones_like(
        agents_stages
    )

    flat = agents_stages.reshape(-1)
    frac = _stage_fraction(flat.numel(), device=flat.device)
    res = _seeded_residual(params, flat.numel())
    out = agents_infected_time.reshape(-1).double()
    e = flat == 1
    out[e] = -(frac[e] * lat)
    i_ = flat == 2
    out[i_] = -(lat + frac[i_] * inf)
    if res is not None:
        he = (res >= 0) & e
        out[he] = -(lat - res[he].double())
        hi = (res >= 0) & i_
        out[hi] = -(lat + inf - res[hi].double())

    return out.reshape(agents_stages.shape).float()


def get_next_stage_time(
    shape, params, exposed_to_infected_times=3, infected_to_recovered_times=5
):
    agents_stages = read_from_file(shape, params)
    lat, inf = _durations(params, exposed_to_infected_times, infected_to_recovered_times)

    agents_next_stage_time = (500) * torch.ones_like(
        agents_stages
    )

    flat = agents_stages.reshape(-1)
    frac = _stage_fraction(flat.numel(), device=flat.device)
    res = _seeded_residual(params, flat.numel())
    nxt = agents_next_stage_time.reshape(-1).double()
    e = flat == 1
    nxt[e] = torch.round((1.0 - frac[e]) * lat).clamp(min=1.0)
    i_ = flat == 2
    nxt[i_] = torch.round((1.0 - frac[i_]) * inf).clamp(min=1.0)
    if res is not None:
        have = (res >= 0) & (flat > 0)
        nxt[have] = torch.round(res[have].double()).clamp(min=1.0)
    agents_next_stage_time = nxt.reshape(agents_stages.shape).to(agents_next_stage_time.dtype)

    return agents_next_stage_time.float()


def initialize_infections(shape, params):
    num_agents = params["num_agents"]
    initial_infections_ratio = params["initial_infected_ratio"]

    prob_infected = initial_infections_ratio * torch.ones((num_agents, 1))
    p = torch.hstack((prob_infected, 1 - prob_infected))
    cat_logits = torch.log(p + 1e-9)
    agents_stages = F.gumbel_softmax(logits=cat_logits, tau=1, hard=True, dim=1)[:, 0]

    return agents_stages.unsqueeze(1)


def _load_cumulative_deaths_before(gt_dir, before_date):
    """Sum real-world SSI deaths before a date (NOT scaled to model pop)."""
    from pathlib import Path
    deaths_csv = Path(gt_dir) / "Deaths_over_time.csv"
    if not deaths_csv.exists():
        return 0
    d = pd.read_csv(deaths_csv, sep=";", encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    d = d[~d["Dato"].str.contains("alt|person", case=False, na=False)]
    d["Dato"] = pd.to_datetime(d["Dato"], errors="coerce")
    d["Antal_døde"] = pd.to_numeric(d["Antal_døde"], errors="coerce").fillna(0)
    return float(d[d["Dato"] < pd.Timestamp(before_date)]["Antal_døde"].sum())


_prior_immunity_cache = {}


def apply_prior_immunity(runner, data_config=None, meta=None,
                         prior_immune_fraction=0.0, prior_immunity_scale=None,
                         verbose=True):
    """Apply prior-wave immunity with date-aware waning."""
    disease_stage = runner.state["agents"]["citizens"]["disease_stage"]
    next_stage_time = runner.state["agents"]["citizens"]["next_stage_time"]
    device = disease_stage.device
    n = disease_stage.shape[0]

    if meta is None:
        meta = {}
    waning_cfg = meta.get("disease_params", {}).get("immunity_waning", {})
    if isinstance(waning_cfg, dict):
        waning_days = waning_cfg.get("mean", meta.get("RECOVERED_TO_SUSCEPTIBLE_TIME", 180))
    else:
        waning_days = float(waning_cfg) if waning_cfg else 180.0
    waning_days = float(waning_days)

    sim_start_str = data_config.start_date if data_config else None
    _empty_proxy = {"n_detected": 0, "n_still_r_detected": 0,
                     "waning_days": 180.0, "n_agents": n, "susc_frac_fixed": 1.0}
    if sim_start_str is None:
        if prior_immune_fraction > 0:
            n_fb, wm_fb = _apply_aggregate_fallback(
                disease_stage, prior_immune_fraction, n, device, verbose)
            return n_fb, wm_fb, _empty_proxy
        return 0, None, _empty_proxy

    sim_start = pd.Timestamp(sim_start_str)
    cache_key = (sim_start_str, n)

    if cache_key not in _prior_immunity_cache:
        detected = {}
        cum_deaths_real = 0.0

        if data_config.individual_test_csv:
            try:
                from ..agent_properties import read_tabular
                test_df = read_tabular(data_config.individual_test_csv)
                col_map = {}
                for c in test_df.columns:
                    cl = c.strip().lower().replace(" ", "_")
                    if cl in ("date", "test_date", "dato"):
                        col_map[c] = "date"
                    elif cl in ("agent_id", "id", "person_id"):
                        col_map[c] = "agent_id"
                    elif cl in ("result", "positive", "pos"):
                        col_map[c] = "result"
                test_df = test_df.rename(columns=col_map)
                test_df["date"] = pd.to_datetime(test_df["date"])

                try:
                    from ..registry_data import _build_agent_id_map
                except ImportError:
                    from danish_abm.registry_data import _build_agent_id_map
                import os.path as _osp
                _tp = str(data_config.individual_test_csv)
                _id_map = None
                for _cand in (_osp.join(_osp.dirname(_osp.dirname(_tp)), "agents.parquet"),
                              _osp.join(_osp.dirname(_tp), "agents.parquet")):
                    _id_map = _build_agent_id_map(_cand)
                    if _id_map is not None:
                        break
                if _id_map is not None:
                    _n_before = len(test_df)
                    test_df = test_df[test_df["agent_id"].isin(set(_id_map))].copy()
                    test_df["agent_id"] = test_df["agent_id"].map(_id_map).astype("int64")
                    if verbose:
                        print(f"  Prior immunity: id map applied, kept "
                              f"{len(test_df):,} of {_n_before:,} register rows")
                elif len(test_df) and int(test_df["agent_id"].max()) >= n:
                    raise SystemExit(
                        f"[prior immunity] no agents.parquet beside {_tp}, and the "
                        f"register's ids run to {int(test_df['agent_id'].max()):,} against "
                        f"n={n:,}. Truncating them would silently understate prior immunity "
                        f"-- that is the 14 September 2026 defect. Put agents.parquet in "
                        f"the wave directory."
                    )

                pre = test_df[
                    (test_df["date"] < sim_start) &
                    (test_df["result"].astype(str).isin(
                        ["1", "positive", "pos", "True"]))
                ]
                if len(pre) > 0:
                    latest = pre.sort_values("date").groupby("agent_id").last()
                    for aid, row in latest.iterrows():
                        if aid < n:
                            detected[int(aid)] = row["date"]
            except Exception as e:
                if verbose:
                    print(f"  Prior immunity: could not load test CSV: {e}")

        if data_config.ground_truth_dir:
            cum_deaths_real = _load_cumulative_deaths_before(
                data_config.ground_truth_dir, sim_start)

        _prior_immunity_cache[cache_key] = {
            "detected": detected,
            "cum_deaths_real": cum_deaths_real,
        }
    else:
        detected = _prior_immunity_cache[cache_key]["detected"]
        cum_deaths_real = _prior_immunity_cache[cache_key]["cum_deaths_real"]

    n_undetected = 0
    _ar = prior_immunity_scale if prior_immunity_scale is not None else meta.get("prior_immunity_scale", None)
    if _ar is not None and len(detected) > 0:
        ar = max(float(_ar), 0.01)
        total_true_model = int(len(detected) / ar)
        n_undetected = max(0, total_true_model - len(detected))
        if verbose:
            print(f"  Prior immunity (PIS={ar:.3f}): "
                  f"detected={len(detected):,} → "
                  f"true_inf={total_true_model:,}, "
                  f"undetected={n_undetected:,}")
    elif cum_deaths_real > 0:
        ifr_cfg = meta.get("ifr", {})
        ifr = ifr_cfg.get("mean", 0.005) if isinstance(ifr_cfg, dict) else float(ifr_cfg or 0.005)
        total_true_real = cum_deaths_real / max(ifr, 1e-6)
        scale = data_config.scale_factor
        total_true_model = int(total_true_real * scale)
        n_undetected = max(0, total_true_model - len(detected))
        if verbose:
            print(f"  Prior immunity (death back-calc): "
                  f"deaths_before={cum_deaths_real:.0f} (real), "
                  f"IFR={ifr:.4f} → "
                  f"true_inf={total_true_real:.0f} (real), "
                  f"model_scale={total_true_model:,}, "
                  f"detected={len(detected):,}, "
                  f"undetected={n_undetected:,}")

    if not detected and n_undetected == 0:
        if prior_immune_fraction > 0:
            n_fb, wm_fb = _apply_aggregate_fallback(
                disease_stage, prior_immune_fraction, n, device, verbose)
            return n_fb, wm_fb, _empty_proxy
        return 0, None, _empty_proxy

    n_still_r = 0
    n_waned = 0
    waned_mask = torch.zeros(n, dtype=torch.bool, device=device)

    for aid, inf_date in detected.items():
        days_since = (sim_start - inf_date).days
        if days_since < waning_days:
            if float(disease_stage[aid, 0]) != 0.0:
                continue
            disease_stage[aid, 0] = 3.0
            remaining = waning_days - days_since
            next_stage_time[aid, 0] = float(remaining)
            n_still_r += 1
        else:
            disease_stage[aid, 0] = 0.0
            waned_mask[aid] = True
            n_waned += 1

    if n_undetected > 0:
        susceptible = (disease_stage.squeeze() == 0).nonzero(as_tuple=True)[0]
        susceptible = susceptible[~waned_mask[susceptible]]
        n_assign = min(n_undetected, len(susceptible))
        if n_assign > 0:
            idx = susceptible[torch.randperm(len(susceptible), device=device)[:n_assign]]
            days_ago = torch.randint(1, 183, (n_assign,), device=device).float()
            still_r = days_ago < waning_days
            n_undetected_r = still_r.sum().item()
            n_undetected_waned = n_assign - n_undetected_r

            r_idx = idx[still_r]
            if len(r_idx) > 0:
                disease_stage[r_idx, 0] = 3.0
                remaining = waning_days - days_ago[still_r]
                next_stage_time[r_idx, 0] = remaining
            n_still_r += n_undetected_r

            w_idx = idx[~still_r]
            if len(w_idx) > 0:
                waned_mask[w_idx] = True
            n_waned += n_undetected_waned

    n_total = n_still_r + n_waned

    if verbose and n_total > 0:
        print(f"  Prior immunity: {len(detected):,} detected + "
              f"{min(n_undetected, n_total - len(detected)):,} undetected = "
              f"{n_total:,} total ({n_total/n*100:.1f}%)")
        print(f"    Still RECOVERED: {n_still_r:,}, "
              f"Waned to S (reduced reinfection): {n_waned:,}, "
              f"waning_days={waning_days:.0f}")

    n_set_to_r = (disease_stage.squeeze() == 3.0).sum().item()
    proxy_info = {
        "n_detected": len(detected),
        "n_still_r_detected": sum(
            1 for aid, d in detected.items()
            if (sim_start - d).days < waning_days
        ),
        "waning_days": waning_days,
        "n_agents": n,
        "susc_frac_fixed": 1.0 - n_set_to_r / max(n, 1),
    }

    return n_total, waned_mask, proxy_info


def _apply_aggregate_fallback(disease_stage, fraction, n, device, verbose):
    """Fallback: set a fraction of random susceptible agents to RECOVERED."""
    n_immune = int(fraction * n)
    susceptible = (disease_stage.squeeze() == 0).nonzero(as_tuple=True)[0]
    if len(susceptible) == 0:
        return 0, None
    n_set = min(n_immune, len(susceptible))
    idx = susceptible[torch.randperm(len(susceptible), device=device)[:n_set]]
    disease_stage[idx, 0] = 3.0
    if verbose:
        print(f"  Prior immunity (fallback): {n_set:,}/{n:,} agents → RECOVERED "
              f"({n_set/n*100:.1f}%)")
    return n_set, None


def read_from_file(shape, params):
    file_path = params["file_path"]
    if file_path.endswith((".parquet", ".pq")):
        df = pd.read_parquet(file_path)
        values = df.values
    elif file_path.endswith("csv"):
        df = pd.read_csv(file_path, header=None)
        values = pd.to_numeric(df.stack(), errors='coerce').unstack().to_numpy()
        if np.isnan(values[0]).any():
            values = values[1:]
    else:
        df = pd.read_pickle(file_path)
        values = df.values
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape != tuple(shape):
        print(f"  WARNING: {file_path} has shape {values.shape} but config expects {tuple(shape)}")
    return torch.from_numpy(values)


IBAR_ENV_VAR = "DANISH_ABM_IBAR"


def parse_ibar_override(raw):
    """Validate ``DANISH_ABM_IBAR``. Returns None when unset, else a positive float."""
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"{IBAR_ENV_VAR}={raw!r} is not a number. Unset it to use the age-band "
            f"default (2 child / 3 adult / 4 elderly), or set a positive value."
        )
    if not (value > 0) or value != value or value == float("inf"):
        raise ValueError(
            f"{IBAR_ENV_VAR}={raw!r} must be finite and > 0; _lam divides by it."
        )
    return value


def get_mean_agent_interactions(shape, params):
    """I_bar, the per-agent divisor in the hazard (`_lam`: ... * integrals / I_bar(i))."""
    agents_ages = load_population_attribute(shape, params)

    ADULT_LOWER_INDEX, ADULT_UPPER_INDEX = (
        2,
        6,
    )

    agents_mean_interactions = 0 * torch.ones(size=shape)
    mean_int_ran_mu = torch.tensor([2, 3, 4]).float()

    _override = parse_ibar_override(os.environ.get(IBAR_ENV_VAR))
    if _override is not None:
        print(f"  [I_bar] {IBAR_ENV_VAR}={_override:g} -- uniform for all "
              f"{int(np.prod(tuple(shape)))} agents, overriding the 2/3/4 age bands. "
              f"This is the I_bar sensitivity; do NOT compare its level against a "
              f"baseline checkpoint without re-balancing and re-fitting.", flush=True)
        mean_int_ran_mu = torch.full((3,), float(_override))

    child_agents = (agents_ages < ADULT_LOWER_INDEX).view(-1)
    adult_agents = torch.logical_and(
        agents_ages >= ADULT_LOWER_INDEX, agents_ages <= ADULT_UPPER_INDEX
    ).view(-1)
    elderly_agents = (agents_ages > ADULT_UPPER_INDEX).view(-1)

    agents_mean_interactions[child_agents.bool(), 0] = mean_int_ran_mu[0]
    agents_mean_interactions[adult_agents.bool(), 0] = mean_int_ran_mu[1]
    agents_mean_interactions[elderly_agents.bool(), 0] = mean_int_ran_mu[2]

    return agents_mean_interactions


COLUMN_ALIASES = {
    "age": ["age", "age_group", "Age", "AGE", "alder"],
    "sex": ["sex", "Sex", "SEX", "koen", "køn"],
    "municipality": ["municipality", "Municipality", "kommune", "muni_id"],
    "income": ["income_level", "income", "income_decile", "Income"],
    "comorbidity": ["charlson_comorbidity_index", "comorbidity", "charlson", "cci"],
    "socio13": ["socio13", "socio_13", "SOCIO13", "socio"],
}


def load_population_attribute(shape, params):
    """Load a per-agent attribute."""
    file_path = params["file_path"]
    col_name = params.get("attribute", os.path.splitext(os.path.basename(file_path))[0])
    pop_dir = os.path.dirname(file_path)

    candidates = COLUMN_ALIASES.get(col_name, [col_name])

    for ext in (".parquet", ".pq", ".csv"):
        agents_path = os.path.join(pop_dir, f"agents{ext}")
        if os.path.exists(agents_path):
            if ext in (".parquet", ".pq"):
                all_cols, _err = [], None
                for _attempt in range(5):
                    try:
                        import pyarrow.parquet as pq
                        all_cols = list(pq.read_schema(agents_path).names)
                        break
                    except Exception as e:
                        _err = e
                        try:
                            all_cols = list(pd.read_parquet(agents_path).columns)
                            break
                        except Exception as e2:
                            _err = e2
                            time.sleep(0.5 * (2 ** _attempt))
                if not all_cols:
                    raise RuntimeError(
                        f"could not read the schema of {agents_path} after 5 attempts: "
                        f"{type(_err).__name__}: {_err}. The file exists; this is a read "
                        f"failure, not a missing attribute.")
            else:
                all_cols = pd.read_csv(agents_path, nrows=0).columns.tolist()

            matched_col = None
            for c in candidates:
                if c in all_cols:
                    matched_col = c
                    break

            if matched_col is not None:
                if ext in (".parquet", ".pq"):
                    df = pd.read_parquet(agents_path, columns=[matched_col])
                else:
                    df = pd.read_csv(agents_path, usecols=[matched_col])
                vals = df[matched_col]
                if hasattr(vals, 'cat'):
                    vals = vals.cat.codes.values.astype(float)
                    vals[vals < 0] = 0
                else:
                    vals = vals.values.astype(float)
                vals = np.nan_to_num(vals, nan=0.0)
                att_tensor = torch.from_numpy(vals).float()
                return att_tensor.unsqueeze(1)

    if os.path.exists(file_path):
        data = pd.read_pickle(file_path)
        if hasattr(data, 'values'):
            arr = data.values.squeeze()
        else:
            arr = np.asarray(data).squeeze()
        att_tensor = torch.from_numpy(arr).float()
        if att_tensor.dim() == 1:
            att_tensor = att_tensor.unsqueeze(1)
        return att_tensor

    _found = ""
    for ext in (".parquet", ".pq"):
        _p = os.path.join(pop_dir, f"agents{ext}")
        if os.path.exists(_p):
            try:
                import pyarrow.parquet as pq
                _found = f"\ncolumns present: {sorted(pq.read_schema(_p).names)}"
            except Exception:
                pass
            break
    raise FileNotFoundError(
        f"Cannot load attribute '{col_name}': no agents.parquet column "
        f"(tried {candidates}) and no pickle at {file_path}{_found}"
    )


def initialize_id(shape, params):
    """Initialize a unique ID for each agent"""
    return torch.arange(0, shape[0]).reshape(-1, 1).float()


_AGENT_ID_COLUMNS = ("agent_id", "id", "person_id", "PERSON_ID")


CONTAINER_DEGREE_LAYERS = ("colleague", "classmate")

CONTAINER_DEGREE_ALPHA = float(os.environ.get("ABM_CONTAINER_DEGREE_ALPHA", "0.5"))


def container_degree_weights_single_layer(src, dst, n_nodes, alpha,
                                          dtype=torch.float32,
                                          chunk=25_000_000):
    """B_n for ONE layer's edge list. THE single definition of the discount."""
    n_l = int(src.numel())
    w = torch.ones(n_l, dtype=dtype)
    if alpha <= 0 or n_l == 0:
        return w, None
    src = src.long()
    dst = dst.long()
    deg = (torch.bincount(src, minlength=n_nodes)
           + torch.bincount(dst, minlength=n_nodes)).float().clamp(min=1.0)
    for a in range(0, n_l, chunk):
        b = min(a + chunk, n_l)
        w[a:b] = (deg[src[a:b]] * deg[dst[a:b]]).pow(-alpha / 2.0).to(dtype)
    stats = {
        "raw_mean_degree": float(deg.mean()),
        "mean_effective_contacts": float(deg.pow(1.0 - alpha).mean()),
    }
    del deg
    return w, stats


def _container_degree_weights(edge_index, edge_type, n_nodes, alpha,
                              layer_names=None):
    """Per-edge weight B_n discounting clique edges by container degree."""
    E = edge_index.shape[1]
    w = torch.ones(E, dtype=torch.float16)
    if alpha <= 0:
        return w
    names = list(layer_names or LAYER_NAMES)
    for layer in CONTAINER_DEGREE_LAYERS:
        if layer not in names:
            continue
        li = names.index(layer)
        mask = (edge_type.long() == li)
        n_l = int(mask.sum())
        if n_l == 0:
            continue
        idx_l = torch.nonzero(mask, as_tuple=False).reshape(-1)
        wl, stats = container_degree_weights_single_layer(
            edge_index[0, mask], edge_index[1, mask], n_nodes, alpha,
            dtype=torch.float16)
        w[idx_l] = wl
        print(f"  [network] {layer}: container-degree discount alpha={alpha} -> "
              f"mean effective contacts {stats['mean_effective_contacts']:.1f} "
              f"(raw mean degree {stats['raw_mean_degree']:.1f})", flush=True)
        del idx_l, wl
    return w


def _remap_edge_ids_if_needed(edge_array, pop_dir):
    """Remap edge IDs from original register person IDs to 0-indexed agent rows."""
    max_id = int(edge_array.max())
    _agents_path = None
    for _ext in (".parquet", ".pq"):
        _ap = os.path.join(pop_dir, f"agents{_ext}")
        if os.path.exists(_ap):
            _agents_path = _ap
            break
    if _agents_path is None:
        return edge_array
    try:
        import pyarrow.parquet as pq
        _num_agents = pq.read_metadata(_agents_path).num_rows
    except Exception:
        return edge_array
    if max_id < _num_agents:
        return edge_array

    _cache_path = os.path.join(pop_dir, "networks", "_remapped_edges_v2.npy")
    if os.path.exists(_cache_path):
        print(f"  [network] Loading cached remapped edges from {_cache_path}")
        return np.load(_cache_path)

    agent_ids = None
    try:
        import pyarrow.parquet as pq
        _schema = pq.read_schema(_agents_path).names
        _id_col = next((c for c in _AGENT_ID_COLUMNS if c in _schema), None)
        if _id_col is not None:
            agent_ids = pd.read_parquet(_agents_path, columns=[_id_col])[_id_col] \
                .to_numpy().astype(np.int64)
            print(f"  [network] Remapping edge IDs against agents.parquet "
                  f"column '{_id_col}' ({len(agent_ids):,} agents)")
    except Exception as e:
        print(f"  [network] Could not read the agent ID column ({e})")

    if agent_ids is not None:
        order = np.argsort(agent_ids, kind="stable")
        sorted_ids = agent_ids[order]
        pos = np.searchsorted(sorted_ids, edge_array)
        pos_clipped = np.clip(pos, 0, len(sorted_ids) - 1)
        found = sorted_ids[pos_clipped] == edge_array
        n_missing = int(found.size - int(found.sum()))
        if n_missing:
            print(f"  [network] WARNING: {n_missing:,} edge endpoints "
                  f"({100.0 * n_missing / found.size:.4f}%) have no matching agent "
                  f"row and were dropped from the mapping")
        edge_array = order[pos_clipped].astype(np.int64)
        if n_missing:
            edge_array = np.where(found, edge_array, -1)
            keep = (edge_array[:, 0] >= 0) & (edge_array[:, 1] >= 0)
            edge_array = edge_array[keep]
            print(f"  [network] Kept {len(edge_array):,} edges with both endpoints "
                  f"matched")
        print(f"  [network] Mapped register IDs onto agent rows "
              f"[0, {len(agent_ids) - 1:,}]")
    else:
        print("  [network] WARNING: no agent ID column found in agents.parquet — "
              "falling back to edge-list rank mapping.")
        print("  [network] WARNING: any agent absent from every layer shifts all "
              "later agents' attributes onto the wrong node. Stratified results "
              "(income, age, comorbidity) will NOT be interpretable. Add an ID "
              "column to agents.parquet to fix this.")
        unique_ids = np.unique(edge_array)
        edge_array = np.searchsorted(unique_ids, edge_array).astype(np.int64)
        n_mapped = len(unique_ids)
        print(f"  [network] Remapped {n_mapped:,} unique node IDs -> "
              f"[0, {n_mapped - 1:,}]")
        if n_mapped != _num_agents:
            print(f"  [network] WARNING: {n_mapped:,} network nodes != "
                  f"{_num_agents:,} agents — attributes ARE misaligned")

    try:
        np.save(_cache_path, edge_array)
        print(f"  [network] Cached remapped edges to {_cache_path}")
    except Exception as e:
        print(f"  [network] Could not cache: {e}")

    return edge_array


def network_from_file(params):
    """Load network edges. Dispatches to multi-file loader if params has
    'network_files'.
    """
    if "network_files" in params:
        return network_from_layered_files(params)

    file_path = params["file_path"]

    if not os.path.exists(file_path):
        net_dir = os.path.dirname(file_path)
        if os.path.isdir(net_dir):
            network_files = {}
            for layer_name in LAYER_NAMES:
                for ext in (".parquet", ".pq", ".csv"):
                    p = os.path.join(net_dir, f"{layer_name}{ext}")
                    if os.path.exists(p):
                        network_files[layer_name] = p
                        break
            if network_files:
                return network_from_layered_files({"network_files": network_files})

    df = pd.read_csv(file_path, header=None)
    if df.iloc[0].apply(lambda v: isinstance(v, str)).any():
        df = pd.read_csv(file_path, header=0)
    df = df.apply(pd.to_numeric, errors='coerce').dropna().astype('int64')

    edges = df.iloc[:, :2].values
    pop_dir = os.path.dirname(os.path.dirname(file_path))
    edges = _remap_edge_ids_if_needed(edges, pop_dir)

    edge_index = torch.tensor(edges.T, dtype=torch.int32)
    E = edge_index.shape[1]

    if df.shape[1] >= 3:
        edge_type = torch.tensor(df.iloc[:, 2].values, dtype=torch.float16)
    else:
        edge_type = torch.ones(E, dtype=torch.float16)

    edge_weight = torch.ones(E, dtype=torch.float16)
    edge_attr = torch.vstack((edge_type, edge_weight))

    return None, (edge_index, edge_attr)


def network_from_layered_files(params):
    """Load network from separate per-layer CSV files (Cremers et al. 2024 format)."""
    network_files = params["network_files"]
    if not network_files:
        edge_index = torch.zeros((2, 0), dtype=torch.int32)
        edge_attr = torch.zeros((2, 0), dtype=torch.float16)
        return None, (edge_index, edge_attr)

    _net_dir = os.path.dirname(next(iter(network_files.values())))
    _cache_pt = os.path.join(
        _net_dir, f"_cached_network_v3_a{int(round(CONTAINER_DEGREE_ALPHA * 100)):03d}.pt")
    if os.path.exists(_cache_pt):
        print(f"  [network] Loading cached network from {_cache_pt}")
        import torch as _torch
        cached = _torch.load(_cache_pt, weights_only=False)
        ei, ea = cached["edge_index"], cached["edge_attr"]
        if ei.dtype == _torch.int64:
            ei = ei.int()
        if ea.dtype == _torch.float32:
            ea = ea.half()
        return None, (ei, ea)

    all_edges = []
    all_types = []

    for layer_name, file_path in network_files.items():
        if not os.path.exists(file_path):
            continue
        layer_idx = LAYER_NAME_TO_IDX.get(layer_name)
        if layer_idx is None:
            continue

        if file_path.endswith((".parquet", ".pq")):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, header=None)
            if df.iloc[0].apply(lambda v: isinstance(v, str)).any():
                df = pd.read_csv(file_path, header=0)
        df = df.apply(pd.to_numeric, errors='coerce').dropna().astype('int64')

        edges = df.iloc[:, :2].values
        all_edges.append(edges)
        all_types.append(np.full(len(edges), layer_idx, dtype=np.int64))

    if not all_edges:
        edge_index = torch.zeros((2, 0), dtype=torch.int32)
        edge_attr = torch.zeros((2, 0), dtype=torch.float16)
        return None, (edge_index, edge_attr)

    all_edges = np.concatenate(all_edges, axis=0)
    all_types = np.concatenate(all_types, axis=0)

    _net_dir = os.path.dirname(next(iter(network_files.values())))
    _pop_dir = os.path.dirname(_net_dir)
    all_edges = _remap_edge_ids_if_needed(all_edges, _pop_dir)

    edge_index = torch.tensor(all_edges.T, dtype=torch.int32)
    edge_type = torch.tensor(all_types, dtype=torch.int16)

    _n_nodes = int(edge_index.max()) + 1 if edge_index.numel() else 1
    if CONTAINER_DEGREE_ALPHA > 0:
        print(f"  [network] Applying container-degree edge weights "
              f"(ABM_CONTAINER_DEGREE_ALPHA={CONTAINER_DEGREE_ALPHA})", flush=True)
    else:
        print("  [network] ABM_CONTAINER_DEGREE_ALPHA=0 — edges UNWEIGHTED "
              "(B_n=1.0); clique layers will dominate transmission", flush=True)
    edge_weight = _container_degree_weights(
        edge_index, edge_type, _n_nodes, CONTAINER_DEGREE_ALPHA)
    edge_attr = torch.vstack((edge_type.half(), edge_weight))

    try:
        import torch as _torch
        _torch.save({"edge_index": edge_index, "edge_attr": edge_attr}, _cache_pt)
        print(f"  [network] Cached processed network to {_cache_pt} "
              f"({edge_index.shape[1]:,} edges, int32+fp16)")
    except Exception as e:
        print(f"  [network] Could not cache: {e}")

    return None, (edge_index, edge_attr)


def load_network_for_year(params, year):
    """Construct per-layer file paths for a given year and load the network."""
    base_dir = params["network_base_dir"]
    network_files = {}
    for layer_name in LAYER_NAMES:
        path = os.path.join(base_dir, f"{layer_name}_{year}.csv")
        network_files[layer_name] = path
    return network_from_layered_files({"network_files": network_files})


def _check_property_length(state, path, value, substep, trans_func, var_name, t):
    """Refuse to replace an agent property with a tensor of a different length."""
    from agent_torch.core.helpers import get_by_path

    if not torch.is_tensor(value) or not path or path[0] != "agents":
        return
    try:
        current = get_by_path(state, path)
    except Exception:
        return
    if not torch.is_tensor(current) or current.numel() == value.numel():
        return
    raise RuntimeError(
        f"substep {substep}/{trans_func} returned '{var_name}' with "
        f"{value.numel():,} elements (shape {tuple(value.shape)}) but the state holds "
        f"{current.numel():,} (shape {tuple(current.shape)}) at step {t}, path "
        f"{'/'.join(path)}. Writing it would change how many agents the model has, "
        f"and the error would surface later somewhere that cannot name this substep. "
        f"An agent property keeps its agent count for the whole run: the bug is in "
        f"whatever built this return value, most likely an index-set or masked "
        f"selection whose size depends on how many agents were in some state.")


def _step_no_snapshots(self, num_steps=None):
    """In-place step loop that avoids AgentTorch's copy_module and snapshot
    accumulation, which would otherwise clone the full state (including the adjacency
    matrices) at every substep.
    """
    import re as _re
    from agent_torch.core.helpers import set_by_path
    assert self.state is not None
    if not num_steps:
        num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]
    num_substeps = self.config["simulation_metadata"]["num_substeps_per_step"]
    print(f"  [patch] _step_no_snapshots active (fully inlined, no copy_module)", flush=True)
    for time_step in range(num_steps):
        self.state["current_step"] = time_step
        for substep in self.config["substeps"].keys():
            observation_profile, action_profile = {}, {}
            for agent_type in self.config["substeps"][substep]["active_agents"]:
                observation_profile[agent_type] = self.controller.observe(
                    self.state, self.initializer.observation_function, agent_type
                )
                action_profile[agent_type] = self.controller.act(
                    self.state, observation_profile[agent_type],
                    self.initializer.policy_function, agent_type,
                )
            next_substep = (int(substep) + 1) % num_substeps
            self.state["current_substep"] = str(next_substep)
            for trans_func in self.config["substeps"][substep]["transition"].keys():
                updated_vals = self.initializer.transition_function[substep][trans_func](
                    state=self.state, action=action_profile
                )
                for var_name, value in updated_vals.items():
                    source_path = self.config["substeps"][substep]["transition"][
                        trans_func]["input_variables"][var_name]
                    _path = _re.split("/", source_path)
                    _check_property_length(self.state, _path, value, substep,
                                           trans_func, var_name, time_step)
                    set_by_path(self.state, _path, value)
            del observation_profile, action_profile


def patch_runner_no_snapshots(runner):
    """Replace runner.step with snapshot-free, in-place version."""
    import types
    runner.step = types.MethodType(_step_no_snapshots, runner)
    runner.state_trajectory = []


def interv_schedule_path(pop_dir):
    """Where this process writes `intervention_schedule.pt`."""
    _override = os.environ.get("DANISH_ABM_INTERV_DIR")
    if _override:
        os.makedirs(_override, exist_ok=True)
        return os.path.join(_override, "intervention_schedule.pt")
    return os.path.join(pop_dir, "intervention_schedule.pt")
