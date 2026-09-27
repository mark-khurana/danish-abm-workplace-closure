"""Composable Non-Pharmaceutical Intervention (NPI) system for the Danish ABM."""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .substeps.utils import LAYER_NAMES, LAYER_NAME_TO_IDX, NUM_STRUCTURED_LAYERS


NUM_SCHEDULE_COLS = NUM_STRUCTURED_LAYERS + 2
COMMUNITY_COL = NUM_STRUCTURED_LAYERS
SUSCEPTIBILITY_COL = NUM_STRUCTURED_LAYERS + 1


SPILLOVER_PRIORS = {
    ("school_closure", "family"): (1.05, 0.05),
    ("school_closure", "household"): (1.15, 0.10),
    ("school_closure", "neighborhood"): (1.10, 0.08),
    ("workplace_closure", "household"): (1.10, 0.08),
    ("workplace_closure", "neighborhood"): (1.05, 0.05),
    ("partial_workplace_closure", "household"): (1.05, 0.06),
    ("partial_workplace_closure", "neighborhood"): (1.03, 0.05),
    ("lockdown_soft", "family"): (1.10, 0.10),
    ("lockdown_soft", "household"): (1.30, 0.15),
    ("lockdown_soft", "neighborhood"): (0.90, 0.10),
    ("lockdown_hard", "family"): (1.15, 0.10),
    ("lockdown_hard", "household"): (1.40, 0.15),
    ("lockdown_hard", "neighborhood"): (0.80, 0.10),
    ("gathering_ban", "household"): (1.02, 0.03),
}


@dataclass
class NPIDefinition:
    """A single NPI with per-layer effect multipliers."""
    name: str
    affected_layers: Dict[str, float] = field(default_factory=dict)
    community_effect: float = 1.0
    susceptibility_effect: float = 1.0
    spillover: Optional[Dict[str, float]] = None
    learnable: bool = False

    def get_multiplier_vector(self) -> torch.Tensor:
        """Return a (NUM_SCHEDULE_COLS,) tensor of multipliers for this NPI."""
        vec = torch.ones(NUM_SCHEDULE_COLS)
        for layer_name, mult in self.affected_layers.items():
            idx = LAYER_NAME_TO_IDX.get(layer_name)
            if idx is not None:
                vec[idx] = mult
        vec[COMMUNITY_COL] = self.community_effect
        vec[SUSCEPTIBILITY_COL] = self.susceptibility_effect
        return vec


BUILTIN_NPIS: Dict[str, NPIDefinition] = {

    "school_closure": NPIDefinition(
        name="school_closure",
        affected_layers={"classmate": 0.0},
        spillover={"household": 1.15},
    ),

    "workplace_closure": NPIDefinition(
        name="workplace_closure",
        affected_layers={"colleague": 0.0},
        spillover={"household": 1.1},
    ),

    "partial_workplace_closure": NPIDefinition(
        name="partial_workplace_closure",
        affected_layers={"colleague": 0.3},
    ),


    "mask_mandate": NPIDefinition(
        name="mask_mandate",
        affected_layers={"classmate": 0.80, "colleague": 0.80, "neighborhood": 0.85},
        community_effect=0.80,
    ),

    "lockdown_soft": NPIDefinition(
        name="lockdown_soft",
        affected_layers={"colleague": 0.3, "classmate": 0.0, "neighborhood": 0.7},
        community_effect=0.5,
        spillover={"household": 1.3},
    ),

    "lockdown_hard": NPIDefinition(
        name="lockdown_hard",
        affected_layers={"colleague": 0.0, "classmate": 0.0, "neighborhood": 0.3},
        community_effect=0.3,
        spillover={"household": 1.4},
    ),

    "social_distancing": NPIDefinition(
        name="social_distancing",
        affected_layers={"neighborhood": 0.7},
        community_effect=0.7,
    ),

    "gathering_ban": NPIDefinition(
        name="gathering_ban",
        affected_layers={"neighborhood": 0.8},
        community_effect=0.6,
    ),

    "travel_restrictions": NPIDefinition(
        name="travel_restrictions",
        affected_layers={},
        community_effect=0.95,
    ),
}


SPILLOVER_NPI_NAMES = list(BUILTIN_NPIS.keys())


def build_spillover_matrix(
    npi_names: Optional[List[str]] = None,
    layer_names: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the spillover prior matrix from SPILLOVER_PRIORS."""
    npi_names = npi_names or SPILLOVER_NPI_NAMES
    layer_names = layer_names or list(LAYER_NAMES)

    n_npis = len(npi_names)
    n_layers = len(layer_names)

    prior_means = torch.ones(n_npis, n_layers)
    prior_stds = torch.ones(n_npis, n_layers) * 0.01

    for (npi, layer), (mean, std) in SPILLOVER_PRIORS.items():
        if npi in npi_names and layer in layer_names:
            ni = npi_names.index(npi)
            li = layer_names.index(layer)
            prior_means[ni, li] = mean
            prior_stds[ni, li] = std

    logits = torch.log(prior_means.clamp(min=1e-6))

    return logits, prior_means, prior_stds


def get_npi(name: str, **overrides) -> NPIDefinition:
    """Get a built-in NPI definition, optionally overriding fields."""
    if name not in BUILTIN_NPIS:
        raise KeyError(f"Unknown NPI '{name}'. Available: {list(BUILTIN_NPIS.keys())}")
    npi = copy.deepcopy(BUILTIN_NPIS[name])
    for k, v in overrides.items():
        if hasattr(npi, k):
            setattr(npi, k, v)
    return npi


class NPISchedule:
    """Compose multiple named NPIs over a time horizon into a multiplier tensor."""

    def __init__(self, n_days: int, layer_names: List[str] = None):
        self.n_days = n_days
        self.layer_names = layer_names or list(LAYER_NAMES)
        self.entries: List[dict] = []

    def add(self, npi_name: str, day_start: int, day_end: Optional[int] = None,
            override_effects: Optional[Dict] = None):
        """Add a named NPI to the schedule."""
        npi = get_npi(npi_name, **(override_effects or {}))
        if day_end is None:
            day_end = self.n_days
        self.entries.append({"npi": npi, "day_start": day_start, "day_end": day_end})

    def add_custom(self, npi: NPIDefinition, day_start: int, day_end: Optional[int] = None):
        """Add a custom NPIDefinition directly."""
        if day_end is None:
            day_end = self.n_days
        self.entries.append({"npi": npi, "day_start": day_start, "day_end": day_end})

    def build_multiplier_tensor(self) -> torch.Tensor:
        """Compose all NPIs into a (n_days, 7) multiplier tensor."""
        tensor = torch.ones(self.n_days, NUM_SCHEDULE_COLS)
        for entry in self.entries:
            npi = entry["npi"]
            s = max(0, entry["day_start"])
            e = min(entry["day_end"], self.n_days)
            if s >= e:
                continue
            vec = npi.get_multiplier_vector()
            tensor[s:e] *= vec.unsqueeze(0)
        return tensor

    def build_npi_active_tensor(self) -> torch.Tensor:
        """Return (n_days, n_npis) binary tensor: which NPIs are active on each day."""
        npi_names = SPILLOVER_NPI_NAMES
        tensor = torch.zeros(self.n_days, len(npi_names))
        for entry in self.entries:
            npi_name = entry["npi"].name
            if npi_name in npi_names:
                idx = npi_names.index(npi_name)
                s = max(0, entry["day_start"])
                e = min(entry["day_end"], self.n_days)
                if s < e:
                    tensor[s:e, idx] = 1.0
        return tensor

    def save(self, path: str):
        """Save schedule as .pt file compatible with transmission.py."""
        tensor = self.build_multiplier_tensor()
        npi_active = self.build_npi_active_tensor()
        torch.save({
            "multipliers": tensor,
            "npi_active": npi_active,
            "n_days": self.n_days,
            "column_names": self.layer_names + ["community", "susceptibility"],
            "entries": [
                {"name": e["npi"].name, "day_start": e["day_start"], "day_end": e["day_end"]}
                for e in self.entries
            ],
        }, path)

    @classmethod
    def load(cls, path: str) -> "NPISchedule":
        """Load a saved schedule. Note: only timing metadata is restored, not full NPI
        defs.
        """
        data = torch.load(path, weights_only=False)
        sched = cls(n_days=data["n_days"])
        sched._loaded_tensor = data["multipliers"]
        if "npi_active" in data:
            sched._loaded_npi_active = data["npi_active"]
        if "entries" in data:
            for e in data["entries"]:
                try:
                    sched.add(e["name"], e["day_start"], e["day_end"])
                except KeyError:
                    pass
        return sched

    def counterfactual(self, remove: Optional[List[str]] = None,
                       add: Optional[List[Tuple[str, int, Optional[int]]]] = None) -> "NPISchedule":
        """Create a modified copy for scenario comparison."""
        new = NPISchedule(self.n_days, self.layer_names)
        remove_set = set(remove or [])
        present = {e["npi"].name for e in self.entries}
        for entry in self.entries:
            if entry["npi"].name not in remove_set:
                new.entries.append(copy.deepcopy(entry))
        if remove_set:
            removed = sorted(remove_set & present)
            noop = sorted(remove_set - present)
            print(f"  [counterfactual] removed {removed or '[]'}; "
                  f"no-op (not in this wave's schedule) {noop or '[]'}", flush=True)
        if add:
            for item in add:
                name, start = item[0], item[1]
                end = item[2] if len(item) > 2 else None
                new.add(name, start, end)
        return new

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [f"NPISchedule: {self.n_days} days, {len(self.entries)} NPIs"]
        for e in self.entries:
            lines.append(f"  [{e['day_start']:>4d}-{e['day_end']:>4d}) {e['npi'].name}")
        return "\n".join(lines)

    def plot(self):
        """Visualize the multiplier tensor."""
        import matplotlib.pyplot as plt
        tensor = self.build_multiplier_tensor()
        col_names = self.layer_names + ["community", "susceptibility"]
        fig, ax = plt.subplots(figsize=(14, 5))
        for i, name in enumerate(col_names):
            ax.plot(tensor[:, i].numpy(), label=name, linewidth=1.5)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Day")
        ax.set_ylabel("Multiplier")
        ax.set_title("NPI Schedule")
        ax.legend()
        ax.set_ylim(-0.05, max(1.5, tensor.max().item() + 0.1))
        plt.tight_layout()
        plt.show()


class LearnableNPISchedule(nn.Module):
    """NPI schedule where effect sizes are learnable nn.Parameters."""

    def __init__(self, n_days: int, entries: Optional[List[dict]] = None):
        super().__init__()
        self.n_days = n_days
        self.entries = entries or []
        self._params = nn.ParameterDict()

        for i, entry in enumerate(self.entries):
            npi = entry["npi"]
            if not npi.learnable:
                continue
            vec = npi.get_multiplier_vector()
            clamped = vec.clamp(1e-4, 1.0 - 1e-4)
            logits = torch.log(clamped / (1.0 - clamped))
            self._params[f"npi_{i}"] = nn.Parameter(logits)

    def forward(self) -> torch.Tensor:
        """Build the multiplier tensor with learned effect sizes."""
        tensor = torch.ones(self.n_days, NUM_SCHEDULE_COLS)
        for i, entry in enumerate(self.entries):
            npi = entry["npi"]
            s = max(0, entry["day_start"])
            e = min(entry["day_end"], self.n_days)
            if s >= e:
                continue
            key = f"npi_{i}"
            if npi.learnable and key in self._params:
                vec = torch.sigmoid(self._params[key])
            else:
                vec = npi.get_multiplier_vector()
            tensor[s:e] = tensor[s:e] * vec.unsqueeze(0)
        return tensor

    @classmethod
    def from_schedule(cls, schedule: NPISchedule, learnable_npis: Optional[List[str]] = None) -> "LearnableNPISchedule":
        """Create from an existing NPISchedule, marking specified NPIs as learnable."""
        entries = copy.deepcopy(schedule.entries)
        learnable_set = set(learnable_npis or [])
        for entry in entries:
            if entry["npi"].name in learnable_set:
                entry["npi"].learnable = True
        return cls(schedule.n_days, entries)


def extract_spillover(cal_params: dict, layer_names: Optional[List[str]] = None) -> dict:
    """Extract calibrated spillover matrix from checkpoint params into a tidy dict."""
    layer_names = layer_names or list(LAYER_NAMES)
    npi_names = SPILLOVER_NPI_NAMES

    if "spillover_values" in cal_params:
        matrix = cal_params["spillover_values"]
    elif "spillover_logits" in cal_params:
        import math
        matrix = [[math.exp(v) for v in row] for row in cal_params["spillover_logits"]]
    else:
        return None

    entries = []
    for ni, npi in enumerate(npi_names):
        for li, layer in enumerate(layer_names):
            val = matrix[ni][li]
            prior = SPILLOVER_PRIORS.get((npi, layer), (1.0, 0.01))
            prior_mean, prior_std = prior
            delta_pct = (val / prior_mean - 1.0) * 100 if prior_mean > 0 else 0.0
            entries.append({
                "npi": npi,
                "layer": layer,
                "calibrated": val,
                "prior_mean": prior_mean,
                "prior_std": prior_std,
                "delta_pct": delta_pct,
                "is_nontrivial": abs(prior_mean - 1.0) > 0.005 or abs(val - 1.0) > 0.005,
            })

    nontrivial = sorted(
        [e for e in entries if e["is_nontrivial"]],
        key=lambda e: abs(e["delta_pct"]),
        reverse=True,
    )

    return {
        "matrix": matrix,
        "npi_names": npi_names,
        "layer_names": layer_names,
        "entries": nontrivial,
    }


def plot_spillover(cal_params: dict, save_path: Optional[str] = None,
                   layer_names: Optional[List[str]] = None):
    """Heatmap of calibrated spillover matrix vs. literature priors."""
    import matplotlib.pyplot as plt
    import numpy as np

    data = extract_spillover(cal_params, layer_names)
    if data is None:
        print("No spillover data found in cal_params")
        return None

    matrix = np.array(data["matrix"])
    npi_names = data["npi_names"]
    layer_names_out = data["layer_names"]
    n_npis, n_layers = matrix.shape

    pretty_npi = {
        "school_closure": "School closure",
        "workplace_closure": "Workplace closure",
        "partial_workplace_closure": "Partial workplace",
        "mask_mandate": "Mask mandate",
        "lockdown_soft": "Soft lockdown",
        "lockdown_hard": "Hard lockdown",
        "social_distancing": "Social distancing",
        "gathering_ban": "Gathering ban",
        "travel_restrictions": "Travel restrictions",
    }
    pretty_layer = {
        "family": "Family",
        "household": "Household",
        "neighborhood": "Neighborhood",
        "colleague": "Colleague",
        "classmate": "Classmate",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    vmin = min(0.7, matrix.min())
    vmax = max(1.5, matrix.max())
    abs_max = max(abs(vmin - 1.0), abs(vmax - 1.0))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=1.0 - abs_max, vmax=1.0 + abs_max,
                   aspect='auto')

    for ni in range(n_npis):
        for li in range(n_layers):
            val = matrix[ni, li]
            prior = SPILLOVER_PRIORS.get((npi_names[ni], layer_names_out[li]), None)
            if prior is not None and abs(prior[0] - 1.0) > 0.005:
                text = f"{val:.2f}\n({prior[0]:.2f})"
                fontweight = 'bold'
            elif abs(val - 1.0) > 0.005:
                text = f"{val:.2f}"
                fontweight = 'bold'
            else:
                text = ""
                fontweight = 'normal'
            color = 'white' if abs(val - 1.0) > 0.15 else 'black'
            ax.text(li, ni, text, ha='center', va='center',
                    fontsize=8, fontweight=fontweight, color=color)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([pretty_layer.get(l, l) for l in layer_names_out], fontsize=10)
    ax.set_yticks(range(n_npis))
    ax.set_yticklabels([pretty_npi.get(n, n) for n in npi_names], fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Spillover multiplier", fontsize=10)

    ax.set_title("NPI spillover: calibrated (prior)", fontsize=12)
    ax.set_xlabel("Receiving layer", fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved spillover figure: {save_path}")

    return fig


def plot_spillover_comparison(wave_params: Dict[str, dict],
                              save_path: Optional[str] = None,
                              layer_names: Optional[List[str]] = None):
    """Compare calibrated spillover across waves — the key stability figure."""
    import matplotlib.pyplot as plt
    import numpy as np

    layer_names = layer_names or list(LAYER_NAMES)
    npi_names = SPILLOVER_NPI_NAMES
    wave_labels = list(wave_params.keys())
    n_waves = len(wave_labels)

    wave_matrices = []
    for label in wave_labels:
        data = extract_spillover(wave_params[label], layer_names)
        if data is None:
            print(f"  WARNING: no spillover data for {label}")
            return None
        wave_matrices.append(np.array(data["matrix"]))

    cells = []
    for ni, npi in enumerate(npi_names):
        for li, layer in enumerate(layer_names):
            prior = SPILLOVER_PRIORS.get((npi, layer), (1.0, 0.01))
            vals = [m[ni, li] for m in wave_matrices]
            if abs(prior[0] - 1.0) > 0.005 or any(abs(v - 1.0) > 0.01 for v in vals):
                cells.append({
                    "npi": npi, "layer": layer,
                    "ni": ni, "li": li,
                    "prior_mean": prior[0], "prior_std": prior[1],
                    "values": vals,
                    "label": f"{npi}\n→ {layer}",
                })

    if not cells:
        print("  No non-trivial spillover cells to compare")
        return None

    n_cells = len(cells)
    x = np.arange(n_cells)
    width = 0.7 / n_waves

    colors = plt.cm.Set2(np.linspace(0, 0.6, n_waves))

    fig, ax = plt.subplots(figsize=(max(10, n_cells * 1.2), 5))

    for w, label in enumerate(wave_labels):
        vals = [c["values"][w] for c in cells]
        offset = (w - n_waves / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=colors[w],
                       edgecolor='white', linewidth=0.5)

    for i, c in enumerate(cells):
        ax.plot([i - 0.4, i + 0.4], [c["prior_mean"]] * 2,
                color='black', linewidth=1.5, linestyle='--', alpha=0.6)
        if i == 0:
            ax.plot([], [], color='black', linewidth=1.5, linestyle='--',
                    alpha=0.6, label='Literature prior')

    ax.axhline(1.0, color='gray', linewidth=0.5, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([c["label"] for c in cells], fontsize=8, ha='center')
    ax.set_ylabel("Spillover multiplier", fontsize=11)
    ax.set_title("NPI spillover: cross-wave comparison", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=min(0.6, min(c["prior_mean"] for c in cells) - 0.1))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved spillover comparison: {save_path}")

    return fig


def spillover_identifiability(npi_schedule: "NPISchedule",
                              save_path: Optional[str] = None) -> dict:
    """Diagnose whether per-NPI spillover parameters are identifiable."""
    import numpy as np

    active = npi_schedule.build_npi_active_tensor().numpy()
    npi_names = SPILLOVER_NPI_NAMES
    n_npis = len(npi_names)

    active_days = {}
    for i, name in enumerate(npi_names):
        active_days[name] = int(active[:, i].sum())

    used_idx = [i for i in range(n_npis) if active_days[npi_names[i]] > 0]
    used_names = [npi_names[i] for i in used_idx]

    if len(used_idx) < 2:
        return {
            "correlation": np.eye(n_npis),
            "vif": {n: 1.0 for n in used_names},
            "identified": used_names,
            "problematic": [],
            "active_days": active_days,
        }

    X = active[:, used_idx]
    identical_pairs = []
    for i in range(len(used_idx)):
        for j in range(i + 1, len(used_idx)):
            if np.array_equal(X[:, i], X[:, j]):
                identical_pairs.append((used_idx[i], used_idx[j]))

    corr_full = np.eye(n_npis)
    corr_used = np.corrcoef(X.T)
    for i in range(len(used_idx)):
        for j in range(len(used_idx)):
            if np.isnan(corr_used[i, j]):
                corr_used[i, j] = 1.0 if np.array_equal(X[:, i], X[:, j]) else 0.0
    for i_out, i_in in enumerate(used_idx):
        for j_out, j_in in enumerate(used_idx):
            corr_full[i_in, j_in] = corr_used[i_out, j_out]

    identical_set = set()
    for a, b in identical_pairs:
        identical_set.add(a)
        identical_set.add(b)

    vif = {}
    for k, idx_k in enumerate(used_idx):
        name_k = npi_names[idx_k]
        if idx_k in identical_set:
            vif[name_k] = float('inf')
            continue
        y = X[:, k]
        others = np.delete(X, k, axis=1)
        if others.shape[1] == 0 or y.var() < 1e-10:
            vif[name_k] = 1.0
            continue
        others_with_const = np.column_stack([others, np.ones(len(y))])
        try:
            beta, _, _, _ = np.linalg.lstsq(others_with_const, y, rcond=None)
            y_hat = others_with_const @ beta
            ss_res = ((y - y_hat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            vif[name_k] = 1.0 / max(1.0 - r2, 1e-10)
        except np.linalg.LinAlgError:
            vif[name_k] = float('inf')

    identified = [n for n in used_names if vif.get(n, 1.0) < 10.0]
    problematic = []
    for i in range(len(used_idx)):
        for j in range(i + 1, len(used_idx)):
            c = abs(corr_used[i, j])
            if c > 0.8:
                problematic.append((
                    npi_names[used_idx[i]],
                    npi_names[used_idx[j]],
                    float(corr_used[i, j]),
                ))

    npis_with_spillover = set()
    for (npi, _), (mean, _) in SPILLOVER_PRIORS.items():
        if abs(mean - 1.0) > 0.005:
            npis_with_spillover.add(npi)
    unidentified = [n for n in used_names if vif.get(n, 1.0) >= 10.0]
    spillover_at_risk = [n for n in unidentified if n in npis_with_spillover]

    result = {
        "correlation": corr_full,
        "vif": vif,
        "identified": identified,
        "problematic": problematic,
        "active_days": active_days,
        "spillover_at_risk": spillover_at_risk,
    }

    print(f"\n  Spillover identifiability diagnostic")
    print(f"  {'NPI':<30s} {'Active days':>11s} {'VIF':>8s}  Status")
    print(f"  {'-'*65}")
    for name in used_names:
        v = vif.get(name, 1.0)
        if v < 10:
            status = "OK"
        elif name not in npis_with_spillover:
            status = "collinear (benign — no spillover prior)"
        else:
            status = "CAUTION (VIF>10, has spillover prior)"
        vif_str = f"{v:>8.1f}" if v < 1e6 else "     inf"
        print(f"  {name:<30s} {active_days[name]:>8d}    {vif_str}  {status}")

    if problematic:
        print(f"\n  Highly correlated NPI pairs (|r| > 0.8):")
        for a, b, c in problematic:
            print(f"    {a} <-> {b}: r = {c:.3f}")
        if spillover_at_risk:
            print(f"\n  WARNING: {len(spillover_at_risk)} NPI(s) with non-trivial spillover "
                  f"priors are non-identifiable: {spillover_at_risk}")
            print(f"  Consider staggering their timing or fixing their spillover to the prior.")
        else:
            print(f"\n  Collinearity detected but only affects NPIs without spillover priors (benign).")
    else:
        print(f"\n  No problematic collinearity detected — all spillover rows identifiable.")

    if save_path:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                        gridspec_kw={"width_ratios": [2, 1]})

        pretty_npi = {
            "school_closure": "School closure",
            "workplace_closure": "Workplace closure",
            "partial_workplace_closure": "Partial workplace",
            "mask_mandate": "Mask mandate",
            "lockdown_soft": "Soft lockdown",
            "lockdown_hard": "Hard lockdown",
            "social_distancing": "Social distancing",
            "gathering_ban": "Gathering ban",
            "travel_restrictions": "Travel restrictions",
        }
        labels = [pretty_npi.get(n, n) for n in used_names]
        im = ax1.imshow(corr_used, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        for i in range(len(used_idx)):
            for j in range(len(used_idx)):
                v = corr_used[i, j]
                color = 'white' if abs(v) > 0.5 else 'black'
                ax1.text(j, i, f"{v:.2f}", ha='center', va='center',
                         fontsize=8, color=color)
        ax1.set_xticks(range(len(used_idx)))
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.set_yticks(range(len(used_idx)))
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_title("NPI temporal correlation", fontsize=11)
        fig.colorbar(im, ax=ax1, shrink=0.7)

        vif_names = list(vif.keys())
        vif_vals = [vif[n] for n in vif_names]
        vif_labels = [pretty_npi.get(n, n) for n in vif_names]
        colors = ['#d32f2f' if v >= 10 else '#1976d2' for v in vif_vals]
        ax2.barh(range(len(vif_names)), vif_vals, color=colors)
        ax2.set_yticks(range(len(vif_names)))
        ax2.set_yticklabels(vif_labels, fontsize=9)
        ax2.axvline(10, color='red', linestyle='--', alpha=0.7, label='VIF=10 threshold')
        ax2.set_xlabel("Variance Inflation Factor", fontsize=10)
        ax2.set_title("Collinearity (VIF)", fontsize=11)
        ax2.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved identifiability figure: {save_path}")

    return result
