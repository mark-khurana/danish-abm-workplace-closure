"""Surveillance optimization framework for optimal test allocation across Denmark's 98
municipalities.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


class MunicipalityGraph:
    """Weighted graph of Danish municipality connectivity."""

    def __init__(
        self,
        n_municipalities: int,
        adjacency: torch.Tensor,
        population: torch.Tensor,
        municipality_names: Optional[List[int]] = None,
    ):
        """Args: n_municipalities: number of municipalities (M) adjacency: (M, M) float
        tensor — edge weights (0 where no edge) population: (M,) int tensor — agents
        per municipality municipality_names: list of municipality codes
        """
        self.n = n_municipalities
        self.adjacency = adjacency.float()
        self.population = population.float()
        self.municipality_names = municipality_names or list(range(n_municipalities))

        self._degree_centrality: Optional[torch.Tensor] = None
        self._pagerank: Optional[torch.Tensor] = None
        self._betweenness: Optional[torch.Tensor] = None


    @classmethod
    def from_network(
        cls,
        edge_index: torch.Tensor,
        municipality_ids: torch.Tensor,
        municipality_names: Optional[List[int]] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> "MunicipalityGraph":
        """Build municipality adjacency from agent-level network edges."""
        n_muni = int(municipality_ids.max().item()) + 1
        src_muni = municipality_ids[edge_index[0]]
        dst_muni = municipality_ids[edge_index[1]]

        adj = torch.zeros(n_muni, n_muni, dtype=torch.float32)
        cross = src_muni != dst_muni
        if cross.any():
            src_cross = src_muni[cross]
            dst_cross = dst_muni[cross]
            if edge_attr is not None:
                weights = edge_attr[cross].float()
                if weights.dim() > 1:
                    weights = weights[:, 0]
            else:
                weights = torch.ones(src_cross.shape[0], dtype=torch.float32)
            pair_idx = src_cross * n_muni + dst_cross
            adj_flat = torch.zeros(n_muni * n_muni, dtype=torch.float32)
            adj_flat.scatter_add_(0, pair_idx, weights)
            adj = adj_flat.view(n_muni, n_muni)

        adj = (adj + adj.t()) / 2.0

        pop = torch.zeros(n_muni, dtype=torch.float32)
        ones = torch.ones(municipality_ids.shape[0], dtype=torch.float32)
        pop.scatter_add_(0, municipality_ids, ones)

        names = municipality_names or list(range(n_muni))
        return cls(n_muni, adj, pop, names)

    @classmethod
    def from_pickle(
        cls,
        municipality_pickle_path: str,
        network_csv_path: str,
    ) -> "MunicipalityGraph":
        """Build from on-disk data files."""
        import pickle
        import pandas as pd

        with open(municipality_pickle_path, "rb") as f:
            muni_data = pickle.load(f)
        if hasattr(muni_data, "values"):
            muni_arr = muni_data.values.squeeze()
        else:
            muni_arr = np.asarray(muni_data).squeeze()

        unique_codes = sorted(set(int(x) for x in muni_arr))
        code_to_idx = {c: i for i, c in enumerate(unique_codes)}
        n_muni = len(unique_codes)
        muni_ids = torch.tensor([code_to_idx[int(x)] for x in muni_arr], dtype=torch.long)

        df = pd.read_csv(network_csv_path)
        if "source" in df.columns and "target" in df.columns:
            src = torch.tensor(df["source"].values, dtype=torch.long)
            dst = torch.tensor(df["target"].values, dtype=torch.long)
        else:
            src = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
            dst = torch.tensor(df.iloc[:, 1].values, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)

        weight = None
        if "weight" in df.columns:
            weight = torch.tensor(df["weight"].values, dtype=torch.float32)

        return cls.from_network(edge_index, muni_ids, unique_codes, weight)


    def get_adjacency_matrix(self) -> torch.Tensor:
        """Return (M, M) weighted adjacency matrix."""
        return self.adjacency

    def get_normalized_adjacency(self) -> torch.Tensor:
        """Row-normalised adjacency (for diffusion / CAR model)."""
        row_sum = self.adjacency.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return self.adjacency / row_sum

    def get_degree(self) -> torch.Tensor:
        """Weighted degree per node."""
        return self.adjacency.sum(dim=1)

    def get_binary_adjacency(self) -> torch.Tensor:
        """Binary adjacency (1 if connected, 0 otherwise)."""
        return (self.adjacency > 0).float()

    def get_shortest_path_distances(self) -> torch.Tensor:
        """Compute all-pairs shortest path distances (unweighted hops)."""
        binary = self.get_binary_adjacency()
        dist = torch.full((self.n, self.n), float("inf"))
        dist.fill_diagonal_(0.0)

        for src in range(self.n):
            visited = torch.zeros(self.n, dtype=torch.bool)
            visited[src] = True
            frontier = [src]
            d = 0
            while frontier:
                next_frontier = []
                for node in frontier:
                    neighbors = (binary[node] > 0).nonzero(as_tuple=True)[0]
                    for nb in neighbors:
                        nb = nb.item()
                        if not visited[nb]:
                            visited[nb] = True
                            dist[src, nb] = d + 1
                            next_frontier.append(nb)
                frontier = next_frontier
                d += 1
        return dist


    def get_centrality(self, metric: str = "degree") -> torch.Tensor:
        """Compute and cache a centrality metric."""
        if metric == "degree":
            return self._compute_degree_centrality()
        elif metric == "pagerank":
            return self._compute_pagerank()
        elif metric == "betweenness":
            return self._compute_betweenness()
        else:
            raise ValueError(f"Unknown centrality metric: {metric}")

    def _compute_degree_centrality(self) -> torch.Tensor:
        if self._degree_centrality is None:
            deg = self.get_degree()
            self._degree_centrality = deg / deg.sum().clamp(min=1e-12)
        return self._degree_centrality

    def _compute_pagerank(self, alpha: float = 0.85, max_iter: int = 100,
                          tol: float = 1e-6) -> torch.Tensor:
        """Power-iteration PageRank on the municipality graph."""
        if self._pagerank is not None:
            return self._pagerank

        W = self.get_normalized_adjacency()
        pr = torch.ones(self.n) / self.n
        teleport = torch.ones(self.n) / self.n

        for _ in range(max_iter):
            pr_new = alpha * (W.t() @ pr) + (1 - alpha) * teleport
            pr_new = pr_new / pr_new.sum()
            if (pr_new - pr).abs().max() < tol:
                break
            pr = pr_new

        self._pagerank = pr_new
        return self._pagerank

    def _compute_betweenness(self) -> torch.Tensor:
        """Approximate betweenness centrality using shortest paths."""
        if self._betweenness is not None:
            return self._betweenness

        binary = self.get_binary_adjacency()
        bc = torch.zeros(self.n)

        for s in range(self.n):
            stack = []
            pred = [[] for _ in range(self.n)]
            sigma = torch.zeros(self.n)
            sigma[s] = 1.0
            d = torch.full((self.n,), -1.0)
            d[s] = 0.0
            queue = [s]
            qi = 0

            while qi < len(queue):
                v = queue[qi]
                qi += 1
                stack.append(v)
                neighbors = (binary[v] > 0).nonzero(as_tuple=True)[0].tolist()
                for w in neighbors:
                    if d[w] < 0:
                        d[w] = d[v] + 1
                        queue.append(w)
                    if d[w] == d[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta = torch.zeros(self.n)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w].clamp(min=1)) * (1.0 + delta[w])
                if w != s:
                    bc[w] += delta[w]

        norm = max((self.n - 1) * (self.n - 2), 1)
        bc = bc / norm
        self._betweenness = bc
        return self._betweenness


@dataclass
class SurveillanceState:
    """What the surveillance system knows at a given timestep."""

    n_municipalities: int
    prevalence_threshold: float = 0.01

    total_tests: Optional[torch.Tensor] = None
    total_positives: Optional[torch.Tensor] = None
    recent_tests: Optional[torch.Tensor] = None
    recent_positives: Optional[torch.Tensor] = None

    belief: Optional[torch.Tensor] = None

    estimated_prevalence: Optional[torch.Tensor] = None
    prevalence_uncertainty: Optional[torch.Tensor] = None

    observation_history: List[Dict[int, Tuple[int, int]]] = field(default_factory=list)

    t: int = 0

    def __post_init__(self):
        M = self.n_municipalities
        if self.total_tests is None:
            self.total_tests = torch.zeros(M)
        if self.total_positives is None:
            self.total_positives = torch.zeros(M)
        if self.recent_tests is None:
            self.recent_tests = torch.zeros(M)
        if self.recent_positives is None:
            self.recent_positives = torch.zeros(M)
        if self.belief is None:
            self.belief = torch.full((M,), 0.5)
        if self.estimated_prevalence is None:
            self.estimated_prevalence = torch.zeros(M)
        if self.prevalence_uncertainty is None:
            self.prevalence_uncertainty = torch.ones(M) * 0.25

    def update(self, observations: Dict[int, Tuple[int, int]], window: int = 7):
        """Update state with new test results."""
        self.observation_history.append(observations)
        self.t += 1

        self.recent_tests.zero_()
        self.recent_positives.zero_()

        start = max(0, len(self.observation_history) - window)
        for obs in self.observation_history[start:]:
            for muni_id, (tests, positives) in obs.items():
                self.recent_tests[muni_id] += tests
                self.recent_positives[muni_id] += positives

        for muni_id, (tests, positives) in observations.items():
            self.total_tests[muni_id] += tests
            self.total_positives[muni_id] += positives

    def update_beliefs(self, prevalence: torch.Tensor, uncertainty: torch.Tensor):
        """Update belief state from surrogate model predictions."""
        self.estimated_prevalence = prevalence
        self.prevalence_uncertainty = uncertainty

        std = uncertainty.sqrt().clamp(min=1e-6)
        z = (prevalence - self.prevalence_threshold) / std
        self.belief = torch.sigmoid(z * 2.0)

    def get_positivity_rate(self) -> torch.Tensor:
        """Recent test positivity rate per municipality."""
        return self.recent_positives / self.recent_tests.clamp(min=1)

    def get_tested_fraction(self, population: torch.Tensor) -> torch.Tensor:
        """Fraction of population recently tested per municipality."""
        return self.recent_tests / population.clamp(min=1)


class SurrogateModel:
    """Conditional Autoregressive (CAR) model for spatial prevalence estimation."""

    def __init__(
        self,
        graph: MunicipalityGraph,
        rho: float = 0.8,
        obs_precision: float = 10.0,
        spatial_precision: float = 5.0,
        prior_prevalence: float = 0.005,
    ):
        """Args: graph: municipality connectivity graph rho: spatial autoregression
        parameter (0 = no spatial, 1 = full) obs_precision: weight of direct test
        observations spatial_precision: weight of spatial prior prior_prevalence:
        prior mean prevalence before any data
        """
        self.graph = graph
        self.M = graph.n
        self.rho = rho
        self.obs_precision = obs_precision
        self.spatial_precision = spatial_precision

        self.W = graph.get_normalized_adjacency()

        self.mean = torch.full((self.M,), prior_prevalence)
        self.variance = torch.ones(self.M) * 0.25

        self._obs_sum = torch.zeros(self.M)
        self._obs_count = torch.zeros(self.M)

    def update(self, observations: Dict[int, Tuple[int, int]]):
        """Update posterior with new test results."""
        for muni_id, (n_tests, n_positives) in observations.items():
            if n_tests > 0:
                self._obs_sum[muni_id] += n_positives
                self._obs_count[muni_id] += n_tests

        self._update_posterior()

    def _update_posterior(self):
        """Bayesian posterior combining observations with CAR spatial prior."""
        obs_prev = (self._obs_sum + 0.5) / (self._obs_count + 1.0)
        obs_prec = self._obs_count * self.obs_precision

        spatial_mean = self.rho * (self.W @ self.mean)
        spatial_prec = torch.full((self.M,), self.spatial_precision)

        post_prec = obs_prec + spatial_prec
        post_mean = (obs_prec * obs_prev + spatial_prec * spatial_mean) / post_prec.clamp(min=1e-12)

        self.mean = post_mean.clamp(0.0, 1.0)
        self.variance = (1.0 / post_prec.clamp(min=1e-12)).clamp(0.0, 0.25)

    def predict(self) -> torch.Tensor:
        """Return (M,) predicted prevalence per municipality."""
        return self.mean

    def uncertainty(self) -> torch.Tensor:
        """Return (M,) posterior variance per municipality."""
        return self.variance

    def reset(self, prior_prevalence: float = 0.005):
        """Reset to prior state."""
        self.mean = torch.full((self.M,), prior_prevalence)
        self.variance = torch.ones(self.M) * 0.25
        self._obs_sum.zero_()
        self._obs_count.zero_()


class AllocationPolicy(ABC):
    """Base class for surveillance test allocation policies."""

    name: str = "base"

    @abstractmethod
    def allocate(
        self,
        state: SurveillanceState,
        total_tests: int,
        graph: MunicipalityGraph,
    ) -> Dict[int, int]:
        """Allocate tests across municipalities."""
        ...

    def _distribute_proportional(
        self,
        scores: torch.Tensor,
        total_tests: int,
        population: torch.Tensor,
    ) -> Dict[int, int]:
        """Distribute tests proportional to scores, respecting population caps."""
        M = scores.shape[0]
        scores = scores.clamp(min=0)
        total_score = scores.sum()

        if total_score < 1e-12:
            per_muni = total_tests // M
            remainder = total_tests - per_muni * M
            alloc = {}
            for i in range(M):
                alloc[i] = per_muni + (1 if i < remainder else 0)
            return alloc

        raw = (scores / total_score) * total_tests
        raw = torch.min(raw, population)

        floored = raw.floor().long()
        remainders = raw - floored.float()
        allocated = int(floored.sum().item())
        shortfall = total_tests - allocated

        if shortfall > 0:
            _, top_idx = remainders.topk(min(shortfall, M))
            for idx in top_idx:
                floored[idx.item()] += 1

        return {i: int(floored[i].item()) for i in range(M) if floored[i] > 0}


class RandomPolicy(AllocationPolicy):
    """Allocate tests uniformly proportional to population."""

    name = "random"

    def allocate(self, state, total_tests, graph):
        return self._distribute_proportional(graph.population, total_tests, graph.population)


class NodeEntropyPolicy(AllocationPolicy):
    """Allocate to municipalities with highest belief entropy."""

    name = "node_entropy"

    def allocate(self, state, total_tests, graph):
        p = state.belief.clamp(1e-6, 1 - 1e-6)
        entropy = -p * p.log() - (1 - p) * (1 - p).log()
        return self._distribute_proportional(entropy, total_tests, graph.population)


class BALDPolicy(AllocationPolicy):
    """Maximize mutual information between prediction and model parameters."""

    name = "bald"

    def allocate(self, state, total_tests, graph):
        p = state.belief.clamp(1e-6, 1 - 1e-6)
        total_entropy = -p * p.log() - (1 - p) * (1 - p).log()

        uncertainty = state.prevalence_uncertainty.clamp(min=1e-8)
        unc_scaled = uncertainty / uncertainty.max().clamp(min=1e-8)
        bald_score = total_entropy * (0.5 + 0.5 * unc_scaled)

        return self._distribute_proportional(bald_score, total_tests, graph.population)


class LocalEntropyPolicy(AllocationPolicy):
    """Distance-weighted neighborhood entropy on municipality graph."""

    name = "local_entropy"

    def __init__(self, distance_decay: float = 1.0):
        """Args: distance_decay: exponent for distance weighting (higher = more local)"""
        self.distance_decay = distance_decay
        self._distance_weights: Optional[torch.Tensor] = None

    def allocate(self, state, total_tests, graph):
        if self._distance_weights is None or self._distance_weights.shape[0] != graph.n:
            dist = graph.get_shortest_path_distances()
            weights = 1.0 / (dist ** self.distance_decay).clamp(min=1e-6)
            weights.fill_diagonal_(0.0)
            weights[dist == float("inf")] = 0.0
            self._distance_weights = weights

        p = state.belief.clamp(1e-6, 1 - 1e-6)
        node_entropy = -p * p.log() - (1 - p) * (1 - p).log()

        local_entropy = self._distance_weights @ node_entropy

        return self._distribute_proportional(local_entropy, total_tests, graph.population)


class DegreeCentralityPolicy(AllocationPolicy):
    """Allocate proportional to degree centrality in commuting graph."""

    name = "degree_centrality"

    def allocate(self, state, total_tests, graph):
        centrality = graph.get_centrality("degree")
        return self._distribute_proportional(centrality, total_tests, graph.population)


class PageRankPolicy(AllocationPolicy):
    """Allocate proportional to PageRank score."""

    name = "pagerank"

    def allocate(self, state, total_tests, graph):
        pr = graph.get_centrality("pagerank")
        return self._distribute_proportional(pr, total_tests, graph.population)


class ReactiveInfectedPolicy(AllocationPolicy):
    """Allocate tests to graph neighbors of recently detected municipalities."""

    name = "reactive_infected"

    def __init__(self, base_fraction: float = 0.3):
        """Args: base_fraction: fraction of tests distributed uniformly as baseline;
        remainder goes to reactive surge testing.
        """
        self.base_fraction = base_fraction

    def allocate(self, state, total_tests, graph):
        positivity = state.get_positivity_rate()
        active = (positivity > state.prevalence_threshold).float()

        neighbor_score = graph.adjacency @ active

        base_score = graph.population * self.base_fraction
        reactive_score = neighbor_score * (1 - self.base_fraction)

        combined = base_score + reactive_score
        return self._distribute_proportional(combined, total_tests, graph.population)


class ProportionalToIncidencePolicy(AllocationPolicy):
    """Allocate proportional to recent reported case rate."""

    name = "proportional_incidence"

    def __init__(self, floor_fraction: float = 0.1):
        """Args: floor_fraction: minimum fraction of population-proportional allocation
        (prevents zero testing in quiet municipalities).
        """
        self.floor_fraction = floor_fraction

    def allocate(self, state, total_tests, graph):
        incidence = state.recent_positives / graph.population.clamp(min=1) * 100_000

        floor = graph.population * self.floor_fraction
        score = incidence * graph.population + floor

        return self._distribute_proportional(score, total_tests, graph.population)


SURVEILLANCE_POLICIES: Dict[str, type] = {
    "random": RandomPolicy,
    "node_entropy": NodeEntropyPolicy,
    "bald": BALDPolicy,
    "local_entropy": LocalEntropyPolicy,
    "degree_centrality": DegreeCentralityPolicy,
    "pagerank": PageRankPolicy,
    "reactive_infected": ReactiveInfectedPolicy,
    "proportional_incidence": ProportionalToIncidencePolicy,
}


def get_surveillance_policy(name: str, **kwargs) -> AllocationPolicy:
    """Get a surveillance policy by name."""
    if name not in SURVEILLANCE_POLICIES:
        raise KeyError(
            f"Unknown surveillance policy '{name}'. "
            f"Available: {', '.join(sorted(SURVEILLANCE_POLICIES.keys()))}"
        )
    return SURVEILLANCE_POLICIES[name](**kwargs)
