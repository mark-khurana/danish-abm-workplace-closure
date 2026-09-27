#!/usr/bin/env python
"""Build a small synthetic population that the model can run on.

Everything here is invented. The distributions are simple and chosen only so that
every part of the model has something to act on (all four workplace groups, all four
income quartiles, children in school classes, a vaccine rollout, a testing record).
They do not reproduce, and are not derived from, any register data. Results from this
population say nothing about Denmark.

Output (in the layout the model reads):

    <out>/
      agents.parquet             one row per person
      networks/<layer>.csv       one edge list per contact layer (source,target)
      events/tests.parquet       agent_id, date, result, test_type
      events/vaccinations.parquet agent_id, date, vaccine_type, dose
      disease_stages.csv         initial state (all susceptible)

Usage:
    python synthetic/make_population.py --out synthetic_population --n 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decomposition.workplace_attachment import band, band_codes, effective_contacts  # noqa: E402

AGE_BANDS = 8                     # 0-9, 10-19, ..., 70+
AGE_WEIGHTS = np.array([10, 11, 13, 12, 13, 13, 12, 16], float)

# A handful of municipality codes; people are spread evenly across them.
MUNICIPALITIES = [101, 147, 151, 153, 157, 159, 161, 163, 165, 167]

# Occupation codes (SOCIO13-style) used by the occupation analysis.
SELF_EMPLOYED = [111, 112, 113, 114]
MANAGER, HIGH, MEDIUM, BASIC = 131, 132, 133, 134
ELEMENTARY = 135
UNEMPLOYED, STUDENT, PENSIONER, OTHER, CHILD = 210, 310, 321, 330, 420


def _edges_within(groups: list[np.ndarray], rng, max_contacts: int | None = None):
    """All pairs inside each group, or a random subset of partners for large groups."""
    src, dst = [], []
    for g in groups:
        if len(g) < 2:
            continue
        if max_contacts is None or len(g) <= max_contacts + 1:
            i, j = np.triu_indices(len(g), k=1)
            src.append(g[i]); dst.append(g[j])
        else:
            for k, a in enumerate(g):
                partners = rng.choice(np.delete(g, k), size=max_contacts, replace=False)
                src.append(np.full(max_contacts, a)); dst.append(partners)
    if not src:
        return np.empty((0, 2), int)
    e = np.stack([np.concatenate(src), np.concatenate(dst)], axis=1)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def _chunk(ids: np.ndarray, sizes) -> list[np.ndarray]:
    out, start = [], 0
    for s in sizes:
        if start >= len(ids):
            break
        out.append(ids[start:start + s]); start += s
    if start < len(ids):
        out.append(ids[start:])
    return out


def build(n: int, seed: int):
    rng = np.random.default_rng(seed)

    # ── people ─────────────────────────────────────────────────────────────
    age = rng.choice(AGE_BANDS, size=n, p=AGE_WEIGHTS / AGE_WEIGHTS.sum())
    sex = rng.integers(0, 2, size=n)
    municipality = rng.choice(MUNICIPALITIES, size=n)
    charlson = np.minimum(rng.poisson(0.08 * (age + 1) ** 1.3), 6)

    # Income quartile: loosely higher in middle age, so income and age are correlated.
    score = rng.normal(0, 1, n) + np.where((age >= 3) & (age <= 5), 0.5, 0.0)
    income = pd.qcut(score, 4, labels=["Q1", "Q2", "Q3", "Q4"])

    # Occupation: working-age people are employed with a probability that rises
    # with income; employees are spread over skill tiers, also by income.
    socio = np.full(n, OTHER)
    socio[age <= 1] = CHILD
    socio[age >= 7] = PENSIONER
    working = (age >= 2) & (age <= 6)
    q = np.asarray(income.codes)
    p_emp = np.array([0.55, 0.75, 0.85, 0.90])[q]
    employed = working & (rng.random(n) < p_emp)
    tiers = np.array([ELEMENTARY, BASIC, MEDIUM, HIGH, MANAGER])
    tier_p = np.array([[.30, .40, .15, .10, .05],
                       [.20, .40, .20, .15, .05],
                       [.10, .30, .25, .25, .10],
                       [.05, .20, .25, .35, .15]])
    for k in range(4):
        idx = np.where(employed & (q == k))[0]
        socio[idx] = rng.choice(tiers, size=len(idx), p=tier_p[k])
    self_emp = employed & (rng.random(n) < 0.08)
    socio[self_emp] = rng.choice(SELF_EMPLOYED, size=self_emp.sum())
    not_emp = working & ~employed
    socio[not_emp] = rng.choice([UNEMPLOYED, STUDENT, OTHER], size=not_emp.sum(),
                                p=[.4, .3, .3])
    socio[(age == 2) & ~employed] = STUDENT

    agents = pd.DataFrame({
        "agent_id": np.arange(n), "age_group": age, "sex": sex,
        "municipality": municipality, "income_level": income,
        "charlson_comorbidity_index": charlson, "socio13": socio,
    })

    # ── contact layers ─────────────────────────────────────────────────────
    ids = rng.permutation(n)
    households = _chunk(ids, rng.choice([1, 2, 3, 4, 5], size=n, p=[.3, .3, .15, .15, .1]))

    # Family living elsewhere: link each household to one other household.
    fam = []
    for h in households:
        other = households[rng.integers(len(households))]
        fam.append(np.stack(np.meshgrid(h[:1], other[:2]), -1).reshape(-1, 2))
    family = np.unique(np.sort(np.concatenate(fam), axis=1), axis=0)
    family = family[family[:, 0] != family[:, 1]]

    # Neighbourhood: small random clusters within a municipality.
    neigh_groups = []
    for m in MUNICIPALITIES:
        members = rng.permutation(np.where(municipality == m)[0])
        neigh_groups += _chunk(members, np.full(len(members), 12))
    neighborhood = _edges_within(neigh_groups, rng, max_contacts=6)

    # School classes: children and teenagers, about 22 per class, by age band.
    classes = []
    for a in (0, 1):
        pupils = rng.permutation(np.where(age == a)[0])
        classes += _chunk(pupils, np.full(len(pupils), 22))
    classmate = _edges_within(classes, rng)

    # Workplaces: a size mix from sole traders to large sites, so colleague degree
    # covers every workplace group (none, small, medium, large).
    workers = rng.permutation(np.where(employed & ~self_emp)[0])
    size_share = {1: .06, 3: .12, 8: .14, 25: .16, 50: .14, 100: .18, 200: .20}
    sizes = np.concatenate([np.full(max(1, round(w * len(workers) / sz)), sz)
                            for sz, w in size_share.items()])
    workplaces = _chunk(workers, rng.permutation(sizes))
    colleague = _edges_within(workplaces, rng, max_contacts=120)

    # Workplace group from colleague degree, with the same rule the analysis uses.
    deg = np.bincount(colleague.ravel(), minlength=n)
    agents["workplace_band"] = band_codes(band(effective_contacts(deg)))

    layers = {
        "family": family,
        "household": _edges_within(households, rng),
        "neighborhood": neighborhood,
        "colleague": colleague,
        "classmate": classmate,
    }

    # ── events ─────────────────────────────────────────────────────────────
    start = pd.Timestamp("2020-12-16"); days = 75
    n_tests = int(0.4 * n)
    tests = pd.DataFrame({
        "agent_id": rng.integers(0, n, n_tests),
        "date": start + pd.to_timedelta(rng.integers(0, days, n_tests), unit="D"),
        "result": (rng.random(n_tests) < 0.02).astype(int),
        "test_type": rng.choice(["pcr", "antigen"], n_tests, p=[.8, .2]),
    })
    # Vaccination: oldest first, starting two weeks into the window.
    order = np.argsort(-(age + rng.random(n)))
    n_vax = int(0.06 * n)
    vax_days = np.sort(rng.integers(12, days, n_vax))
    vaccinations = pd.DataFrame({
        "agent_id": order[:n_vax],
        "date": start + pd.to_timedelta(vax_days, unit="D"),
        "vaccine_type": "mRNA", "dose": 1,
    })
    return agents, layers, tests, vaccinations


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="synthetic_population")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "networks").mkdir(parents=True, exist_ok=True)
    (out / "events").mkdir(exist_ok=True)
    agents, layers, tests, vaccinations = build(a.n, a.seed)
    agents.to_parquet(out / "agents.parquet", index=False)
    for name, e in layers.items():
        pd.DataFrame(e, columns=["source", "target"]).to_csv(
            out / "networks" / f"{name}.csv", index=False)
    tests.to_parquet(out / "events" / "tests.parquet", index=False)
    vaccinations.to_parquet(out / "events" / "vaccinations.parquet", index=False)
    # Initial state: 0 susceptible, 1 exposed, 2 infectious. A small random seed so
    # the epidemic has somewhere to start.
    stages = np.zeros(a.n, dtype=int)
    seed = np.random.default_rng(a.seed + 1).choice(a.n, size=max(2, a.n // 100), replace=False)
    stages[seed[: len(seed) // 2]] = 1
    stages[seed[len(seed) // 2:]] = 2
    np.savetxt(out / "disease_stages.csv", stages, fmt="%d")

    print(f"{a.n} synthetic people written to {out}/")
    print("  workplace group sizes (0 none, 1 small, 2 medium, 3 large):",
          agents["workplace_band"].value_counts().sort_index().to_dict())
    for name, e in layers.items():
        print(f"  {name:13s} {len(e):>9d} edges")


if __name__ == "__main__":
    main()
