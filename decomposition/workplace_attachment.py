"""Who is actually attached to a workplace, measured from the network."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

LAYER_NAMES = ["family", "household", "neighborhood", "colleague", "classmate"]

COLLEAGUE = LAYER_NAMES.index("colleague")

BANDS = [(0.0, "none"), (1e-9, "small"), (4.0, "medium"), (8.0, "large")]


def colleague_degree(pop_dir, alpha=None):
    """Per-agent degree in the colleague layer, from the cached remapped edge list."""
    import torch

    net = os.path.join(pop_dir, "networks")
    pats = sorted(glob.glob(os.path.join(net, "_cached_network_v3_a*.pt")))
    if alpha is not None:
        want = os.path.join(net, f"_cached_network_v3_a{int(round(alpha * 100)):03d}.pt")
        if not os.path.exists(want):
            raise SystemExit(
                f"--alpha {alpha} was requested but {os.path.basename(want)} is not in "
                f"{net}. Build that alpha's network first. Falling back to another "
                f"alpha's cache would report its degrees under this alpha's name.")
        pats = [want]
    if not pats:
        raise SystemExit(
            f"no cached network in {net}. Run any decomposition or calibration once to "
            f"build it (network_from_layered_files writes _cached_network_v3_a*.pt), "
            f"then re-run this. Reading the raw per-layer files instead would skip the "
            f"ID remap and silently mis-assign degrees.")
    path = pats[0]
    print(f"  [network] {path}")
    cached = torch.load(path, weights_only=False)
    ei, ea = cached["edge_index"], cached["edge_attr"]
    etype = ea[0].to(torch.int16)
    mask = etype == COLLEAGUE
    n_edges = int(mask.sum())
    n_nodes = int(ei.max()) + 1 if ei.numel() else 0
    src = ei[0, mask].long()
    dst = ei[1, mask].long()
    deg = (torch.bincount(src, minlength=n_nodes)
           + torch.bincount(dst, minlength=n_nodes)).numpy()
    print(f"  [network] {n_edges:,} colleague edges over {n_nodes:,} nodes")
    return deg, n_nodes


def effective_contacts(deg, alpha=0.5):
    """(n-1)^(1-alpha): what the discount actually leaves a person exposed to."""
    return np.power(np.maximum(deg, 0), 1.0 - alpha)


def band(eff):
    """Assign every agent to exactly one band. Gaps are a bug, not a default."""
    eff = np.asarray(eff, dtype=float)
    out = np.full(len(eff), "", dtype=object)
    for lo, name in BANDS:
        out[eff >= lo] = name
    unassigned = int((out == "").sum())
    if unassigned:
        raise AssertionError(
            f"{unassigned} agents fell outside every band -- BANDS must tile the "
            f"line from 0 upward")
    n_none = int((out == "none").sum())
    n_zero = int((eff <= 0).sum())
    if n_none != n_zero:
        raise AssertionError(
            f"'none' holds {n_none:,} agents but only {n_zero:,} have zero effective "
            f"contacts; the unattached band must mean exactly that")
    return out


def band_codes(bands):
    """Bands as small ints, ordered by exposure. Written to agents.parquet instead of
    strings: every non-ID column becomes a registered agent property, and small ints
    keep memory low and sort correctly.
    """
    order = {name: i for i, (_, name) in enumerate(BANDS)}
    return np.array([order[b] for b in bands], dtype=np.int8)


def load_agents(pop_dir):
    for ext in (".parquet", ".pq", ".csv"):
        p = os.path.join(pop_dir, f"agents{ext}")
        if os.path.exists(p):
            df = pd.read_parquet(p) if ext != ".csv" else pd.read_csv(p)
            return df, p
    raise SystemExit(f"no agents.parquet in {pop_dir}")


def _col(df, *names):
    return next((n for n in names if n in df.columns), None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pop_dir")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Container-degree discount, matching ABM_CONTAINER_DEGREE_ALPHA")
    ap.add_argument("--write-band", action="store_true",
                    help="Add a `workplace_band` column to agents.parquet so outcomes "
                         "can be stratified by measured attachment. Writes a .bak first.")
    ap.add_argument("--emit-map", default=None,
                    help="Write a SOCIO13 -> tier JSON ordered by MEASURED attachment, "
                         "for read_decomposition.py --socio13-map.")
    ap.add_argument("--out", default=None,
                    help="Write the income x attachment table as CSV, for figure 3's "
                         "panel A. Copy it to <results_dir>/attachment.csv.")
    a = ap.parse_args(argv)

    deg, n_nodes = colleague_degree(a.pop_dir, a.alpha)
    df, apath = load_agents(a.pop_dir)
    if len(df) != n_nodes:
        print(f"  [warn] agents={len(df):,} but network nodes={n_nodes:,}; "
              f"padding/truncating to agents")
        deg = np.resize(deg, len(df))
    eff = effective_contacts(deg, a.alpha)
    bands = band(eff)

    s_col = _col(df, "socio13", "socio_13", "SOCIO13", "socio")
    i_col = _col(df, "income_level", "income", "income_decile")
    print(f"\n  overall: {(deg == 0).mean() * 100:5.1f}% of agents have NO colleague "
          f"edge; median effective contacts among the attached "
          f"{np.median(eff[deg > 0]):.2f}")

    order = None
    if s_col is None:
        print("  [skip] no socio13 column")
    else:
        g = pd.DataFrame({"socio13": df[s_col].astype(str), "deg": deg, "eff": eff})
        t = g.groupby("socio13").agg(n=("deg", "size"),
                                     pct_attached=("deg", lambda v: float((v > 0).mean() * 100)),
                                     mean_eff=("eff", "mean"))
        t = t.sort_values("mean_eff", ascending=False)
        print(f"\n  ── SOCIO13 by MEASURED workplace attachment "
              f"(alpha={a.alpha}) ──")
        print(f"    {'code':<8}{'n':>12}{'% attached':>13}{'mean eff. contacts':>21}")
        for code, r in t.iterrows():
            print(f"    {code:<8}{r['n']:>12,.0f}{r['pct_attached']:>12.1f}%"
                  f"{r['mean_eff']:>21.2f}")
        order = list(t.index)

        if a.emit_map:
            share = t["n"] / t["n"].sum()
            mid = share.cumsum() - share / 2.0
            tier = {}
            for code, c in mid.items():
                k = min(int(c * 4) + 1, 4)
                tier[str(code)] = f"WA{5 - k}"
            if len(set(tier.values())) < 2:
                print(f"  [warn] measured attachment collapsed to "
                      f"{sorted(set(tier.values()))} — one code dominates the "
                      f"population, so a four-tier split is not meaningful here")
            with open(a.emit_map, "w") as fh:
                json.dump({"_comment": "SOCIO13 -> tier, ordered by measured colleague "
                                       "attachment (descending), split at population "
                                       "quartiles. Exogenous to the arms.",
                           "_alpha": a.alpha, "map": tier}, fh, indent=2)
            print(f"\n  wrote {a.emit_map}")

    if i_col is not None:
        g = pd.DataFrame({"income": df[i_col].astype(str), "deg": deg, "eff": eff})
        t = g.groupby("income").agg(n=("deg", "size"),
                                    pct_attached=("deg", lambda v: float((v > 0).mean() * 100)),
                                    mean_eff=("eff", "mean")).sort_index()
        print(f"\n  ── income by measured workplace attachment ──")
        print(f"    {'q':<8}{'n':>12}{'% attached':>13}{'mean eff. contacts':>21}")
        for q, r in t.iterrows():
            print(f"    {q:<8}{r['n']:>12,.0f}{r['pct_attached']:>12.1f}%"
                  f"{r['mean_eff']:>21.2f}")
        print("\n  If attachment barely varies across quartiles, the benefit gradient "
              "is about workplace SIZE and household structure, not about who works.")
        if a.out:
            t.reset_index().rename(columns={"mean_eff": "mean_eff_contacts"}) \
             .to_csv(a.out, index=False)
            print(f"  wrote {a.out}")

    vc = pd.Series(bands).value_counts()
    print(f"\n  ── workplace_band ──")
    for i, (lo, name) in enumerate(BANDS):
        hi = BANDS[i + 1][0] if i + 1 < len(BANDS) else None
        rng = "= 0" if name == "none" else (f"[{lo:g}, {hi:g})" if hi else f">= {lo:g}")
        print(f"    {i} {name:<8}{vc.get(name, 0):>12,}  "
              f"({vc.get(name, 0) / max(len(bands), 1) * 100:5.1f}%)   eff {rng}")

    if a.write_band:
        _have_band = "workplace_band" in df.columns
        _have_wa = "workplace_band_wa" in df.columns
        if _have_band and _have_wa:
            print("\n  [skip] workplace_band and workplace_band_wa both present; "
                  "not overwriting either")
        else:
            bak = apath + ".bak"
            if not os.path.exists(bak):
                import shutil
                shutil.copy2(apath, bak)
                print(f"\n  backed up {apath} -> {bak}")
            if _have_band:
                print("  [keep] workplace_band already present, left as it is")
            else:
                df["workplace_band"] = band_codes(bands)

            _age = None
            for _c in ("age_group", "age"):
                if _c in df.columns:
                    _age = df[_c].to_numpy()
                    break
            if _age is None:
                raise KeyError(
                    "no age_group/age column in agents.parquet, so the working-age "
                    "restriction cannot be built -- and it is the primary stratifier. "
                    "Refusing to write a band column that silently omits it.")
            if _have_wa:
                print("  [keep] workplace_band_wa already present, left as it is")
            else:
                _wa = (_age >= 2) & (_age <= 6)
                _codes_wa = band_codes(bands).astype("int8")
                _codes_wa[~_wa] = -1
                df["workplace_band_wa"] = _codes_wa
                _n_wa = int(_wa.sum())
                print(f"\n  workplace_band_wa: {_n_wa:,} agents in ages 20-69 "
                      f"({_n_wa / max(len(df), 1) * 100:.1f}%), rest coded -1")
                for i, (_, name) in enumerate(BANDS):
                    _k = int((_codes_wa == i).sum())
                    print(f"    {i} {name:<8}{_k:>12,}  "
                          f"({_k / max(_n_wa, 1) * 100:5.1f}% of working age)")

            df.to_parquet(apath) if apath.endswith((".parquet", ".pq")) \
                else df.to_csv(apath, index=False)
            codes = ", ".join(f"{i}={n}" for i, (_, n) in enumerate(BANDS))
            print(f"  wrote workplace_band + workplace_band_wa into {apath} as int8 "
                  f"({codes}; _wa adds -1 = outside ages 20-69)")
            print("  PRIMARY stratifier:   --strata 'workplace_band_wa'")
            print("  descriptive only:     --strata 'income*workplace_band'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
