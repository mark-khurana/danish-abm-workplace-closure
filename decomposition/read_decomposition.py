"""The paper's results table: paired intervals, quartile contrasts, RII/SII."""
from __future__ import annotations

import argparse
import json
import os
import sys

import re

import numpy as np
import pandas as pd

PCT = (2.5, 97.5)


def _iv(x):
    """Median and 95% interval across draws, as a plain dict."""
    x = np.asarray([v for v in np.ravel(x) if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    return {"median": float(np.median(x)),
            "lo": float(np.percentile(x, PCT[0])),
            "hi": float(np.percentile(x, PCT[1])),
            "n": int(x.size)}


def _fmt(d, scale=1.0, digits=4):
    if not np.isfinite(d["median"]):
        return "n/a"
    return (f"{d['median'] * scale:,.{digits}g} "
            f"[{d['lo'] * scale:,.{digits}g}, {d['hi'] * scale:,.{digits}g}]")


def _arm_col(df, letter):
    labs = sorted({a for a in df["arm"].dropna().unique()
                   if str(a).upper().startswith(letter)})
    if len(labs) != 1:
        raise SystemExit(f"expected exactly one arm starting with {letter!r}, got {labs}")
    return labs[0]


def select_policy(df, policy, where=""):
    """Narrow a stratified_arms frame to ONE policy, or explain why it cannot."""
    if "policy" not in df.columns:
        return df
    pols = sorted({str(p) for p in df["policy"].dropna().unique()})
    if policy is None:
        if len(pols) <= 1:
            return df
        raise SystemExit(
            f"{where or 'this frame'} holds {len(pols)} policies ({', '.join(pols)}) "
            f"and every quantity below is per-policy. Pass --policy <name>; run the "
            f"reader once per policy to get one readout each.")
    if policy not in pols:
        raise SystemExit(
            f"--policy {policy!r} is not in {where or 'this frame'}. Present: "
            f"{', '.join(pols)}. A policy absent from the wave's schedule is dropped "
            f"by the decomposition's A==B gate, so check that log before assuming a "
            f"typo.")
    return df[df["policy"].astype(str) == policy]


def policy_suffix(policy):
    """Filename tag so per-policy readouts in one directory cannot overwrite each other."""
    return f"__{policy}" if policy else ""


def materialise_arm_c(df, policy=None):
    """Give a reduction-only policy the arm C it has by construction."""
    labs = {str(a).upper()[0] for a in df["arm"].dropna().unique()}
    if "C" in labs:
        return df, True
    if "A" not in labs:
        raise SystemExit(
            f"{policy or 'this frame'} has neither arm A nor arm C; there is nothing "
            f"to read. Check the decomposition log for a failed arm.")
    a_lab = _arm_col(df, "A")
    faux = df[df["arm"] == a_lab].copy()
    faux["arm"] = "C_reduction_only"
    print(f"  [note] {policy or 'this policy'} has no simulated arm C. It has no "
          f"SPILLOVER_PRIORS row, so arm C IS arm A by construction; aliasing it so "
          f"the benefit and protection gradients can be read.\n"
          f"         Its cost of displacement is structurally zero -- no mechanism -- "
          f"and must not be reported as a measured null.")
    return pd.concat([df, faux], ignore_index=True), False


def paired_by_run(df, metric):
    """{stratum: {run: (A, B, C)}} for one stratum_kind, aligned on the draw index."""
    a, b, c = (_arm_col(df, x) for x in "ABC")
    if "policy" in df.columns:
        pols = sorted({str(p) for p in df["policy"].dropna().unique()})
        if len(pols) > 1:
            raise SystemExit(
                f"paired_by_run was given {len(pols)} policies ({', '.join(pols)}). "
                f"The arm pivot would average them into one number. Filter to a single "
                f"policy first (--policies <name>).")
    out = {}
    for s, g in df.groupby("stratum"):
        piv = g.pivot_table(index="run", columns="arm", values=metric, aggfunc="median")
        if not {a, b, c}.issubset(piv.columns):
            continue
        out[str(s)] = piv[[a, b, c]].dropna()
    return out


def ridit_scores(pop):
    """Cumulative-population midpoints, ordered most- to least-deprived."""
    p = np.asarray(pop, dtype=float)
    p = p / p.sum()
    return np.cumsum(p) - p / 2.0


def rii_sii(x, y, w):
    """(RII, SII) from a population-weighted linear fit of y on rank x."""
    x, y, w = map(lambda v: np.asarray(v, dtype=float), (x, y, w))
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    b, a = np.polyfit(x[ok], y[ok], 1, w=np.sqrt(w[ok]))
    at_most, at_least = a + b * 0.0, a + b * 1.0
    sii = at_most - at_least
    rii = at_most / at_least if abs(at_least) > 1e-12 else float("nan")
    return float(rii), float(sii)


JOINT_SEP = "*"

DROP = {
    "age": {"-1", "-1.0"},
    "socio13": {"999", "999.0"},
    "sep": {"", "nan", "None"},
    "workplace_band_wa": {"-1", "-1.0", "-1 (outside working age)"},
}

SOCIO13_SEP = {}

SOCIO13_LABELS = {
    "111": "Self-employed, 10+ employees",
    "112": "Self-employed, 5-9 employees",
    "113": "Self-employed, 1-4 employees",
    "114": "Self-employed, no employees",
    "120": "Assisting spouse",
    "131": "Employees - management",
    "132": "Employees - highest skill level (ISCO 2)",
    "133": "Employees - medium skill level (ISCO 3)",
    "134": "Employees - basic skill level (ISCO 4-8)",
    "135": "Other employees",
    "139": "Employees - unspecified",
    "210": "Unemployed (>half the year)",
    "220": "Sick / parental / education leave",
    "310": "Students (incl. pupils 15+)",
    "321": "Disability pension (foertidspension)",
    "322": "Old-age pension (folkepension)",
    "323": "Early retirement (efterloen)",
    "330": "Social assistance (kontanthjaelp)",
    "410": "Other, minimal labour market attachment",
    "420": "Children under 15",
    "999": "Unknown",
}

SOCIO13_SKILL = {"134": "S1_basic", "133": "S2_medium", "132": "S3_high",
                 "131": "S4_management"}

ORDERED = {
    "income": ["Q1", "Q2", "Q3", "Q4"],
    "sep": ["SEP1", "SEP2", "SEP3", "SEP4"],
    "skill": ["S1_basic", "S2_medium", "S3_high", "S4_management"],
}


def load_socio13_map(path):
    """Replace the hand-assigned SEP tiering with one from a file."""
    with open(path) as fh:
        j = json.load(fh)
    m = j.get("map", j)
    m = {k: v for k, v in m.items() if not str(k).startswith("_")}
    blank = sorted(k for k, v in m.items() if v is None or str(v).strip() == "")
    if blank:
        raise SystemExit(
            f"{path}: {len(blank)} code(s) have no tier assigned yet: "
            f"{', '.join(blank)}.\nAssign every code, or set it to \"DROP\" to "
            f"exclude it deliberately (excluded codes are reported).")
    m = {_strip_num(k): str(v) for k, v in m.items() if str(v).upper() != "DROP"}
    tiers = sorted(set(m.values()))
    SOCIO13_SEP.clear()
    SOCIO13_SEP.update(m)
    ORDERED["sep"] = tiers
    return tiers


def _strip_num(s):
    """'322.0' -> '322'. Joint labels arrive via CSV, so codes come back as floats."""
    s = str(s).strip()
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else s


def derive(df, kind):
    """Collapse a raw stratum_kind onto a derived one ('socio13' -> 'sep'/'skill')."""
    base, mapping = {"sep": ("socio13", SOCIO13_SEP),
                     "skill": ("socio13", SOCIO13_SKILL)}[kind]
    if kind == "sep" and not mapping:
        raise SystemExit(
            "the `sep` occupational ladder is requested but no map is loaded. Pass "
            "--socio13-map socio13_map.json (the pre-registered SEPLINE four-tier "
            "collapse) or socio13_map_skill5.json (the five-tier sensitivity). There "
            "is deliberately no built-in fallback: a differently-tiered gradient "
            "emitted under the name `sep` is indistinguishable in readout.json from "
            "the pre-registered one, and that has already happened once.")
    present = set(df["stratum_kind"].astype(str))
    frames, kinds = [], []
    for src, out_kind in ((base, kind),
                          (f"{base}{JOINT_SEP}age", f"{kind}{JOINT_SEP}age")):
        if src not in present:
            continue
        sub = df[df["stratum_kind"].astype(str) == src].copy()
        if sub.empty:
            continue
        joint = JOINT_SEP in src
        lab = sub["stratum"].astype(str).str.split(JOINT_SEP, n=1, expand=True)
        code = lab[0].map(_strip_num)
        tier = code.map(mapping)
        keep = tier.notna()
        if joint:
            keep &= lab.shape[1] > 1
        if not keep.all():
            lost = (sub[~keep].groupby(sub.loc[~keep, "stratum"]
                                       .astype(str).map(_strip_num))["n_real"]
                    .median().sort_values(ascending=False))
            tot = float(sub.groupby(sub["stratum"].astype(str))["n_real"].median().sum())
            names = ", ".join(f"{c} ({SOCIO13_LABELS.get(c, '?')}, n={n:,.0f})"
                              for c, n in lost.items())
            print(f"    [{out_kind}] excluded {len(lost)} code(s), "
                  f"{lost.sum() / max(tot, 1) * 100:.1f}% of the population: {names}")
        sub, tier = sub[keep].copy(), tier[keep]
        if sub.empty:
            continue
        sub["stratum"] = (tier + JOINT_SEP + lab.loc[tier.index, 1].astype(str)
                          if joint else tier)
        num = [c for c in ("n_real", "n_agents", "infections", "infections_real",
                           "deaths", "deaths_real", "hospitalizations",
                           "hospitalizations_real") if c in sub.columns]
        agg = sub.groupby(["run", "arm", "stratum"], as_index=False)[num].sum()
        if {"infections_real", "n_real"} <= set(agg.columns):
            agg["attack_rate"] = agg["infections_real"] / agg["n_real"].replace(0, np.nan)
        agg["stratum_kind"] = out_kind
        frames.append(agg)
        kinds.append(out_kind)
    if not frames:
        return None, []
    return pd.concat(frames, ignore_index=True), kinds


def standardise(df, kind, metric="infections_real", adjust="age", ref_levels=None,
                ref_weights=None, with_draws=False):
    """Direct standardisation of a gradient over `adjust`, from JOINT cells."""
    keys = [k.strip() for k in str(kind).split(JOINT_SEP)]
    adj = [a.strip() for a in str(adjust).split(JOINT_SEP) if a.strip()]
    idx_keys = [k for k in keys if k not in adj]
    if len(keys) < 2 or len(idx_keys) != 1 or not set(adj) <= set(keys):
        return None
    ipos = keys.index(idx_keys[0])
    lab = df["stratum"].astype(str).str.split(JOINT_SEP, expand=True)
    if lab.shape[1] != len(keys):
        return None
    apos = [i for i in range(len(keys)) if i != ipos]
    a_lab = lab[apos[0]].map(_strip_num)
    for i in apos[1:]:
        a_lab = a_lab + JOINT_SEP + lab[i].map(_strip_num)
    d = df.assign(_i=lab[ipos].astype(str), _a=a_lab)
    for i in apos:
        bad = DROP.get(keys[i], set())
        if bad:
            d = d[~lab[i].map(_strip_num).reindex(d.index).isin(bad)]
    d = d[~d["_i"].isin(DROP.get(keys[ipos], set()))]
    if d.empty:
        return None

    cells = paired_by_run(d.assign(stratum=d["_i"] + JOINT_SEP + d["_a"]), metric)
    if not cells:
        return None
    pop_cell = {c: float(d[d["_i"] + JOINT_SEP + d["_a"] == c]["n_real"].median())
                for c in cells}
    idx = sorted({c.split(JOINT_SEP, 1)[0] for c in cells})
    ages = sorted({c.split(JOINT_SEP, 1)[1] for c in cells})

    if ref_levels is not None:
        _ref = set(ref_levels)
        ages = [a for a in ages
                if a in _ref or _strip_num(str(a).rsplit(JOINT_SEP, 1)[-1]) in _ref]
        if not ages:
            print("    [note] no adjust cell survives the common reference; "
                  "standardisation skipped")
            return None
    common = [a for a in ages
              if all(f"{i}{JOINT_SEP}{a}" in cells
                     and pop_cell.get(f"{i}{JOINT_SEP}{a}", 0.0) > 0 for i in idx)]
    n_dropped = len(ages) - len(common)
    if n_dropped:
        missing = [a for a in ages if a not in common]
        print(f"    [note] {n_dropped} of {len(ages)} adjust cells are not present for "
              f"every index stratum and were excluded from the standardisation: "
              f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
        ages = common
    if len(ages) < 1:
        print("    [note] no adjust cell is present for every index stratum; "
              "the standardisation cannot be computed")
        return None

    if ref_weights is not None and not any(a in ref_weights for a in ages):
        print(f"    [note] the {len(ages)} adjust cells are composite "
              f"({', '.join(map(str, ages[:3]))}...), which the external age reference "
              f"does not key. Standardising over the POOLED population of the common "
              f"cells: comparable across the index strata of this stratifier, NOT "
              f"against a plainly age-standardised gradient on another one.")
        ref_weights = None
    if ref_weights is not None:
        n_age = {a: float(ref_weights.get(a, 0.0)) for a in ages}
        _miss = [a for a in ages if a not in ref_weights]
        if _miss:
            print(f"    [note] {len(_miss)} adjust cells are absent from the common "
                  f"reference and carry zero weight: {', '.join(_miss[:8])}")
    else:
        n_age = {a: sum(pop_cell.get(f"{i}{JOINT_SEP}{a}", 0.0) for i in idx)
                 for a in ages}
    tot = sum(n_age.values())
    if tot <= 0:
        print(f"    [note] every one of the {len(ages)} adjust cells carries zero "
              f"reference weight, so the standardisation has no population to "
              f"recombine over. The reference does not describe these cells.")
        return None
    w = {a: n_age[a] / tot for a in ages}

    runs = sorted(set.intersection(*(set(cells[c].index) for c in cells)))
    per = {i: {"benefit": [], "cost": [], "disp_frac": [], "prot_frac": [],
               "attack_A": [], "attack_B": []}
           for i in idx}
    attack_d = {i: [] for i in idx}
    for r in runs:
        for i in idx:
            b_std = c_std = a_std = n_std = 0.0
            for age in ages:
                key = f"{i}{JOINT_SEP}{age}"
                n = pop_cell[key]
                A, B, C = cells[key].loc[r].values
                b_std += w[age] * (B - C) / n
                c_std += w[age] * (A - C) / n
                a_std += w[age] * (A / n)
                n_std += w[age] * (B / n)
            per[i]["benefit"].append(b_std * 1e5)
            per[i]["cost"].append(c_std * 1e5)
            per[i]["disp_frac"].append(c_std / b_std if b_std else float("nan"))
            per[i]["prot_frac"].append(b_std / n_std if n_std else float("nan"))
            attack_d[i].append(a_std)
            per[i]["attack_A"].append(a_std * 1e5)
            per[i]["attack_B"].append(n_std * 1e5)
    idx = order_strata({i: None for i in idx}, keys[ipos])
    if len(idx) < 2:
        return None
    pop = {i: 1e5 for i in idx}
    weight = {i: sum(pop_cell.get(f"{i}{JOINT_SEP}{a}", 0.0) for a in ages) for i in idx}
    attack = {i: _iv(attack_d[i]) for i in idx}
    out = summarise(per, pop, attack, idx, len(runs), kind=keys[ipos],
                    weight=weight, standardised=True)
    out["adjust_cells_used"] = len(ages)
    out["adjust_cells_dropped"] = n_dropped
    out["adjust_levels"] = sorted(ages)
    out["common_reference"] = ref_levels is not None
    out["common_reference_weights"] = ref_weights is not None
    out["reference_weights"] = {a: float(w[a]) for a in sorted(ages)}
    if with_draws:
        out["per_draw"] = {i: {k: [float(x) for x in v] for k, v in per[i].items()}
                           for i in idx}
    return out


def compare_ratios(a, b, name_a, name_b):
    """Paired difference between two stratifiers' protection ratios, within draw."""
    x = np.asarray(a.get("prot_ratio_draws") or [], dtype=float)
    y = np.asarray(b.get("prot_ratio_draws") or [], dtype=float)
    if x.size == 0 or y.size == 0:
        return None
    n = min(x.size, y.size)
    if x.size != y.size:
        print(f"    [note] {name_a} has {x.size} usable draws and {name_b} has {y.size}; "
              f"the paired difference is not well defined and is withheld")
        return None
    d = x[:n] - y[:n]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    return {"pair": f"{name_a} - {name_b}", "diff": _iv(d),
            "frac_positive": float(np.mean(d > 0)), "n": int(d.size)}


def _label_sort_key(s):
    """Sort key that puts numeric-coded strata in NUMERIC order."""
    t = str(s).strip()
    head = t.split()[0] if t else t
    try:
        return (0, float(head), t)
    except ValueError:
        return (1, 0.0, t)


def order_strata(pairs, kind):
    """Strata in DEPRIVATION order where one is declared, else numeric/label order."""
    want = ORDERED.get(str(kind).split(JOINT_SEP)[0])
    if want is None:
        return sorted(pairs, key=_label_sort_key)
    have = {_strip_num(s): s for s in pairs}
    return [have[k] for k in want if k in have]


def is_ordered(kind):
    """Whether RII/SII mean anything for this stratifier. See ORDERED."""
    return str(kind).split(JOINT_SEP)[0] in ORDERED


def _endpoint_ratio(per, strata, key):
    """Paired most-deprived / least-deprived ratio of `key`, with an interval."""
    v0 = np.array(per[strata[0]].get(key) or [], dtype=float)
    v1 = np.array(per[strata[-1]].get(key) or [], dtype=float)
    if v0.size == 0 or v0.size != v1.size:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.divide(v0, v1, out=np.full(v1.shape, np.nan), where=v1 != 0)
    r = r[np.isfinite(r)]
    den = _iv(v1)
    if not (np.isfinite(den["lo"]) and den["lo"] > 0) or r.size == 0:
        return None
    out = _iv(r)
    out["pair"] = f"{strata[0]} / {strata[-1]}"
    out["frac_above_1"] = float(np.mean(r > 1.0))
    return out


def summarise(per, pop, attack, strata, n_draws, kind=None, weight=None,
              standardised=False):
    """Table, paired contrasts, monotonicity and (where defined) RII/SII."""
    ordered = is_ordered(kind)
    weight = weight or pop
    rii_d, sii_d, mono_d, monoup_d = [], [], [], []
    rii_p, sii_p = [], []
    n = min((len(per[s]["disp_frac"]) for s in strata), default=0)
    for j in range(n):
        vals = [per[s]["disp_frac"][j] for s in strata]
        if not all(np.isfinite(v) for v in vals):
            continue
        mono_d.append(all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)))
        monoup_d.append(all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)))
        if ordered:
            w = [weight[s] for s in strata]
            rii, sii = rii_sii(ridit_scores(w), vals, w)
            rii_d.append(rii)
            sii_d.append(sii)
    for j in range(min((len(per[s]["prot_frac"]) for s in strata), default=0)):
        vals = [per[s]["prot_frac"][j] for s in strata]
        if not all(np.isfinite(v) for v in vals) or not ordered:
            continue
        w = [weight[s] for s in strata]
        rii, sii = rii_sii(ridit_scores(w), vals, w)
        rii_p.append(rii)
        sii_p.append(sii)

    def _rii_on(metric):
        r_, s_ = [], []
        n = min((len(per[st].get(metric) or []) for st in strata), default=0)
        for j in range(n):
            vals = [per[st][metric][j] for st in strata]
            if not all(np.isfinite(v) for v in vals) or not ordered:
                continue
            w = [weight[st] for st in strata]
            a, b = rii_sii(ridit_scores(w), vals, w)
            r_.append(a)
            s_.append(b)
        return r_, s_

    rii_a, sii_a = _rii_on("attack_A")
    rii_b, sii_b = _rii_on("attack_B")

    rows = []
    for s in strata:
        rows.append({
            "stratum": s,
            "n_real": weight[s],
            "attack_A": attack[s],
            "attack_B": _iv(per[s]["attack_B"]) if per[s].get("attack_B") else None,
            "benefit_per_100k": _iv(np.array(per[s]["benefit"]) / pop[s] * 1e5),
            "cost_per_100k": _iv(np.array(per[s]["cost"]) / pop[s] * 1e5),
            "disp_frac": _iv(per[s]["disp_frac"]),
            "prot_frac": _iv(per[s]["prot_frac"]),
        })

    contrasts = []
    for i in range(len(strata) - 1):
        s1, s2 = strata[i], strata[i + 1]
        d = np.array(per[s1]["disp_frac"]) - np.array(per[s2]["disp_frac"])
        d = d[np.isfinite(d)]
        contrasts.append({"pair": f"{s1} - {s2}", "diff": _iv(d),
                          "frac_positive": float(np.mean(d > 0)) if d.size else float("nan")})
    ends = (np.array(per[strata[0]]["disp_frac"])
            - np.array(per[strata[-1]]["disp_frac"]))
    ends = ends[np.isfinite(ends)]

    contrasts_p, mono_pd, mono_pu = [], [], []
    for i in range(len(strata) - 1):
        s1, s2 = strata[i], strata[i + 1]
        d = np.array(per[s1]["prot_frac"]) - np.array(per[s2]["prot_frac"])
        d = d[np.isfinite(d)]
        contrasts_p.append({"pair": f"{s1} - {s2}", "diff": _iv(d),
                            "frac_positive": float(np.mean(d > 0)) if d.size
                            else float("nan")})
    for j in range(min((len(per[s]["prot_frac"]) for s in strata), default=0)):
        vals = [per[s]["prot_frac"][j] for s in strata]
        if not all(np.isfinite(v) for v in vals):
            continue
        mono_pd.append(all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)))
        mono_pu.append(all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)))
    ends_p = (np.array(per[strata[0]]["prot_frac"])
              - np.array(per[strata[-1]]["prot_frac"]))
    ends_p = ends_p[np.isfinite(ends_p)]

    r0 = np.array(per[strata[0]]["prot_frac"], dtype=float)
    r1 = np.array(per[strata[-1]]["prot_frac"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(r1, r0, out=np.full(r0.shape, np.nan), where=r0 != 0)
    ratio = ratio[np.isfinite(ratio)]
    den = _iv(r0)
    well_conditioned = bool(np.isfinite(den["lo"]) and den["lo"] > 0)
    ratio_draws = ratio.tolist() if well_conditioned else []

    return {
        "n_draws": n_draws,
        "prot_ratio_draws": ratio_draws,
        "standardised": bool(standardised),
        "ordered_scale": ordered,
        "strata": rows,
        "adjacent_contrasts": contrasts,
        "endpoint_contrast": {"pair": f"{strata[0]} - {strata[-1]}", "diff": _iv(ends),
                              "frac_positive": float(np.mean(ends > 0)) if ends.size else float("nan")},
        "monotone_decreasing_frac": float(np.mean(mono_d)) if mono_d else float("nan"),
        "monotone_increasing_frac": float(np.mean(monoup_d)) if monoup_d else float("nan"),
        "adjacent_contrasts_prot": contrasts_p,
        "endpoint_contrast_prot": {
            "pair": f"{strata[0]} - {strata[-1]}", "diff": _iv(ends_p),
            "frac_positive": float(np.mean(ends_p > 0)) if ends_p.size
            else float("nan")},
        "monotone_decreasing_prot_frac": float(np.mean(mono_pd)) if mono_pd
        else float("nan"),
        "monotone_increasing_prot_frac": float(np.mean(mono_pu)) if mono_pu
        else float("nan"),
        "RII_disp_frac": _iv(rii_d) if ordered else None,
        "SII_disp_frac": _iv(sii_d) if ordered else None,
        "RII_prot_frac": _iv(rii_p) if ordered else None,
        "SII_prot_frac": _iv(sii_p) if ordered else None,
        "RII_attack_A": _iv(rii_a) if ordered else None,
        "SII_attack_A": _iv(sii_a) if ordered else None,
        "RII_attack_B": _iv(rii_b) if (ordered and rii_b) else None,
        "SII_attack_B": _iv(sii_b) if (ordered and sii_b) else None,
        "attack_A_endpoint_ratio": _endpoint_ratio(per, strata, "attack_A"),
        "prot_ratio_least_over_most": {
            "pair": f"{strata[-1]} / {strata[0]}", "ratio": _iv(ratio),
            "frac_above_1": float(np.mean(ratio > 1)) if ratio.size else float("nan"),
        } if (ordered and well_conditioned) else None,
        "prot_ratio_denominator": den,
    }


def analyse(df, metric="infections_real", attack_col="attack_rate", kind=None):
    """Everything for one marginal stratum_kind."""
    pairs = paired_by_run(df, metric)
    if not pairs:
        return None
    drop = DROP.get(str(kind).split(JOINT_SEP)[0], set())
    pairs = {s: v for s, v in pairs.items() if _strip_num(s) not in drop}
    strata = order_strata(pairs, kind)
    if len(strata) < 2:
        return None
    pop = {s: float(df[df["stratum"].astype(str) == s]["n_real"].median())
           for s in strata}
    attack = {}
    a_lab = _arm_col(df, "A")
    if attack_col not in df.columns:
        print(f"  [note] no {attack_col!r} column; per-stratum attack rates omitted "
              f"(gradients are unaffected -- they are computed from {metric!r})")
        attack = {s: (float("nan"),) * 3 for s in strata}
    else:
        for s in strata:
            g = df[(df["stratum"].astype(str) == s) & (df["arm"] == a_lab)]
            attack[s] = _iv(g[attack_col].values)

    runs = sorted(set.intersection(*(set(pairs[s].index) for s in strata)))
    per = {s: {"benefit": [], "cost": [], "disp_frac": [], "prot_frac": [],
               "attack_A": [], "attack_B": []}
           for s in strata}
    for r in runs:
        for s in strata:
            A, B, C = pairs[s].loc[r].values
            ben, cost = B - C, A - C
            per[s]["benefit"].append(ben)
            per[s]["cost"].append(cost)
            per[s]["attack_A"].append(A / pop[s] * 1e5 if pop[s] else float("nan"))
            per[s]["attack_B"].append(B / pop[s] * 1e5 if pop[s] else float("nan"))
            per[s]["disp_frac"].append(cost / ben if ben != 0 else float("nan"))
            per[s]["prot_frac"].append(ben / B if B != 0 else float("nan"))
    return summarise(per, pop, attack, strata, len(runs), kind=kind)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--strata", default="income,sep,skill,socio13,age",
                    help="'sep' and 'skill' are DERIVED from socio13 here, not "
                         "exported by the model. Each kind is also reported "
                         "age-standardised whenever its '<kind>*age' cells exist.")
    ap.add_argument("--metric", default="infections_real")
    ap.add_argument("--working-age-bands", default="2,3,4,5,6",
                    help="Age bands (decades: 2=20-29 ... 7=70+) kept for the '<kind>_wa' "
                         "restricted gradients, emitted alongside the age-standardised "
                         "ones. Standardisation reweights the pensioners; it does not "
                         "remove them, and the model under-infects 70+ by 3.4-8x because "
                         "the network has no institutional layer. Pass an empty string to "
                         "skip. Band 6 (60-69) straddles retirement and is excluded by "
                         "default.")
    ap.add_argument("--policy", default=None,
                    help="Which policy's arms to read. Required when the wave was "
                         "decomposed with more than one; the readout is then written to "
                         "readout__<policy>.json so the policies do not overwrite "
                         "one another.")
    ap.add_argument("--socio13-map", default=None,
                    help="JSON remapping SOCIO13 codes to tiers, replacing the "
                         "hand-assigned SEP ladder. Written by "
                         "workplace_attachment.py --emit-map. Re-reading with a "
                         "different map costs seconds and re-runs nothing.")
    ap.add_argument("--emit-template", default=None,
                    help="Write a SOCIO13 mapping skeleton -- every code present in "
                         "this directory, with its label and population, and a blank "
                         "tier to fill in from the literature. Authoring it BEFORE "
                         "the outcomes are read is what makes the grouping a priori "
                         "rather than a choice made in sight of the answer.")
    a = ap.parse_args(argv)
    wa_bands = {int(x) for x in str(a.working_age_bands).split(",") if x.strip() != ""}
    if a.emit_template:
        sp = os.path.join(a.outdir, "stratified_arms.csv")
        if not os.path.exists(sp):
            raise SystemExit(f"no stratified_arms.csv in {a.outdir}")
        d = pd.read_csv(sp)
        d = d[d["stratum_kind"].astype(str) == "socio13"]
        if d.empty:
            raise SystemExit(f"no socio13 rows in {sp}")
        pop = (d.groupby(d["stratum"].astype(str).map(_strip_num))["n_real"]
               .median().sort_index())
        tmpl = {
            "_comment": "Assign every code a tier, or \"DROP\" to exclude it. Tier "
                        "labels must SORT most-deprived first (T1, T2, ... or "
                        "SEP1..SEP4) -- the sort order becomes the deprivation rank "
                        "RII is regressed on.",
            "_authored": "fill in: who, when, and against which sources",
            "_sources": [],
            "map": {c: None for c in pop.index},
            "_reference": {c: {"label": SOCIO13_LABELS.get(c, "?"),
                               "n": int(n)} for c, n in pop.items()},
        }
        with open(a.emit_template, "w") as fh:
            json.dump(tmpl, fh, indent=2)
        print(f"  wrote {a.emit_template}  "
              f"({len(pop)} codes, {pop.sum():,.0f} people)")
        print("  Fill `map`, record the citations in `_sources`, commit it, THEN read "
              "the results.")
        return 0
    if a.socio13_map:
        tiers = load_socio13_map(a.socio13_map)
        print(f"  socio13 map: {a.socio13_map} -> tiers {tiers}")

    print("=" * 78)
    print(f"  {a.outdir}")
    print("=" * 78)

    jpath = os.path.join(a.outdir, "decomposition.json")
    summary = {"outdir": a.outdir, "metric": a.metric}
    if os.path.exists(jpath):
        j = json.load(open(jpath))
        summary["n_runs"] = j.get("n_runs")
        summary["reference"] = j.get("reference_unperturbed")
        print(f"  n_runs {j.get('n_runs')}   reference "
              f"{j.get('reference_unperturbed')}")
        for pol, dec in (j.get("decomposition") or {}).items():
            for m, v in sorted(dec.items()):
                if not isinstance(v, dict):
                    continue
                ben, cost = v["benefit_of_closure"], v["cost_of_displacement"]
                flag = "" if abs(v["identity_residual"]) < 1e-6 else "  ** RESIDUAL **"
                if cost is None:
                    print(f"    {pol} {m:24s} benefit {ben:>12,.0f}  cost "
                          f"{'no mechanism':>10s}{flag}")
                    continue
                share = (cost / ben * 100) if ben else float("nan")
                print(f"    {pol} {m:24s} benefit {ben:>12,.0f}  cost {cost:>10,.0f}"
                      f"  = {share:5.2f}%{flag}")
                if ben == 0 and cost == 0:
                    print("      ** ALL ZERO: arms A, B and C are identical. The policy "
                          "switch did nothing — check the schedule is being READ from "
                          "DANISH_ABM_INTERV_DIR. **")
    else:
        print(f"  [warn] no decomposition.json in {a.outdir}")

    spath = os.path.join(a.outdir, "stratified_arms.csv")
    if not os.path.exists(spath):
        raise SystemExit(f"no stratified_arms.csv in {a.outdir}")
    s = select_policy(pd.read_csv(spath), a.policy, spath)
    s, _has_disp = materialise_arm_c(s, a.policy)
    summary["policy"] = a.policy
    summary["displacement_mechanism"] = _has_disp

    kinds = [k.strip() for k in a.strata.split(",") if k.strip()]
    have = set(s["stratum_kind"].astype(str).unique())
    derived = {}
    for k in list(kinds):
        if k in ("sep", "skill"):
            agg, agg_kinds = derive(s, k)
            if agg is None:
                print(f"\n  [skip] {k}: no socio13 rows to collapse")
                continue
            derived[k] = agg
            have.update(agg_kinds)

    def _report(label, res):
        summary[label] = res
        tag = " (age-standardised)" if res.get("standardised") else ""
        print(f"\n  ── {label}{tag}   ({res['n_draws']} paired draws; "
              f"median [95% interval across parameter draws])")
        print(f"    {'stratum':<14}{'n':>11}  {'attack_A':<22}"
              f"{'benefit/100k':<26}{'cost/100k':<24}{'disp_frac':<24}"
              f"{'prot_frac'}")
        for r in res["strata"]:
            print(f"    {str(r['stratum']):<14}{r['n_real']:>11,.0f}  "
                  f"{_fmt(r['attack_A'], digits=3):<22}"
                  f"{_fmt(r['benefit_per_100k'], digits=4):<26}"
                  f"{_fmt(r['cost_per_100k'], digits=4):<24}"
                  f"{_fmt(r['disp_frac'], digits=3):<24}"
                  f"{_fmt(r['prot_frac'], digits=3)}")
        print("\n    adjacent contrasts in disp_frac (paired within draw):")
        for c in res["adjacent_contrasts"]:
            print(f"      {c['pair']:<16} {_fmt(c['diff'], digits=3):<26}"
                  f"positive in {c['frac_positive'] * 100:5.1f}% of draws")
        e = res["endpoint_contrast"]
        print(f"      {e['pair']:<16} {_fmt(e['diff'], digits=3):<26}"
              f"positive in {e['frac_positive'] * 100:5.1f}% of draws")
        print(f"    monotone decreasing in {res['monotone_decreasing_frac'] * 100:5.1f}% "
              f"of draws (increasing in {res['monotone_increasing_frac'] * 100:.1f}%)")
        ep = res.get("endpoint_contrast_prot")
        if ep:
            print(f"    in prot_frac: {ep['pair']:<16} "
                  f"{_fmt(ep['diff'], digits=3):<26}"
                  f"positive in {ep['frac_positive'] * 100:5.1f}% of draws"
                  f"   (monotone "
                  f"{res.get('monotone_decreasing_prot_frac', float('nan')) * 100:.1f}%"
                  f" dec / "
                  f"{res.get('monotone_increasing_prot_frac', float('nan')) * 100:.1f}%"
                  f" inc)")
        pr = res.get("prot_ratio_least_over_most")
        if pr:
            print(f"    protection ratio {pr['pair']:<34}"
                  f"{_fmt(pr['ratio'], digits=3):<26}"
                  f"above 1 in {pr['frac_above_1'] * 100:5.1f}% of draws")
        elif res.get("ordered_scale"):
            _d = res.get("prot_ratio_denominator")
            print(f"    protection ratio withheld: the denominator's interval includes "
                  f"zero{'' if _d is None else ' (' + _fmt(_d, digits=3) + ')'}; "
                  f"read RII_prot below instead")
        if res.get("ordered_scale"):
            print(f"    RII {_fmt(res['RII_disp_frac'], digits=3)}   "
                  f"(>1 = most deprived bear a larger share)")
            print(f"    SII {_fmt(res['SII_disp_frac'], digits=3)}")
            print(f"    RII_prot {_fmt(res['RII_prot_frac'], digits=3)}   "
                  f"(<1 = most deprived RECEIVE LESS protection)")
            print(f"    SII_prot {_fmt(res['SII_prot_frac'], digits=3)}")
        else:
            print("    RII/SII not reported: this stratifier has no deprivation "
                  "ordering (see ORDERED in read_decomposition.py)")

    ref_ages = None
    _cand = []
    for kind in kinds:
        if kind == "age":
            continue
        jk = f"{kind}{JOINT_SEP}age"
        src = derived.get(kind)
        sub = src[src["stratum_kind"] == jk] if src is not None \
            else s[s["stratum_kind"] == jk]
        if sub is None or sub.empty:
            continue
        lv = {_strip_num(str(x).rsplit(JOINT_SEP, 1)[1]) for x in sub["stratum"].unique()
              if JOINT_SEP in str(x)}
        lv -= DROP.get("age", set())
        if lv:
            _cand.append(lv)
    ref_pop = None
    if _cand:
        ref_ages = sorted(set.intersection(*_cand))
        _isub = s[s["stratum_kind"] == f"income{JOINT_SEP}age"]
        if not _isub.empty:
            _p = {}
            for _st, _g in _isub.groupby("stratum"):
                _parts = str(_st).split(JOINT_SEP)
                if len(_parts) != 2:
                    continue
                _a = _strip_num(_parts[1])
                if _a in ref_ages:
                    _p[_a] = _p.get(_a, 0.0) + float(_g["n_real"].median())
            if _p:
                ref_pop = _p
                print(f"  [reference] one age reference for every stratifier: the "
                      f"national distribution over {len(_p)} bands, "
                      f"{sum(_p.values()):,.0f} people.")
        widest = max(len(c) for c in _cand)
        if len(ref_ages) < widest:
            print(f"\n  [reference] age-standardising every stratifier over the "
                  f"{len(ref_ages)} bands common to all of them, not the {widest} the "
                  f"widest one has. Gradients on different stratifiers are only "
                  f"comparable when standardised to the same population.")


    for kind in kinds:
        src = derived.get(kind)
        sub = src[src["stratum_kind"] == kind] if src is not None \
            else s[s["stratum_kind"] == kind]
        if sub is None or sub.empty:
            print(f"\n  [skip] no rows for stratum_kind={kind!r}")
        else:
            res = analyse(sub, metric=a.metric, kind=kind)
            if res is None:
                print(f"\n  [skip] {kind}: arms incomplete or fewer than 2 strata")
            else:
                _report(kind, res)

        if kind == "age":
            continue
        jkind = f"{kind}{JOINT_SEP}age"
        jsub = src[src["stratum_kind"] == jkind] if src is not None \
            else s[s["stratum_kind"] == jkind]
        if jsub is None or jsub.empty:
            if kind != "income*socio13":
                print(f"    [no {jkind} cells: age-standardisation unavailable — "
                      f"re-run the decomposition with --strata ...,{jkind}]")
            continue
        _ax = jkind.split(JOINT_SEP)
        _adjust = JOINT_SEP.join(_ax[1:]) if len(_ax) > 2 else "age"
        _name = f"{_ax[0]}_{'_'.join(_ax[1:])}_std" if len(_ax) > 2 \
            else f"{kind}_age_std"
        std = standardise(jsub, jkind, metric=a.metric, adjust=_adjust,
                          ref_levels=ref_ages, ref_weights=ref_pop)
        if std is None:
            print(f"    [{jkind}: too few complete cells to standardise]")
        else:
            _report(_name, std)

        if wa_bands and len(_ax) == 2:
            _w = jsub.copy()
            _parts = _w["stratum"].astype(str).str.split(re.escape(JOINT_SEP), n=1,
                                                         expand=True)
            if _parts.shape[1] == 2:
                _w["_lead"] = _parts[0]
                _w["_band"] = pd.to_numeric(_parts[1], errors="coerce")
                _w = _w[_w["_band"].isin(wa_bands)]
                if _w.empty:
                    print(f"    [{jkind}: no cell in working-age bands "
                          f"{sorted(wa_bands)}]")
                else:
                    _agg = {a.metric: "sum", "n_real": "sum"}
                    if "attack_rate" in _w.columns:
                        _w["_ar_num"] = _w["attack_rate"] * _w["n_real"]
                        _agg["_ar_num"] = "sum"
                    _g = _w.groupby(["_lead", "arm", "run"], as_index=False).agg(_agg)
                    if "_ar_num" in _g.columns:
                        _g["attack_rate"] = _g["_ar_num"] / _g["n_real"].replace(0, np.nan)
                        _g = _g.drop(columns=["_ar_num"])
                    _g = _g.rename(columns={"_lead": "stratum"})
                    _g["stratum_kind"] = f"{kind}_wa"
                    _wres = analyse(_g, metric=a.metric, kind=kind)
                    if _wres is None:
                        print(f"    [{kind}_wa: too few strata after restriction]")
                    else:
                        _wres["restricted_age_bands"] = sorted(wa_bands)
                        _report(f"{kind}_wa", _wres)

    if wa_bands:
        _j3 = "income" + JOINT_SEP + "workplace_band" + JOINT_SEP + "age"
        _base = src if src is not None else s
        _sub = _base[_base["stratum_kind"] == _j3] if _base is not None else None
        if _sub is None or _sub.empty:
            print(f"    [no {_j3} cells: attachment cannot be restricted to working age. "
                  f"Re-run the decomposition with --strata ...,{_j3}]")
        else:
            _p3 = _sub["stratum"].astype(str).str.split(re.escape(JOINT_SEP), n=2,
                                                        expand=True)
            if _p3.shape[1] != 3:
                print(f"    [{_j3}: stratum labels are not three-part; attachment "
                      f"restriction skipped]")
            else:
                _sub = _sub.copy()
                _sub["_band"] = _p3[1]
                _sub["_age"] = pd.to_numeric(_p3[2], errors="coerce")
                _sub = _sub[_sub["_age"].isin(wa_bands)]
                if _sub.empty:
                    print(f"    [{_j3}: no cell in working-age bands "
                          f"{sorted(wa_bands)}]")
                else:
                    _ag = {a.metric: "sum", "n_real": "sum"}
                    if "attack_rate" in _sub.columns:
                        _sub["_arn"] = _sub["attack_rate"] * _sub["n_real"]
                        _ag["_arn"] = "sum"
                    _g3 = _sub.groupby(["_band", "arm", "run"], as_index=False).agg(_ag)
                    if "_arn" in _g3.columns:
                        _g3["attack_rate"] = _g3["_arn"] / _g3["n_real"].replace(0, np.nan)
                        _g3 = _g3.drop(columns=["_arn"])
                    _g3 = _g3.rename(columns={"_band": "stratum"})
                    _g3["stratum_kind"] = "workplace_band_wa"
                    _w3 = analyse(_g3, metric=a.metric, kind="workplace_band")
                    if _w3 is None:
                        print("    [workplace_band_wa: too few strata after restriction]")
                    else:
                        _w3["restricted_age_bands"] = sorted(wa_bands)
                        _w3["derived_from"] = _j3
                        _report("workplace_band_wa", _w3)

    cmps = []
    for x, y in (("sep_age_std", "income_age_std"), ("sep", "income"),
                 ("skill_age_std", "income_age_std")):
        if x in summary and y in summary:
            c = compare_ratios(summary[x], summary[y], x, y)
            if c:
                cmps.append(c)
    if cmps:
        print("\n  -- protection ratio, one stratifier against another, PAIRED within "
              "draw")
        print("     (computed on their respective populations: the occupational ladder "
              "excludes\n      people with no rung on it, income covers everyone)")
        for c in cmps:
            print(f"    {c['pair']:34s} {_fmt(c['diff'], digits=3)}   "
                  f"above 0 in {c['frac_positive'] * 100:.0f}% of {c['n']} draws")
        summary["protection_ratio_comparisons"] = cmps

    out = os.path.join(a.outdir, f"readout{policy_suffix(a.policy)}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\n  wrote {out}  (the figures are built from this)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
