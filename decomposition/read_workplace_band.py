"""The income gradient at FIXED workplace exposure, from income*workplace_band cells."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_decomposition as R

BAND_NAME = {"-1": "outside working age",
             "0": "none", "1": "small", "2": "medium", "3": "large"}

BAND_SENTINEL_CODES = {"-1"}

NATIVE_WA_BANDS = (2, 3, 4, 5, 6)


def _drop_sentinel_bands(df, col="stratum", label=""):
    """Remove sentinel bands by CODE, before any ordering or ratio."""
    keep = ~df[col].astype(str).map(R._strip_num).isin(BAND_SENTINEL_CODES)
    n = int((~keep).sum())
    if n:
        print(f"    [dropped] {n} row(s) on sentinel band(s) "
              f"{sorted(BAND_SENTINEL_CODES)}{label}")
    return df[keep.values]


def _band(code):
    return f"{code} ({BAND_NAME.get(R._strip_num(code), '?')})"


def _pair_stats(vals, dens, label):
    """One paired contrast, with the denominator conditioning rule applied."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return None
    out = {"pair": label, "n_draws": int(v.size)}
    dv = R._iv([x for x in dens if np.isfinite(x)]) if dens else None
    if dv and np.isfinite(dv.get("lo", float("nan"))) and dv["lo"] > 0:
        out["ratio"] = R._iv(list(v))
        out["frac_above_1"] = float(np.mean(v > 1))
        out["denominator"] = dv
    else:
        out["ratio_withheld"] = ("the denominator interval reaches zero, so the ratio's "
                                 "interval would be produced by division")
    return out


def _all_pairs_crude(df, strata, metric):
    """EVERY ordered pair on a crude scale, paired within draw: ratio and difference."""
    if len(strata) < 2:
        return None
    pairs = R.paired_by_run(df, metric)
    have = [x for x in strata if x in pairs]
    if len(have) < 2:
        return None
    out = {}
    for i in range(len(have)):
        for j in range(i + 1, len(have)):
            lo_s, hi_s = have[i], have[j]
            runs = sorted(set(pairs[lo_s].index) & set(pairs[hi_s].index))
            rats, diffs, dens = [], [], []
            for r in runs:
                _, Bl, Cl = pairs[lo_s].loc[r].values
                _, Bh, Ch = pairs[hi_s].loc[r].values
                if not Bl or not Bh:
                    continue
                pl, ph = (Bl - Cl) / Bl, (Bh - Ch) / Bh
                if not (np.isfinite(pl) and np.isfinite(ph)):
                    continue
                diffs.append(ph - pl)
                dens.append(pl)
                if pl != 0:
                    rats.append(ph / pl)
            if not diffs:
                continue
            st = _pair_stats(rats or [float("nan")], dens, f"{hi_s} / {lo_s}")
            if st is None:
                st = {"pair": f"{hi_s} / {lo_s}", "n_draws": len(diffs)}
            st["difference"] = R._iv(diffs)
            st["frac_diff_above_0"] = float(np.mean(np.asarray(diffs) > 0))
            out[f"{hi_s} vs {lo_s}"] = st
    return out or None


def _all_pairs_from_draws(per_draw, strata):
    """The same, on values `standardise` has already reweighted."""
    if len(strata) < 2:
        return None
    have = [x for x in strata if per_draw.get(x)]
    if len(have) < 2:
        return None
    out = {}
    for i in range(len(have)):
        for j in range(i + 1, len(have)):
            lo_s, hi_s = have[i], have[j]
            d = per_draw[lo_s].get("prot_frac", [])
            n = per_draw[hi_s].get("prot_frac", [])
            if len(d) != len(n) or not d:
                continue
            rats, diffs, dens = [], [], []
            for a, b in zip(n, d):
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                diffs.append(a - b)
                dens.append(b)
                if b != 0:
                    rats.append(a / b)
            if not diffs:
                continue
            st = _pair_stats(rats or [float("nan")], dens, f"{hi_s} / {lo_s}")
            if st is None:
                st = {"pair": f"{hi_s} / {lo_s}", "n_draws": len(diffs)}
            st["difference"] = R._iv(diffs)
            st["frac_diff_above_0"] = float(np.mean(np.asarray(diffs) > 0))
            out[f"{hi_s} vs {lo_s}"] = st
    return out or None


def _monotone_frac(per_draw_or_pairs, strata, from_draws):
    """How often the WHOLE ordered scale runs one way, counted per draw."""
    if len(strata) < 3:
        return None
    if from_draws:
        series = []
        for st in strata:
            v = (per_draw_or_pairs.get(st) or {}).get("prot_frac")
            if not v:
                return None
            series.append(list(v))
    else:
        pairs = per_draw_or_pairs
        if not all(st in pairs for st in strata):
            return None
        runs = sorted(set.intersection(*(set(pairs[st].index) for st in strata)))
        if not runs:
            return None
        series = []
        for st in strata:
            col = []
            for r in runs:
                _, B, C = pairs[st].loc[r].values
                col.append((B - C) / B if B else float("nan"))
            series.append(col)
    n = min(len(c) for c in series)
    up = dn = 0
    used = 0
    for k in range(n):
        vals = [c[k] for c in series]
        if not all(np.isfinite(v) for v in vals):
            continue
        used += 1
        if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
            up += 1
        if all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
            dn += 1
    if not used:
        return None
    return {"n_draws": used, "frac_increasing": up / used, "frac_decreasing": dn / used,
            "order": list(strata)}


def _endpoint_ratio(df, strata, metric):
    """Paired last/first ratio of prot_frac = (B-C)/B. Same arithmetic as `summarise`,
    on a scale that is ordered by exposure rather than deprivation.
    """
    if len(strata) < 2:
        return None
    pairs = R.paired_by_run(df, metric)
    if not all(s in pairs for s in (strata[0], strata[-1])):
        return None
    runs = sorted(set(pairs[strata[0]].index) & set(pairs[strata[-1]].index))
    vals, dens = [], []
    for r in runs:
        num, den = [], []
        for s in (strata[0], strata[-1]):
            A, B, C = pairs[s].loc[r].values
            (den if s == strata[0] else num).append((B - C) / B if B else float("nan"))
        if all(np.isfinite(v) for v in num + den) and den[0] != 0:
            vals.append(num[0] / den[0])
            dens.append(den[0])
    if not vals:
        return None
    den = R._iv(dens)
    if not (np.isfinite(den["lo"]) and den["lo"] > 0):
        return None
    return {"pair": f"{strata[-1]} / {strata[0]}", "ratio": R._iv(vals),
            "frac_above_1": float(np.mean(np.array(vals) > 1)),
            "denominator": den, "draws": [float(v) for v in vals]}


def _endpoint_ratio_from_draws(per_draw, strata):
    """The same last/first prot_frac ratio as `_endpoint_ratio`, but on values that are
    already standardised, so it cannot be recomputed from the raw cells.
    """
    if len(strata) < 2:
        return None
    lo, hi = per_draw.get(strata[0]), per_draw.get(strata[-1])
    if not lo or not hi:
        return None
    num, den = hi.get("prot_frac", []), lo.get("prot_frac", [])
    if len(num) != len(den) or not num:
        return None
    vals, dens = [], []
    for n, d in zip(num, den):
        if np.isfinite(n) and np.isfinite(d) and d != 0:
            vals.append(n / d)
            dens.append(d)
    if not vals:
        return None
    dv = R._iv(dens)
    if not (np.isfinite(dv["lo"]) and dv["lo"] > 0):
        return None
    return {"pair": f"{strata[-1]} / {strata[0]}", "ratio": R._iv(vals),
            "frac_above_1": float(np.mean(np.array(vals) > 1)),
            "denominator": dv, "draws": [float(v) for v in vals]}


def report(label, res, tag=""):
    """Same layout as read_decomposition._report, minus its hardcoded age tag."""
    print(f"\n  -- {label}{tag}   ({res['n_draws']} paired draws; "
          f"median [95% interval across parameter draws])")
    print(f"    {'stratum':<14}{'n':>11}  {'attack_A':<22}"
          f"{'benefit/100k':<26}{'cost/100k':<24}{'disp_frac':<24}"
          f"{'prot_frac'}")
    for r in res["strata"]:
        print(f"    {str(r['stratum']):<14}{r['n_real']:>11,.0f}  "
              f"{R._fmt(r['attack_A'], digits=3):<22}"
              f"{R._fmt(r['benefit_per_100k'], digits=4):<26}"
              f"{R._fmt(r['cost_per_100k'], digits=4):<24}"
              f"{R._fmt(r['disp_frac'], digits=3):<24}"
              f"{R._fmt(r['prot_frac'], digits=3)}")
    print("\n    adjacent contrasts in disp_frac (paired within draw):")
    for c in res["adjacent_contrasts"]:
        print(f"      {c['pair']:<16} {R._fmt(c['diff'], digits=3):<26}"
              f"positive in {c['frac_positive'] * 100:5.1f}% of draws")
    e = res["endpoint_contrast"]
    print(f"      {e['pair']:<16} {R._fmt(e['diff'], digits=3):<26}"
          f"positive in {e['frac_positive'] * 100:5.1f}% of draws")
    print(f"    monotone decreasing in {res['monotone_decreasing_frac'] * 100:5.1f}% "
          f"of draws (increasing in {res['monotone_increasing_frac'] * 100:.1f}%)")
    pr = res.get("prot_ratio_least_over_most")
    if pr:
        print(f"    protection ratio {pr['pair']:<34}"
              f"{R._fmt(pr['ratio'], digits=3):<26}"
              f"above 1 in {pr['frac_above_1'] * 100:5.1f}% of draws")
    elif res.get("ordered_scale"):
        _d = res.get("prot_ratio_denominator")
        print(f"    protection ratio withheld: the denominator's interval includes zero"
              f"{'' if _d is None else ' (' + R._fmt(_d, digits=3) + ')'}; "
              f"read RII_prot below instead")
    if res.get("ordered_scale"):
        print(f"    RII {R._fmt(res['RII_disp_frac'], digits=3)}   "
              f"(>1 = most deprived bear a larger share)")
        print(f"    SII {R._fmt(res['SII_disp_frac'], digits=3)}")
        print(f"    RII_prot {R._fmt(res['RII_prot_frac'], digits=3)}   "
              f"(<1 = most deprived RECEIVE LESS protection)")
        print(f"    SII_prot {R._fmt(res['SII_prot_frac'], digits=3)}")
    else:
        print("    RII/SII not reported: this stratifier has no deprivation ordering")


def _report_pairs(title, pairs):
    print(f"\n    {title}  --  every pair, paired within draw")
    for name, st in pairs.items():
        d = st.get("difference") or {}
        line = (f"      {name:<26} diff {R._fmt(d, digits=4)}"
                f"   above 0 in {st.get('frac_diff_above_0', float('nan')) * 100:5.1f}%")
        if "ratio" in st:
            line += (f"   ratio {R._fmt(st['ratio'], digits=3)}"
                     f"  above 1 in {st['frac_above_1'] * 100:5.1f}%")
        else:
            line += "   [ratio withheld: denominator interval reaches zero]"
        print(line)


def _report_monotone(order, m):
    print(f"      MONOTONE over {' < '.join(str(x) for x in order)}: "
          f"increasing in {m['frac_increasing'] * 100:5.1f}% of {m['n_draws']} draws, "
          f"decreasing in {m['frac_decreasing'] * 100:5.1f}%")
    print("      (one number instead of six contrasts, and no multiplicity to adjust: "
          "this is\n       the summary a multi-level scale should lead with.)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--joint", default="income*workplace_band")
    ap.add_argument("--adjust", default=None,
                    help="Axes to standardise over, '*'-joined. Defaults to every "
                         "axis after the first. For income*workplace_band*age the "
                         "default 'workplace_band*age' is the one that settles the "
                         "protection gradient: adjusting for either alone leaves the "
                         "other, and on the W3 population band-only adjustment makes "
                         "the age imbalance WORSE than no adjustment at all.")
    ap.add_argument("--metric", default="infections_real")
    ap.add_argument("--restrict-age", default="",
                    help="Age bands (decades: 2=20-29 ... 7=70+) to KEEP, restricting "
                         "every table below to that population. '2,3,4,5,6' = ages "
                         "20-69, which is exactly what workplace_attachment.py:337 "
                         "writes into the `workplace_band_wa` column, so the restriction "
                         "reproduces the estimand's own axis from the all-ages joint "
                         "cells -- no re-run. Every output key then carries `_wa`, and "
                         "the band axis is named `workplace_band_wa`, which is what "
                         "export_paper.py's STRATIFIER_PREFERENCE looks for. Requires "
                         "`age` in --joint. Empty = all ages.")
    ap.add_argument("--refresh", action="store_true",
                    help="with --merge, RE-COMPUTE keys this read already owns. Allowed "
                         "only when the file records the identical (joint, "
                         "restricted_age_bands): same read, same inputs, same numbers, "
                         "so overwriting is idempotent. A file can record matching "
                         "provenance and still hold an INCOMPLETE key set if it was "
                         "written by older code -- that is the case this exists for. "
                         "A differently-derived read is still refused.")
    ap.add_argument("--merge", action="store_true",
                    help="Update the existing readout_workplace_band json instead of "
                         "replacing it. Use it for the second of an all-ages / "
                         "working-age pair: the two reads write disjoint keys (`_wa` "
                         "suffixed or not) into ONE file, which is the only file "
                         "readout_to_csv.py merges, so without this the second read "
                         "silently deletes the first one's tables.")
    ap.add_argument("--policy", default=None,
                    help="Which policy's arms to read; required for a multi-policy "
                         "decomposition. Suffixes the output filename.")
    args = ap.parse_args(argv)

    spath = os.path.join(args.outdir, "stratified_arms.csv")
    if not os.path.exists(spath):
        raise SystemExit(f"no stratified_arms.csv in {args.outdir}")
    s = R.select_policy(pd.read_csv(spath), args.policy, spath)
    s, _has_disp = R.materialise_arm_c(s, args.policy)
    s["stratum_kind"] = s["stratum_kind"].astype(str)

    sub = s[s["stratum_kind"] == args.joint]
    if sub.empty:
        raise SystemExit(
            f"no rows for stratum_kind={args.joint!r} in {spath}\n"
            f"  present: {sorted(s['stratum_kind'].unique())}\n"
            f"  re-run run_displacement_decomposition.py with "
            f"--strata '...,{args.joint}' if the cells were never exported.")

    keys = [k.strip() for k in args.joint.split(R.JOINT_SEP) if k.strip()]
    index_kind = keys[0]
    adjust_kind = args.adjust or R.JOINT_SEP.join(keys[1:])
    cond_kind = keys[1]
    rest = [k for k in keys[2:] if k in adjust_kind.split(R.JOINT_SEP)]

    wa = [x.strip() for x in str(args.restrict_age).split(",") if x.strip() != ""]
    suffix = ""
    if wa:
        if "age" not in keys:
            raise SystemExit(
                f"--restrict-age needs an `age` axis in --joint, got {args.joint!r}. "
                f"Without one there is nothing to restrict on and every table below "
                f"would silently be all-ages under a `_wa` name.")
        _ai = keys.index("age")
        _lab0 = sub["stratum"].astype(str).str.split(R.JOINT_SEP, expand=True)
        if _lab0.shape[1] != len(keys):
            raise SystemExit(
                f"stratum labels in {args.joint} have {_lab0.shape[1]} part(s), "
                f"expected {len(keys)}")
        _keepmask = _lab0[_ai].map(R._strip_num).isin(set(wa))
        _n_before = len(sub)
        sub = sub[_keepmask.values]
        if sub.empty:
            raise SystemExit(
                f"--restrict-age {args.restrict_age!r} kept 0 of {_n_before} rows. "
                f"Age values present: "
                f"{sorted(set(_lab0[_ai].map(R._strip_num)))}")
        suffix = "_wa"
        cond_kind = keys[1] + suffix
        print(f"\n  [restricted] age bands {sorted(wa)} kept: {len(sub):,} of "
              f"{_n_before:,} joint rows. Band axis reported as {cond_kind!r}; every "
              f"key below carries {suffix!r}.")

    summary = {"joint": args.joint, "metric": args.metric, "adjust": adjust_kind,
               "policy": args.policy, "displacement_mechanism": _has_disp,
               "restricted_age_bands": sorted(int(x) for x in wa) if wa else None}

    print("=" * 78)
    print(f"  {args.outdir}   {args.joint}")
    print("=" * 78)

    std = R.standardise(sub, args.joint, metric=args.metric, adjust=adjust_kind,
                        with_draws=True)
    if std is None:
        print(f"\n  [skip] too few complete cells to standardise over {adjust_kind}")
    else:
        if len(keys) > 2 and keys[1] in adjust_kind.split(R.JOINT_SEP):
            _l = sub["stratum"].astype(str).str.split(R.JOINT_SEP, expand=True)
            _g = (sub.assign(_ii=_l[0].astype(str), _bb=_l[1].map(R._strip_num))
                     .groupby(["run", "arm", "_ii", "_bb"], as_index=False)
                     .agg({args.metric: "sum", "n_real": "sum"}))
            _g["attack_rate"] = _g[args.metric] / _g["n_real"]
            _g["stratum"] = _g["_ii"] + R.JOINT_SEP + _g["_bb"]
            _ex = R.standardise(_g, f"{index_kind}{R.JOINT_SEP}{cond_kind}",
                                metric=args.metric, adjust=cond_kind, with_draws=True)
            if _ex is not None:
                _kx = f"{index_kind}{suffix}_exposure_fixed_std"
                _dx = _ex.pop("per_draw", {})
                summary[_kx] = _ex
                _ox = R.order_strata({r["stratum"]: None for r in _ex["strata"]},
                                     index_kind)
                _px = _all_pairs_from_draws(_dx, _ox)
                if _px:
                    summary[f"{_kx}_pairs"] = _px
                    _report_pairs(f"{index_kind} pairs, exposure fixed", _px)
                _mx = _monotone_frac(_dx, _ox, True)
                if _mx:
                    summary[f"{_kx}_monotone"] = _mx
                    _report_monotone(_ox, _mx)
                report(f"{index_kind}_exposure_fixed_std", _ex,
                       tag=f"  [{cond_kind}-standardised, age NOT held: the cascade's "
                           f"'exposure fixed' step]")
            else:
                print(f"\n  [skip] {index_kind}_exposure_fixed_std: too few complete "
                      f"cells to standardise over {cond_kind} alone")

        _k1 = f"{index_kind}{suffix}_{adjust_kind}_std"
        _d1 = std.pop("per_draw", {})
        summary[_k1] = std
        report(f"{index_kind}_{adjust_kind}_std",
               std, tag=f" ({adjust_kind}-standardised)")
        _o1 = R.order_strata({r["stratum"]: None for r in std["strata"]}, index_kind)
        _p1 = _all_pairs_from_draws(_d1, _o1)
        if _p1:
            summary[f"{_k1}_pairs"] = _p1
            _report_pairs(f"{index_kind} pairs, {adjust_kind}-standardised", _p1)
        _m1 = _monotone_frac(_d1, _o1, True)
        if _m1:
            summary[f"{_k1}_monotone"] = _m1
            _report_monotone(_o1, _m1)

    if suffix:
        _l1 = sub["stratum"].astype(str).str.split(R.JOINT_SEP, expand=True)
        _ai = keys.index("age")
        _g1 = sub.assign(_i=_l1[0].astype(str),
                         _a=_l1[_ai].map(R._strip_num))
        _sum = (_g1.groupby(["run", "arm", "_i", "_a"], as_index=False)
                   .agg({args.metric: "sum", "n_real": "sum"}))
        if "attack_rate" in sub.columns:
            _sum["attack_rate"] = _sum[args.metric] / _sum["n_real"]

        _m = (_sum.groupby(["run", "arm", "_i"], as_index=False)
                  .agg({args.metric: "sum", "n_real": "sum"}))
        if "attack_rate" in sub.columns:
            _m["attack_rate"] = _m[args.metric] / _m["n_real"]
        _m = _m.assign(stratum=_m["_i"], stratum_kind=index_kind)
        _res = R.analyse(_m, metric=args.metric, kind=index_kind)
        if _res is None:
            print(f"\n  [skip] {index_kind}{suffix}: arms incomplete after the restriction")
        else:
            summary[f"{index_kind}{suffix}"] = _res
            report(f"{index_kind}{suffix}", _res,
                   tag="  [working-age population, no adjustment]")

        _j = _sum.assign(stratum=_sum["_i"] + R.JOINT_SEP + _sum["_a"].astype(str),
                         stratum_kind=f"{index_kind}{R.JOINT_SEP}age")
        _std2 = R.standardise(_j, f"{index_kind}{R.JOINT_SEP}age",
                              metric=args.metric, adjust="age", with_draws=True)
        if _std2 is None:
            print(f"\n  [skip] {index_kind}{suffix}_age_std: too few complete cells")
        else:
            _d3 = _std2.pop("per_draw", {})
            _k3 = f"{index_kind}{suffix}_age_std"
            summary[_k3] = _std2
            report(_k3, _std2, tag="  [working-age population, age-standardised]")
            _o3 = R.order_strata({r["stratum"]: None for r in _std2["strata"]}, index_kind)
            _p3b = _all_pairs_from_draws(_d3, _o3)
            if _p3b:
                summary[f"{_k3}_pairs"] = _p3b
                _report_pairs(f"{index_kind} pairs, age-standardised (working age)", _p3b)

    lab = sub["stratum"].astype(str).str.split(R.JOINT_SEP, expand=True)
    if lab.shape[1] != len(keys):
        raise SystemExit(
            f"stratum labels in {args.joint} have {lab.shape[1]} part(s), expected "
            f"{len(keys)}: {sorted(set(sub['stratum'].astype(str)))[:5]}")
    sub = sub.assign(_i=lab[0].astype(str), _b=lab[1].map(R._strip_num))
    for j, k in enumerate(keys[2:], start=2):
        sub[f"_r{j}"] = lab[j].map(R._strip_num)

    for band in sorted(set(sub["_b"])):
        d = sub[sub["_b"] == band].copy()
        key = f"{index_kind}{suffix}_within_band_{band}"
        if rest:
            r_lab = d[f"_r{2}"].astype(str)
            for j in range(3, len(keys)):
                r_lab = r_lab + R.JOINT_SEP + d[f"_r{j}"].astype(str)
            d["stratum"] = d["_i"] + R.JOINT_SEP + r_lab
            jk = R.JOINT_SEP.join([index_kind] + keys[2:])
            res = R.standardise(d, jk, metric=args.metric,
                                adjust=R.JOINT_SEP.join(keys[2:]), with_draws=True)
            key += "_" + "_".join(keys[2:]) + "_std"
            tag = f"  [{'*'.join(keys[2:])}-standardised within the band]"
        else:
            d["stratum"] = d["_i"]
            res = R.analyse(d, metric=args.metric, kind=index_kind)
            tag = "  [NOT standardised over anything else]"
        if res is None:
            print(f"\n  [skip] {key}: arms incomplete or fewer than 2 strata")
            continue
        _d2 = res.pop("per_draw", {}) if isinstance(res, dict) else {}
        summary[key] = res
        report(f"{index_kind} within workplace_band {_band(band)}", res, tag=tag)
        _o2 = R.order_strata({r["stratum"]: None for r in res["strata"]}, index_kind)
        if _d2:
            _p2 = _all_pairs_from_draws(_d2, _o2)
            if _p2:
                summary[f"{key}_pairs"] = _p2
                _report_pairs(f"{index_kind} pairs within band {_band(band)}", _p2)
            _m2 = _monotone_frac(_d2, _o2, True)
            if _m2:
                summary[f"{key}_monotone"] = _m2
                _report_monotone(_o2, _m2)
        else:
            _p2 = _all_pairs_crude(d, _o2, args.metric)
            if _p2:
                summary[f"{key}_pairs"] = _p2
                _report_pairs(f"{index_kind} pairs within band {_band(band)} (crude)",
                              _p2)

    band_marg = s[s["stratum_kind"] == cond_kind]
    if wa and not band_marg.empty and \
            sorted(int(x) for x in wa) != sorted(NATIVE_WA_BANDS):
        print(f"\n  [restricted] the native {cond_kind!r} marginal is ages "
              f"{sorted(NATIVE_WA_BANDS)} by construction; --restrict-age "
              f"{sorted(int(x) for x in wa)} differs, so the band gradient and its "
              f"exposure protection ratio are SUMMED from the restricted joint cells, "
              f"not read from that marginal.")
        band_marg = band_marg.iloc[0:0]
    if not band_marg.empty:
        band_marg = _drop_sentinel_bands(band_marg, label=f" in the {cond_kind} marginal")
    if not band_marg.empty:
        b = band_marg.assign(stratum=band_marg["stratum"].astype(str)
                             .map(R._strip_num).map(_band))
    else:
        g = (sub.groupby(["run", "arm", "_b"], as_index=False)
                .agg({args.metric: "sum", "n_real": "sum"}))
        g["attack_rate"] = g[args.metric] / g["n_real"]
        b = g.assign(stratum=g["_b"].map(_band), stratum_kind=cond_kind)
    b = _drop_sentinel_bands(b, label=" before the band gradient")
    res = R.analyse(b, metric=args.metric, kind=cond_kind)
    if res is None:
        print(f"\n  [skip] {cond_kind}: arms incomplete or fewer than 2 strata")
    else:
        summary[cond_kind] = res
        report(cond_kind, res, tag="  [exposure order, NOT a deprivation scale]")
        ends = R.order_strata({r["stratum"]: None for r in res["strata"]}, cond_kind)
        pr = _endpoint_ratio(b, ends, args.metric)
        if pr is not None:
            print(f"    exposure protection ratio {ends[-1]} / {ends[0]}   "
                  f"{R._fmt(pr['ratio'], digits=3)}   "
                  f"above 1 in {pr['frac_above_1'] * 100:5.1f}% of draws")
            summary[f"{cond_kind}_exposure_protection_ratio"] = pr
        _p3 = _all_pairs_crude(b, ends, args.metric)
        if _p3:
            summary[f"{cond_kind}_pairs"] = _p3
            _report_pairs(f"{cond_kind} pairs (crude, exposure order)", _p3)
        _m3 = _monotone_frac(R.paired_by_run(b, args.metric), ends, False)
        if _m3:
            summary[f"{cond_kind}_monotone"] = _m3
            _report_monotone(ends, _m3)

    if rest:
        gk = ["run", "arm", "_b"] + [f"_r{j}" for j in range(2, len(keys))]
        g = (sub.groupby(gk, as_index=False)
                .agg({args.metric: "sum", "n_real": "sum"}))
        g["attack_rate"] = g[args.metric] / g["n_real"]
        r_lab = g["_r2"].astype(str)
        for j in range(3, len(keys)):
            r_lab = r_lab + R.JOINT_SEP + g[f"_r{j}"].astype(str)
        g = _drop_sentinel_bands(g, col="_b",
                                 label=" before the standardised band gradient")
        g["stratum"] = g["_b"].map(_band) + R.JOINT_SEP + r_lab
        jk = R.JOINT_SEP.join([cond_kind] + keys[2:])
        adj = R.JOINT_SEP.join(keys[2:])
        sres = R.standardise(g, jk, metric=args.metric, adjust=adj, with_draws=True)
        if sres is None:
            print(f"\n  [skip] {cond_kind}_{adj}_std: too few complete cells to "
                  f"standardise the band gradient over {adj}")
        else:
            draws = sres.pop("per_draw", {})
            key = f"{cond_kind}_{adj}_std"
            summary[key] = sres
            report(cond_kind, sres,
                   tag=f"  [exposure order, {adj}-standardised -- NOT a deprivation "
                       f"scale]")
            ends = R.order_strata({r["stratum"]: None for r in sres["strata"]},
                                  cond_kind)
            spr = _endpoint_ratio_from_draws(draws, ends)
            if spr is not None:
                print(f"    exposure protection ratio {ends[-1]} / {ends[0]}   "
                      f"{R._fmt(spr['ratio'], digits=3)}   "
                      f"above 1 in {spr['frac_above_1'] * 100:5.1f}% of draws"
                      f"   [{adj}-standardised]")
                summary[f"{key}_exposure_protection_ratio"] = spr
            _p4 = _all_pairs_from_draws(draws, ends)
            if _p4:
                summary[f"{key}_pairs"] = _p4
                _report_pairs(f"{key} pairs", _p4)
            _m4 = _monotone_frac(draws, ends, True)
            if _m4:
                summary[f"{key}_monotone"] = _m4
                _report_monotone(ends, _m4)
            else:
                print(f"    [note] the {adj}-standardised exposure protection ratio is "
                      f"withheld: its denominator interval reaches zero, so the "
                      f"interval would be produced by division, not by uncertainty.")
    else:
        print(f"\n  [note] {cond_kind} is reported as the MARGINAL only -- "
              f"--joint {args.joint!r} carries no axis to standardise it over. Band 0 "
              f"holds every child and pensioner, so re-run with "
              f"--joint '{index_kind}{R.JOINT_SEP}{cond_kind}{R.JOINT_SEP}age' before "
              f"quoting the exposure gradient.")

    print("\n" + "=" * 78)
    print("  ON THE PAIRWISE TABLES")
    print("=" * 78)
    print("  A four-level scale has six pairs, and each table above reports all of them\n"
          "  for two measures across however many waves and bands are read. There is NO\n"
          "  multiplicity adjustment, so:\n\n"
          "    - the ENDPOINT contrast is the pre-specified one. Lead with it.\n"
          "    - the MONOTONE fraction is the honest whole-scale summary: one number,\n"
          "      no adjustment needed, and it answers 'is this a gradient' directly.\n"
          "    - the individual pairs are EXPLORATORY. Quote one only to answer a\n"
          "      question that was asked before the table was read, and say so.\n\n"
          "  Every interval here is a description of the ensemble, not a hypothesis\n"
          "  test, and 'above 0 in X% of draws' is the frequency over draws -- at n=60\n"
          "  its resolution is 1/60, so '100%' means 'in all 60', not 'p < 0.017'.")

    out = os.path.join(
        args.outdir, f"readout_workplace_band{R.policy_suffix(args.policy)}.json")
    if args.merge and os.path.exists(out):
        with open(out) as fh:
            prior = json.load(fh)
        clash = sorted(set(prior) & set(summary) - {"joint", "metric", "adjust",
                                                   "policy",
                                                   "displacement_mechanism",
                                                   "restricted_age_bands"})
        if clash:
            _recorded = prior.get("reads") or [{
                "joint": prior.get("joint"),
                "restricted_age_bands": prior.get("restricted_age_bands")}]
            _want = summary.get("restricted_age_bands")
            _same = [r for r in _recorded
                     if r.get("joint") == args.joint
                     and (r.get("restricted_age_bands") or None) == (_want or None)]
            if not (args.refresh and _same):
                raise SystemExit(
                    f"--merge refuses to overwrite {len(clash)} existing key(s) in {out}: "
                    f"{clash[:6]}. Two reads that produce the same key name are the same "
                    f"read, and keeping the second silently would put a differently-"
                    f"derived number under a name the audit already uses.\n"
                    + (f"  This file DOES record the identical read "
                       f"(joint={args.joint!r}, bands={_want}). If it was written by "
                       f"older code its key set can be incomplete -- pass --refresh to "
                       f"recompute the keys this read owns.\n" if _same else
                       f"  This file records a DIFFERENT read: {_recorded}. Wanted "
                       f"joint={args.joint!r}, bands={_want}. Delete the file or fix "
                       f"the joint; --refresh will NOT override this.\n"))
            _added = sorted(set(summary) - set(prior))
            print(f"\n  [refresh] recomputing {len(clash)} key(s) this read owns; "
                  f"{len(_added)} key(s) are NEW to the file.")
            if _added:
                print(f"            new: {_added[:8]}"
                      + (" ..." if len(_added) > 8 else ""))
        prior.update(summary)
        prior["reads"] = (prior.get("reads") or []) + [
            {"joint": args.joint, "restricted_age_bands":
             summary.get("restricted_age_bands")}]
        summary = prior
        print(f"\n  merged into the existing {os.path.basename(out)} "
              f"({len(clash)} clashes)")
    else:
        summary["reads"] = [
            {"joint": args.joint,
             "restricted_age_bands": summary.get("restricted_age_bands")}]
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
