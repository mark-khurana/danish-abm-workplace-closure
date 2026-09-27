"""The headline analysis: separate the REDUCTION effect of closing a setting from the
DISPLACEMENT of that contact into the home.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POLICY_NPIS = {
    "workplace_closure": ["workplace_closure", "partial_workplace_closure"],
    "school_closure": ["school_closure"],
    "gathering_ban": ["gathering_ban"],
    "mask_mandate": ["mask_mandate"],
}

ARMS = [
    ("A_observed", True, True),
    ("B_no_closure", False, False),
    ("C_reduction_only", True, False),
    ("D_displacement_only", False, True),
]


def arms_for(policy, want_d=True):
    """The arms worth SIMULATING for this policy."""
    from danish_abm.npi import SPILLOVER_PRIORS
    members = set(POLICY_NPIS[policy])
    has_spill = any(n in members and abs(m - 1.0) > 1e-9
                    for (n, _), (m, _s) in SPILLOVER_PRIORS.items())
    if not has_spill:
        return [a for a in ARMS if a[0] in ("A_observed", "B_no_closure")], False
    return ([a for a in ARMS if want_d or a[0] != "D_displacement_only"],
            True)


def build_arm_spec(base_spec_factory, policy, closure_on, displacement_on,
                   spillover_scale=1.0):
    """One arm of the 2x2 for a given closure policy."""
    spec = base_spec_factory()
    spec.name = f"{policy}__{'closed' if closure_on else 'open'}" \
                f"__{'displaced' if displacement_on else 'nodisplace'}"
    if float(spillover_scale) != 1.0:
        spec.spillover_scale = float(spillover_scale)
        spec.name += f"__sp{spillover_scale:g}"
    npis = list(POLICY_NPIS[policy])
    if not closure_on:
        spec.remove_npis = npis
    elif not displacement_on:
        spec.spillover_disabled_npis = npis
    return spec


def _totals(res):
    """Population-level outcome totals from a ScenarioResult."""
    out = {}
    for k in ("total_infections", "total_deaths", "total_hospitalizations"):
        v = getattr(res, k, None)
        if v is not None:
            out[k] = float(v)
    return out


_STRATUM_FAILURES = set()

_STRATUM_DROPS = []


JOINT_SEP = "*"


def stratum_keys(by):
    """['income', 'age'] for 'income*age'; ['income'] for 'income'."""
    return [k for k in (p.strip() for p in str(by).split(JOINT_SEP)) if k]


def stratum_labels(d, by):
    """One label per row: 'Q1' for a marginal, 'Q1*3.0' for a joint cell."""
    keys = [k for k in stratum_keys(by) if k in d.columns]
    if not keys:
        return d.index.astype(str) if d.index.name else d.iloc[:, 0].astype(str)
    lab = d[keys[0]].astype(str)
    for k in keys[1:]:
        lab = lab + JOINT_SEP + d[k].astype(str)
    return lab


def _stratified(res, by, arm=None, run=None):
    """Per-stratum outcome rows, via the existing OutcomeAnalysis machinery."""
    oa = getattr(res, "outcome_analysis", None)
    if oa is None:
        print(f"    [warn] stratify_by({by}) skipped for arm={arm} run={run}: the "
              f"result carries no outcome_analysis, so this draw contributes no "
              f"stratified rows to this arm.")
        _STRATUM_DROPS.append({"stratum": by, "arm": arm, "run": run,
                               "why": "no outcome_analysis on the result"})
        return None
    try:
        return oa.stratify_by(stratum_keys(by))
    except Exception as e:
        print(f"    [warn] stratify_by({by}) failed for arm={arm} run={run}: "
              f"{type(e).__name__}: {e}")
        _STRATUM_DROPS.append({"stratum": by, "arm": arm, "run": run,
                               "why": f"{type(e).__name__}: {e}"})
        return None


def decompose(arm_totals, has_displacement=True):
    """Factorial decomposition of the closure policy into benefit and cost."""
    A, B = arm_totals.get("A_observed"), arm_totals.get("B_no_closure")
    C, D = arm_totals.get("C_reduction_only"), arm_totals.get("D_displacement_only")
    if None in (A, B):
        return None
    if C is None:
        if has_displacement:
            return None
        return {k: {
            "arm_A_observed": A[k], "arm_B_no_closure": B[k],
            "net_averted": B[k] - A[k],
            "benefit_of_closure": B[k] - A[k],
            "cost_of_displacement": None,
            "displacement_mechanism": False,
            "identity_residual": 0.0,
        } for k in set(A) & set(B)}
    out = {}
    keys = set(A) & set(B) & set(C)
    for k in keys:
        net = B[k] - A[k]
        benefit = B[k] - C[k]
        cost = A[k] - C[k]
        row = {
            "arm_A_observed": A[k], "arm_B_no_closure": B[k],
            "arm_C_reduction_only": C[k],
            "net_averted": net,
            "benefit_of_closure": benefit,
            "cost_of_displacement": cost,
            "identity_residual": net - (benefit - cost),
        }
        if D is not None and k in D:
            row["arm_D_displacement_only"] = D[k]
            row["cost_without_closure"] = D[k] - B[k]
            row["interaction"] = cost - (D[k] - B[k])
            row["arm_D_is_degenerate"] = abs(D[k] - B[k]) <= 1e-9 * max(abs(B[k]), 1.0)
        out[k] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave", required=True, help="Wave population directory")
    ap.add_argument("--checkpoint", default=None,
                    help="calibrated_params.pt (default: <wave>/calibrated_params.pt)")
    ap.add_argument("--policies", default="workplace_closure,school_closure",
                    help="Comma-separated. Use every policy the wave's schedule "
                         "actually contains -- a policy that is absent trips the "
                         "A==B gate and is reported as a schedule property, not a "
                         "null result. Known: " + ",".join(sorted(POLICY_NPIS)))
    ap.add_argument("--arm-d", choices=("first", "all", "none"), default="first",
                    help="Arm D is degenerate onto arm B by construction and serves "
                         "only as a leak test on the shared spillover code path. "
                         "'first' runs it for the first displacing policy in this "
                         "wave (default), 'all' for every policy, 'none' skips it.")
    ap.add_argument("--n-runs", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strata",
                    default="income,socio13,age,workplace_band,income*age,socio13*age,income*socio13,income*workplace_band",
                    help="Comma-separated strata. A '*' makes a JOINT stratum "
                         "('income*age' = one row per income x age cell). The joint "
                         "cells cost no extra simulation -- they are another groupby "
                         "on a result already in memory -- and they are the only way "
                         "to age-standardise a gradient or to ask whether the income "
                         "gradient survives conditioning on occupation. Export them "
                         "even when unsure: re-reading is cheap, re-running is not.")
    ap.add_argument("--spillover-scale", type=float, default=1.0,
                    help="Scale every NPI's spillover, log-interpolated "
                         "between 0.0 (no displacement at all) and 1.0 "
                         "(the Jarvis 2020 priors as given). The matrix is "
                         "NOT estimated from these data — it moved 0.0%% in "
                         "wave 3's fit and wave 1 reports three of its NPIs "
                         "non-identifiable — so sweep this and report only "
                         "conclusions whose SIGN holds across the range.")
    ap.add_argument("--reference-run", action="store_true",
                    help="Before the arms, run arm A ONCE with no uncertainty scales "
                         "and report its attack rate. Settles whether a gap between "
                         "the ensemble-median attack rate and the calibration's "
                         "fitted level is Jensen (final size is convex in R, so the "
                         "median OUTCOME is not the outcome at the median PARAMETER "
                         "even with scales centred on 1.0) or a parameter-restoration "
                         "discrepancy between the calibration's forward run and "
                         "ScenarioRunner.from_checkpoint — the failure mode this "
                         "codebase has hit repeatedly. Reproduces the calibrated "
                         "level => Jensen, nothing to fix, quote the calibrated "
                         "trajectory rather than the median. Comes out high too => "
                         "the arms are mis-specified and no contrast is valid. "
                         "Costs one run.")
    ap.add_argument("--shard", default=None,
                    help="k/K: simulate only draws k, k+K, k+2K, ... of the n_runs plan. "
                         "Draw i is the same parameter set and random stream as in an "
                         "unsplit run (the plan is built for all n_runs, seeded seed+i), "
                         "so merge_decomposition_shards.py recombines K shards into the "
                         "output one process would have written. The reference run is "
                         "made by shard 0 only.")
    a = ap.parse_args()
    run_indices = None
    shard = None
    if a.shard:
        _k, _K = (int(x) for x in a.shard.split("/"))
        if not (0 <= _k < _K):
            raise SystemExit(f"--shard {a.shard}: need 0 <= k < K")
        shard = [_k, _K]
        run_indices = list(range(_k, a.n_runs, _K))
        print(f"  SHARD {_k}/{_K}: draws {run_indices[:4]}... ({len(run_indices)} of {a.n_runs})",
              flush=True)

    from danish_abm.scenario import ScenarioRunner, ScenarioSpec
    from danish_abm.uncertainty import UncertaintyConfig

    ckpt = a.checkpoint or os.path.join(a.wave, "calibrated_params.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"no checkpoint at {ckpt}. Run the calibration first — the arms must "
            f"share one calibrated pathogen parameter set, or differences between "
            f"them would reflect the fit rather than the policy.")

    dc_path = os.path.join(a.wave, "data_config.yaml")
    dc_file = dc_path if os.path.exists(dc_path) else os.path.join(a.wave, "config.yaml")
    runner = ScenarioRunner.from_checkpoint(ckpt, dc_file, device=a.device)

    try:
        wave_num = int("".join(c for c in os.path.basename(a.wave) if c.isdigit())[:1])
    except Exception:
        wave_num = 1
    unc = UncertaintyConfig.calibrated_scales(wave_num)

    strata = [s.strip() for s in a.strata.split(",") if s.strip()]
    os.makedirs(a.out, exist_ok=True)

    def base_spec():
        return ScenarioSpec(name="base", npi_base="denmark_historical")

    all_totals, strat_rows, gate_info = {}, [], {}
    chan_draws = {}
    run_totals = {}
    reference = None

    policies = [p.strip() for p in a.policies.split(",") if p.strip()]

    if a.reference_run and policies and (shard is None or shard[0] == 0):
        import torch
        print(f"\n{'=' * 66}\n  UNPERTURBED REFERENCE RUN (arm A, no uncertainty)"
              f"\n{'=' * 66}")
        torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        _spec = build_arm_spec(base_spec, policies[0], True, True,
                               spillover_scale=a.spillover_scale)
        _spec.name = "reference_unperturbed"
        _r0 = runner.run(_spec)
        reference = _totals(_r0)
        _n = float(getattr(runner, "n_agents", 0) or 0)
        _inf = reference.get("total_infections")
        if _inf is not None and _n > 0:
            reference["attack_rate"] = _inf / _n
            print(f"  infections {_inf:,.0f} of {_n:,.0f} agents  ->  "
                  f"attack rate {_inf / _n * 100:.1f}%")
            print(f"  Compare against the calibration's fitted attack rate. If they "
                  f"agree, the\n  ensemble median running high is Jensen and the "
                  f"calibrated trajectory is the\n  point estimate. If this run is "
                  f"ALSO high, the arms are mis-specified.")
        _r0.outcome_analysis = None
        del _r0
        gc.collect()

    _d_done = False
    _policy_has_spill = {}
    for policy in policies:
        if policy not in POLICY_NPIS:
            print(f"  skipping unknown policy '{policy}' "
                  f"(known: {sorted(POLICY_NPIS)})")
            continue
        print(f"\n{'=' * 66}\n  POLICY: {policy}\n{'=' * 66}")
        arm_totals = {}

        _want_d = (a.arm_d == "all"
                   or (a.arm_d == "first" and not _d_done))
        _arms, _has_spill = arms_for(policy, want_d=_want_d)
        if not _has_spill:
            print(f"  {policy} has no non-trivial SPILLOVER_PRIORS row: arms C and D\n"
                  f"  are identical to A and B by construction. Simulating A/B only and\n"
                  f"  reporting a benefit with NO cost term (not a cost measured as 0).")
        elif not _want_d:
            print("  arm D omitted (degenerate onto B; tested once per wave)")
        else:
            _d_done = True
        _policy_has_spill[policy] = _has_spill

        for label, closure_on, displacement_on in _arms:
            spec = build_arm_spec(base_spec, policy, closure_on,
                                  displacement_on,
                                  spillover_scale=a.spillover_scale)
            print(f"\n  ── arm {label}: closure={'ON' if closure_on else 'OFF'}, "
                  f"displacement={'ON' if displacement_on else 'OFF'}")

            rows = []
            lp_runs = []
            rt = run_totals.setdefault((policy, label), {"runs": [], "total_infections": [],
                                                         "total_deaths": [],
                                                         "total_hospitalizations": []})

            def cb(run_idx, res, _label=label, _policy=policy, _rows=rows,
                   _lp=lp_runs, _rt=rt):
                _rt["runs"].append(int(run_idx))
                for _k in ("total_infections", "total_deaths", "total_hospitalizations"):
                    _v = getattr(res, _k, None)
                    _rt[_k].append(None if _v is None else float(_v))
                _p = getattr(res, "layer_pressure", None)
                if _p:
                    _lp.append((int(run_idx), dict(_p)))
                for by in strata:
                    df = _stratified(res, by, arm=_label, run=run_idx)
                    if df is None:
                        continue
                    d = df.copy()
                    _lab = stratum_labels(d, by)
                    d["stratum_kind"] = by
                    d["stratum"] = _lab
                    d["arm"] = _label
                    d["policy"] = _policy
                    d["run"] = run_idx
                    _rows.append(d)

            ens = runner.run_ensemble_uncertain(
                spec, uncertainty=unc, n_runs=a.n_runs, seed=a.seed,
                verbose=False, per_run_callback=cb, run_indices=run_indices,
            )

            _METRIC_KEYS = (("total_infections", "total_infections"),
                            ("total_deaths", "total_deaths"),
                            ("total_hosp", "total_hospitalizations"))
            med = {}
            for _ens_key, _out_key in _METRIC_KEYS:
                _st = ens.get(_ens_key)
                if isinstance(_st, dict) and isinstance(_st.get("median"), (int, float)):
                    med[_out_key] = float(_st["median"])
            arm_totals[label] = med
            if not med:
                print(f"     ** no scalar medians in the ensemble result for {label}; "
                      f"keys were {sorted(ens)[:12]} **")
            else:
                print("     median: " + ", ".join(
                    f"{k}={v:,.0f}" for k, v in med.items()))

            if lp_runs:
                _layers = sorted({k for _, d in lp_runs for k in d})
                gate_info[(policy, label)] = {
                    k: float(np.median([d[k] for _, d in lp_runs if k in d]))
                    for k in _layers
                }
                chan_draws[(policy, label)] = {
                    "runs": [r for r, _ in lp_runs],
                    "layers": {k: [(float(d[k]) if k in d else None)
                                   for _, d in lp_runs] for k in _layers},
                }

            if rows:
                d = pd.concat(rows, ignore_index=True)
                strat_rows.append(d)
            del ens, rows, lp_runs
            gc.collect()

        all_totals[policy] = arm_totals

        dec = decompose(arm_totals, has_displacement=_has_spill)
        if dec:
            print(f"\n  ── DECOMPOSITION: {policy}")
            for k, v in sorted(dec.items()):
                print(f"     {k}")
                print(f"       benefit of closure    {v['benefit_of_closure']:+,.0f}")
                _c = v["cost_of_displacement"]
                print("       cost of displacement  " +
                      (f"{_c:+,.0f}" if _c is not None
                       else "no mechanism (no spillover row for this policy)"))
                print(f"       net averted           {v['net_averted']:+,.0f}")
                if abs(v["identity_residual"]) > 1e-6:
                    print(f"       [!] identity residual {v['identity_residual']:+,.3g}")
                if "interaction" in v:
                    print(f"       interaction           {v['interaction']:+,.0f}")

    with open(os.path.join(a.out, "decomposition.json"), "w") as fh:
        json.dump({"totals": all_totals,
                   "decomposition": {
                       p: decompose(t, has_displacement=_policy_has_spill.get(p, True))
                       for p, t in all_totals.items()},
                   "displacement_mechanism": _policy_has_spill,
                   "arm_d_mode": a.arm_d,
                   "reference_unperturbed": reference,
                   "channel_pressure": {f"{p}|{arm}": v
                                        for (p, arm), v in gate_info.items()},
                   "channel_pressure_draws": {f"{p}|{arm}": v
                                              for (p, arm), v in chan_draws.items()},
                   "run_totals": {f"{p}|{arm}": v for (p, arm), v in run_totals.items()},
                   "shard": shard,
                   "n_runs": a.n_runs, "wave": a.wave, "checkpoint": ckpt},
                  fh, indent=2, default=float)
    if strat_rows:
        _sdf = pd.concat(strat_rows, ignore_index=True)
        _sdf.to_csv(os.path.join(a.out, "stratified_arms.csv"), index=False)
        print(f"\n  wrote {a.out}/stratified_arms.csv "
              f"({len(_sdf):,} rows, strata: {sorted(_sdf['stratum_kind'].unique())})")
        _missing = sorted(set(strata) - set(_sdf["stratum_kind"].unique()))
        if _missing:
            print(f"\n  ** {_missing} WERE REQUESTED BUT PRODUCED NO ROWS. The file "
                  f"above is non-empty, so the 'no rows at all' check below does not "
                  f"fire and the missing dimension would be invisible. **")
            _STRATUM_FAILURES.update(_missing)
    else:
        print("\n  ** NO STRATIFIED ROWS WERE PRODUCED. The stratified outcomes are "
              "the point of this analysis, so this is a failure and not a warning. "
              "Check the [warn] stratify_by(...) lines above: the requested stratum "
              "name has to resolve against a column in agents.parquet (see "
              "COLUMN_ALIASES in danish_abm/substeps/utils.py). **")
    print(f"  wrote {a.out}/decomposition.json")

    print(f"\n{'=' * 66}\n  VALIDITY GATES\n{'=' * 66}")
    failures = []
    for policy, _tot in all_totals.items():
        _a = (_tot.get("A_observed") or {}).get("total_infections")
        _b = (_tot.get("B_no_closure") or {}).get("total_infections")
        if _a is not None and _b is not None:
            _rel = abs(_b - _a) / max(abs(_a), 1e-12)
            ok = _rel > 1e-9
            print(f"  {policy}: removing the closure changes the outcome  "
                  f"(A {_a:,.0f} vs B {_b:,.0f})  {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(
                    f"{policy}: arm B is identical to arm A, so this policy has no "
                    f"schedule entry in this wave and there is nothing to decompose. "
                    f"Its benefit, cost and net are all structurally zero — a "
                    f"property of the schedule, NOT a measured null result. Drop it "
                    f"from --policies for this wave. (Check the arm C log line: "
                    f"'displacement suppressed for {policy}(0d)' confirms zero "
                    f"active days.)")

        A = gate_info.get((policy, "A_observed"), {}).get("household")
        C = gate_info.get((policy, "C_reduction_only"), {}).get("household")
        B = gate_info.get((policy, "B_no_closure"), {}).get("household")
        D = gate_info.get((policy, "D_displacement_only"), {}).get("household")
        if None not in (A, C):
            ok = C < A
            print(f"  {policy}: arm C household < arm A  "
                  f"({C:.4g} vs {A:.4g})  {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{policy}: removing displacement did not lower "
                                f"household transmission")
        if None not in (B, D):
            _rel = abs(D - B) / max(abs(B), 1e-12)
            ok = _rel < 1e-6
            print(f"  {policy}: arm D household == arm B  "
                  f"({D:.4g} vs {B:.4g})  {'PASS' if ok else 'FAIL'}"
                  f"  [displacement is gated on npi_active, so removing the closure "
                  f"removes its displacement: D is degenerate by construction]")
            if not ok:
                failures.append(
                    f"{policy}: arm D differs from arm B by {_rel * 100:.2g}%, so "
                    f"spillover is firing for an INACTIVE NPI. Displacement must be "
                    f"conditional on the closure being in force "
                    f"(transition.py active_spill), or the decomposition attributes "
                    f"displacement to a policy that was not applied.")
    if not gate_info:
        print("  [warn] no per-layer pressure surfaced by the runner — gates not "
              "evaluated. Check the layer-pressure export before trusting the "
              "decomposition; a miswired switch would look like a real result.")
    if failures:
        print("\n  GATES FAILED — the displacement switch is not doing what the "
              "decomposition assumes:")
        for f in failures:
            print(f"    - {f}")
        return 1
    if not strat_rows:
        return 1
    if _STRATUM_FAILURES:
        print(f"\n  ** REQUESTED STRATA PRODUCED NO ROWS: {sorted(_STRATUM_FAILURES)}. "
              f"The stratified outcomes are the point of this analysis, so a missing "
              f"dimension is a failed run. **")
        return 1
    if _STRATUM_DROPS:
        _cov = (_sdf.groupby(["arm", "stratum_kind"])["run"].nunique()
                .rename("draws").reset_index())
        _cov["of"] = a.n_runs
        _short = _cov[_cov["draws"] < a.n_runs]
        print(f"\n  DRAW COVERAGE: {len(_STRATUM_DROPS)} (stratum, arm, draw) "
              f"stratifications were skipped — see the [warn] lines above.")
        for _, r in _short.iterrows():
            print(f"    {r['arm']} / {r['stratum_kind']}: {r['draws']} of {a.n_runs} "
                  f"draws ({100.0 * r['draws'] / a.n_runs:.1f}%)")
        _floor = 0.90
        _bad = _cov[_cov["draws"] < _floor * a.n_runs]
        if len(_bad):
            print(f"\n  ** AN ARM LOST MORE THAN {100 * (1 - _floor):.0f}% OF ITS "
                  f"DRAWS. The paired contrasts rest on the surviving draws only, so "
                  f"the interval is narrower in basis than --n-runs {a.n_runs} "
                  f"advertises. Treat this as a failed run. **")
            return 1
    if not any(decompose(t, has_displacement=_policy_has_spill.get(p, True))
               for p, t in all_totals.items()):
        print("\n  ** THE DECOMPOSITION IS EMPTY for every policy. The arms ran but no "
              "scalar totals were read off the ensemble result, so benefit/cost/net "
              "could not be computed. Check the '** no scalar medians **' lines above "
              "against run_ensemble_uncertain's return keys. **")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
