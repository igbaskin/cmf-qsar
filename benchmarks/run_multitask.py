#!/usr/bin/env python
"""
Two endpoints on one CMF kernel: does the sparse one gain, and is the difference predictable?

``cmfqsar/multitask.py`` fits several endpoints on a shared kernel so that selectivity comes out
as a difference of co-fields.  The claim worth testing is the practical one: that an endpoint with
few measurements predicts better when it shares information with a related endpoint with many, and
that the *difference* of the two is itself modellable.

The corrosion database carries two cases.  Its compounds were measured in two media, and the
columns are in the series files: ``mwila_hcl_adj`` (hydrochloric acid) and ``mwila_h2so4_adj``
(sulfuric acid).  On the benzene series that is 244 compounds in HCl of which 40 were also run in
H2SO4, the two agreeing at r = +0.93 -- one rich endpoint, one sparse, strongly related, which is
the structure an Am/Eu separation problem has, and the one series with enough of the sparse
endpoint to test it.  The second case is the potency against the Langmuir adsorption free energy
``dG_ads`` (kJ/mol), which most series carry for most of their compounds: a different measurement
of the same adsorption, *anti*-correlated with the potency (a more negative free energy is a
stronger inhibitor), so the task correlation has to be allowed to go negative.  The difference
of the two is a selectivity only in the first case; in the second the two endpoints are in
different units and only the sharing is of interest.

Reported, all on the same alignment, fields, alpha grid and 10-fold cross-validation **over
molecules** (never over measurements, or the task correlation would carry the answer across the
split):

    rho = 0      the two endpoints modelled separately, sharing only the kernel
    best rho     the sharing chosen by cross-validation
    difference   the cross-validated q2 of the predicted endpoint difference, on the molecules
                 that carry both measurements

Usage
-----
    python benchmarks/run_multitask.py                 # every pair in PAIRS, then the table
    python benchmarks/run_multitask.py --set 0_c1ccccc1 --tasks mwila_hcl_adj mwila_h2so4_adj
    python benchmarks/run_multitask.py --report        # only rebuild results/multitask/table.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

import kernel_cache as KC  # noqa: E402
import run_corrosion as C  # noqa: E402
import run_topomer as T  # noqa: E402

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.multitask import (Observations, cv_predict, fit, score_by_task,  # noqa: E402
                               score_selectivity, search)

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "multitask")
# the task covariance of two tasks is positive definite for -1 < rho < 1, and the grid has to
# cover both signs: the potency and the adsorption free energy of the same compounds are
# *anti*-correlated, and a grid of non-negative rho could never use that relation
RHOS = (-0.95, -0.9, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95)
TASKS = ("mwila_hcl_adj", "mwila_h2so4_adj")
# the pairings reported: the same potency in two media (only the benzene series has enough
# sulfuric-acid measurements), and the potency against the Langmuir adsorption free energy
PAIRS = (("0_c1ccccc1", ("mwila_hcl_adj", "mwila_h2so4_adj")),
         ("0_c1ccccc1", ("activity", "dG_ads")),
         ("2_c1c_nH_cn1", ("activity", "dG_ads")),
         ("11_c1c_nH_c_nH_1", ("activity", "dG_ads")),
         ("12_c1ccoc1", ("activity", "dG_ads")),
         ("13_c1c_nH_nn1", ("activity", "dG_ads")),
         ("161_c1ccc2scnc2c1", ("activity", "dG_ads")))


def columns(path, names):
    """The named activity columns of a series file, NaN where the measurement is missing.

    A column the file does not have at all counts as missing everywhere: the sulfuric-acid
    columns exist only in the series that have such measurements.
    """
    rows = list(csv.DictReader(open(path)))
    out = np.full((len(rows), len(names)), np.nan)
    for i, r in enumerate(rows):
        for t, c in enumerate(names):
            v = r.get(c)
            if v not in (None, ""):
                out[i, t] = float(v)
    return out


def result_path(set_name, tasks, results=RESULTS):
    return os.path.join(results, f"{set_name}__{tasks[0]}__{tasks[1]}.json")


def prepare(set_name, tasks=TASKS, data=None):
    """Molecules of one series, aligned as the corrosion benchmark aligns them, plus the columns."""
    data = data or C.DATA
    path = os.path.join(data, set_name + ".csv")
    ds = load_dataset(path)
    Y = columns(path, tasks)
    core = C.core_of(ds.mols, path)
    keep = [i for i, m in enumerate(ds.mols) if m.GetSubstructMatch(core)
            and np.isfinite(Y[i]).any()]
    mols = [ds.mols[i] for i in keep]
    names = [ds.names[i] for i in keep]
    Y = Y[keep]
    # the alignment reference is the most active large member on the first (richer) endpoint
    anchor = np.where(np.isfinite(Y[:, 0]), Y[:, 0], np.nanmin(Y[:, 0]) - 1.0)
    ref_i, rmsds = C.align_on_core(mols, anchor, core)
    finite = [r for r in rmsds if np.isfinite(r)]
    return mols, names, Y, core, names[ref_i], float(np.median(finite)) if finite else None


def kernels(mols, names, alphas=T.ALPHAS):
    out = {}
    for alpha in alphas:
        fs = FieldSet.create(physchem=T.FIELDS, alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        out[alpha] = KC.joint_training_kernel(model, AtomTable.from_molecules(mf))
    return out


def run(set_name="0_c1ccccc1", tasks=TASKS, results=RESULTS, folds=T.FOLDS):
    t0 = time.time()
    mols, names, Y, core, ref, rmsd = prepare(set_name, tasks)
    obs = Observations.from_columns(Y, names=tasks)
    print(f"=== {set_name}: {len(mols)} molecules on the scaffold {Chem.MolToSmiles(core)}, "
          f"aligned on {ref} (median RMSD {rmsd:.2f} A)", flush=True)
    both = np.isfinite(Y).all(axis=1)
    if min(obs.counts()) < 8:
        print(f"    {tasks[1]} has {obs.counts()[1]} measurements here; nothing to share",
              flush=True)
        return None
    r_both = float(np.corrcoef(Y[both, 0], Y[both, 1])[0, 1]) if both.sum() > 4 else None
    print(f"    {tasks[0]}: {obs.counts()[0]} measured; {tasks[1]}: {obs.counts()[1]}; "
          f"both: {both.sum()}"
          + (f"; the two agree at r = {r_both:+.2f}" if r_both is not None else ""), flush=True)
    Ks = kernels(mols, names)
    print(f"    kernels built ({time.time() - t0:.0f} s)", flush=True)

    res = {"set": set_name, "tasks": list(tasks), "n": len(mols), "counts": obs.counts(),
           "n_both": int(both.sum()), "r_both": r_both, "sd": [float(np.nanstd(Y[:, t]))
                                                             for t in range(Y.shape[1])],
           "core": Chem.MolToSmiles(core), "reference": ref,
           "align_rmsd_median": rmsd, "alphas": list(T.ALPHAS), "lams": list(T.LAMS),
           "rhos": list(RHOS), "folds": folds}

    # separate models: rho = 0, each endpoint on its own, sharing only the kernel
    best_sep = None
    for alpha, K in Ks.items():
        for lam in T.LAMS:
            per = score_by_task(obs, cv_predict(K, obs, rho=0.0, lam=lam, folds=folds, seed=0))
            s = float(np.nanmean([p["q2"] for p in per]))
            if best_sep is None or s > best_sep[0]:
                best_sep = (s, alpha, lam, per)
    res["separate"] = {"mean_q2": best_sep[0], "alpha": best_sep[1], "lam": best_sep[2],
                       "tasks": best_sep[3]}
    print(f"    rho = 0   (separate)  mean q2 {best_sep[0]:+.3f}  "
          f"(alpha {best_sep[1]:g}, lam {best_sep[2]:g})", flush=True)
    for p in best_sep[3]:
        print(f"        {p['task']:18s} n {p['n']:4d}  q2 {p['q2']:+.3f}  "
              f"RMSEcv {p['rmse_cv']:.3f}", flush=True)

    # shared models: rho chosen by cross-validation
    best_sh = None
    for alpha, K in Ks.items():
        rho, lam, s, per = search(K, obs, rhos=RHOS, lams=T.LAMS, folds=folds, seed=0)
        if best_sh is None or s > best_sh[0]:
            best_sh = (s, alpha, rho, lam, per)
    res["shared"] = {"mean_q2": best_sh[0], "alpha": best_sh[1], "rho": best_sh[2],
                     "lam": best_sh[3], "tasks": best_sh[4]}
    print(f"    best rho = {best_sh[2]:g}  (shared)   mean q2 {best_sh[0]:+.3f}  "
          f"(alpha {best_sh[1]:g}, lam {best_sh[3]:g})", flush=True)
    for p in best_sh[4]:
        print(f"        {p['task']:18s} n {p['n']:4d}  q2 {p['q2']:+.3f}  "
              f"RMSEcv {p['rmse_cv']:.3f}", flush=True)

    # the difference: what a selectivity model predicts, on the molecules that carry both
    if both.sum() >= 8:
        K = Ks[best_sh[1]]
        sel = score_selectivity(K, obs, 0, 1, rho=best_sh[2], lam=best_sh[3], folds=folds, seed=0)
        sel0 = score_selectivity(K, obs, 0, 1, rho=0.0, lam=best_sep[2], folds=folds, seed=0)
        res["difference"] = {"q2": sel["q2"], "rmse_cv": sel["rmse_cv"], "n": sel["n"],
                             "q2_separate": sel0["q2"], "rmse_cv_separate": sel0["rmse_cv"],
                             "sd_observed": float(np.std(sel["observed"]))}
        print(f"    difference ({tasks[0]} - {tasks[1]}) on {sel['n']} molecules: "
              f"q2 {sel['q2']:+.3f} shared, {sel0['q2']:+.3f} separate "
              f"(observed sd {np.std(sel['observed']):.2f})", flush=True)

    # the selectivity co-field is a coefficient vector: report its spread, for the GUI to use
    f = fit(Ks[best_sh[1]], obs, rho=best_sh[2], lam=best_sh[3])
    c = f.selectivity_coefficients(0, 1)
    top = np.argsort(-np.abs(c))[:5]
    res["selectivity_coefficients"] = {"max_abs": float(np.abs(c).max()),
                                       "top_molecules": [names[i] for i in top],
                                       "top_values": [float(c[i]) for i in top]}
    print("    the selectivity co-field depends most on: "
          + ", ".join(f"{names[i]} ({c[i]:+.3f})" for i in top[:3]), flush=True)
    res["seconds"] = time.time() - t0
    os.makedirs(results, exist_ok=True)
    out = result_path(set_name, tasks, results)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out}  ({res['seconds']:.0f} s)", flush=True)
    return res


def report(results=RESULTS):
    """One table over every result file: separate against shared, and the difference."""
    files = sorted(f for f in os.listdir(results) if f.endswith(".json"))
    if not files:
        raise SystemExit(f"nothing in {results}")
    lines = ["# Two endpoints on one kernel", "",
             "One row per series and pair of endpoints. *n* is molecules with the first / the "
             "second measurement, *r* their agreement where both exist; q² per endpoint with the "
             "endpoints fitted separately (ρ = 0) and with the task correlation ρ chosen by "
             "cross-validation over molecules; and the q² of the predicted difference on the "
             "molecules that carry both, separate and shared.", "",
             "| series | endpoints | n | r | q² separate | ρ | q² shared | difference q² "
             "(separate → shared) |", "|---|---|---|---|---|---|---|---|"]
    for f in files:
        r = json.load(open(os.path.join(results, f)))
        sep = r["separate"]
        sh = r["shared"]
        q = lambda tasks: " / ".join(f"{p['q2']:+.2f}" for p in tasks)  # noqa: E731
        d = r.get("difference")
        diff = (f"{d['q2_separate']:+.2f} → {d['q2']:+.2f} (n {d['n']})" if d else "–")
        rb = f"{r['r_both']:+.2f}" if r.get("r_both") is not None else "–"
        lines.append(f"| {r['set']} | {r['tasks'][0]} / {r['tasks'][1]} | "
                     f"{r['counts'][0]} / {r['counts'][1]} | {rb} | {q(sep['tasks'])} | "
                     f"{sh['rho']:+g} | {q(sh['tasks'])} | {diff} |")
    out = os.path.join(results, "table.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten {out}")


#: how the README names the series and the pairs
SERIES_NAMES = {"0_c1ccccc1": "benzene", "2_c1c_nH_cn1": "imidazole",
                "11_c1c_nH_c_nH_1": "imidazolium", "12_c1ccoc1": "furan",
                "13_c1c_nH_nn1": "triazole", "161_c1ccc2scnc2c1": "benzothiazole"}
PAIR_NAMES = {("mwila_hcl_adj", "mwila_h2so4_adj"): "HCl / H₂SO₄ potency",
              ("activity", "dG_ads"): "potency / ΔG_ads"}


def readme_table(results=RESULTS):
    """The table of the README section "Two endpoints on one kernel", from the result files."""
    rows = []
    for f in sorted(os.listdir(results)):
        if f.endswith(".json"):
            rows.append(json.load(open(os.path.join(results, f))))
    # the two-media pair first, then by the number of molecules
    rows.sort(key=lambda r: (r["tasks"][0] != "mwila_hcl_adj", -r["n"]))
    lines = ["| series | endpoints | n | r | q² separate | ρ | q² shared | difference q² |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sep, sh, d = r["separate"], r["shared"], r.get("difference")
        q = lambda tasks: " / ".join(f"{p['q2']:.2f}" for p in tasks)  # noqa: E731
        rho = "0" if sh["rho"] == 0 else f"{sh['rho']:+g}"
        diff = f"{d['q2_separate']:.2f} → {d['q2']:.2f} (n {d['n']})" if d else "–"
        rb = f"{r['r_both']:+.2f}" if r.get("r_both") is not None else "–"
        lines.append(f"| {SERIES_NAMES.get(r['set'], r['set'])} | "
                     f"{PAIR_NAMES.get(tuple(r['tasks']), ' / '.join(r['tasks']))} | "
                     f"{r['counts'][0]} / {r['counts'][1]} | {rb} | {q(sep['tasks'])} | {rho} | "
                     f"{q(sh['tasks'])} | {diff} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=None, help="one series (default: every pair in PAIRS)")
    ap.add_argument("--tasks", nargs=2, default=None)
    ap.add_argument("--folds", type=int, default=T.FOLDS)
    ap.add_argument("--report", action="store_true", help="only rebuild table.md")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")      # the table carries rho and q2; a cp1252 console must not stop it
    if args.report:
        report()
        return
    if args.set is None and args.tasks is None:
        for set_name, tasks in PAIRS:
            run(set_name, tasks, folds=args.folds)
    else:
        run(args.set or "0_c1ccccc1", tuple(args.tasks or TASKS), folds=args.folds)
    report()


if __name__ == "__main__":
    main()
