#!/usr/bin/env python
"""
Am/Eu separation: classic fields, continuous indicator fields, and quantum-chemical fields.

``examples/am-eu/AmEu1-all.mol2`` is the americium/europium separation series of Sitnikov et al.,
J. Comput.-Aided Mol. Des. 29 (2015) 233-247 -- 46 extractants, mostly bis-triazinyl pyridines and
bipyridines, with lg SF(Am/Eu) in ``AmEu1-all.txt``.  It is the set that paper introduced the
continuous indicator fields on, and it arrives the way that work left it: Sybyl atom types,
Gasteiger charges in the file, molecules already superimposed, and **no hydrogens** -- 2 in the
whole file of 1283 atoms.

That last point governs everything here.  The published fields were built on these heavy-atom
structures, so the classic and indicator models are run on them as they are.  A quantum
calculation cannot be: a heavy-atom skeleton is a set of radicals, and ORCA would faithfully
compute the wrong molecule.  So the hydrogens are rebuilt -- every atom's implicit-hydrogen count
restored from its valence, then ``AddHs(addCoords=True)``, which leaves every heavy atom exactly
where the alignment put it and places hydrogens at idealised positions.  All 46 come back neutral,
closed-shell and with the formulas their structures imply (median 46 atoms against 28 heavy ones).

Because the two structure sets differ, the classic baseline is reported on both: on the published
heavy-atom structures, which is the number comparable with the paper, and on the hydrogen-rebuilt
ones, which is the number comparable with the quantum models.

Models
------
    classic             electrostatic (file charges), steric, hydrophobic, hba, hbd
    indicator           one continuous indicator field per Sybyl atom type (the 2015 method)
    classic+indicator   both
    counts              atom-type counts: no geometry at all, the alignment-free control
    quantum_charges     the electrostatic field from Hirshfeld charges instead of Gasteiger
    quantum_fukui       classic plus f+, f- and the dual descriptor as per-atom fields
    quantum_desc        classic plus the per-molecule quantum descriptors
    cdft_fitted         classic plus the Fukui function fitted from its grid (cmfqsar/cdft.py)

Everything is scored the same way: the alpha and lambda grids of the other benchmarks, 10-fold
cross-validation, each model choosing its own (alpha, lambda) on the CV it reports, and the best
model also put through a y-randomisation.

Usage
-----
    python benchmarks/run_am_eu.py --stage fields        # no quantum chemistry, seconds
    python benchmarks/run_am_eu.py --stage orca --jobs 8 # the quantum jobs, about an hour
    python benchmarks/run_am_eu.py --stage quantum       # the quantum models
    python benchmarks/run_am_eu.py --report
"""
from __future__ import annotations

import argparse
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

import run_topomer as T  # noqa: E402

from cmfqsar.chemotypes import count_kernel  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields, cif_types_in  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, r2_score, rmse  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "am-eu")
MOL2 = os.path.join(DATA, "AmEu1-all.mol2")
ACTIVITY = os.path.join(DATA, "AmEu1-all.txt")
RESULTS = os.path.join(HERE, "results", "am_eu")
WORKDIR = os.path.join(RESULTS, "orca")

CLASSIC = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
#: the per-molecule quantum descriptors, as in the corrosion benchmark
DESC = ["E_homo", "E_lumo", "gap", "dipole", "hardness", "electrophilicity"]
FUKUI = ["fukui_plus", "fukui_minus", "dual"]
CIF_MIN = 3


def add_hydrogens(mol):
    """The same molecule with its hydrogens restored, heavy atoms left where they are.

    The structures carry no hydrogens, which RDKit reads as radical centres.  Clearing the
    no-implicit-hydrogen flag and sanitising lets the valence model put the hydrogen count back;
    ``addCoords`` then places them without touching a heavy atom.
    """
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
        a.SetNumRadicalElectrons(0)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return Chem.AddHs(out, addCoords=True)


def prepare():
    """(heavy-atom molecules, hydrogen-rebuilt molecules, names, activities)."""
    ds = load_dataset(MOL2)
    y = np.loadtxt(ACTIVITY, skiprows=1)
    if len(y) != len(ds.mols):
        raise SystemExit(f"{len(ds.mols)} molecules but {len(y)} activities")
    names = [f"AmEu-{i + 1:02d}" for i in range(len(ds.mols))]
    for m, n in zip(ds.mols, names):
        m.SetProp("_Name", n)
    heavy = list(ds.mols)
    withh = []
    for m, n in zip(heavy, names):
        h = add_hydrogens(m)
        h.SetProp("_Name", n)
        withh.append(h)
    return heavy, withh, names, np.asarray(y, float)


def kernels(mols, names, physchem, cif_types=(), props=(), descriptors=(), charge_model="file",
            param_scheme="tripos", alphas=T.ALPHAS):
    """The joint kernel of one model at every attenuation, descriptor block included."""
    out = {}
    for alpha in alphas:
        fs = FieldSet.create(physchem=physchem, cif_types=list(cif_types), alpha=alpha,
                             charge_model=charge_model, cif_scheme="sybyl",
                             param_scheme=param_scheme, atom_properties=list(props),
                             descriptors=list(descriptors))
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        table = AtomTable.from_molecules(mf)
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        K = model.prepare_training_kernels(table)
        model.prepare_descriptors(table)
        out[alpha] = model.joint_kernel(K, table.descriptors)
    return out


def best(kerns, y, lams=T.LAMS):
    """(q2, RMSEcv, alpha, lambda) of the best point of the grid, scored by the CV it selects on."""
    found = None
    for alpha, K in kerns.items():
        for lam in lams:
            pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=T.FOLDS, seed=T.SEED)
            q2 = r2_score(y, pred)
            if found is None or q2 > found[0]:
                found = (q2, rmse(y, pred), float(alpha), float(lam))
    return found


def report_row(tag, res, store):
    q2, err, alpha, lam = res
    store[tag] = {"q2": q2, "rmse_cv": err, "alpha": alpha, "lam": lam}
    print(f"    {tag:22s} q2 {q2:+.3f}  RMSEcv {err:.3f}  (alpha {alpha:g}, lam {lam:g})",
          flush=True)


def load(path):
    return json.load(open(path)) if os.path.isfile(path) else {}


def save(res, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {path}", flush=True)


def stage_fields(out_path, scramble=200):
    heavy, withh, names, y = prepare()
    cif = cif_types_in(heavy, "sybyl", CIF_MIN)
    print(f"=== Am/Eu: {len(heavy)} extractants, lg SF from {y.min():+.2f} to {y.max():+.2f} "
          f"(sd {y.std():.2f})", flush=True)
    print(f"    indicator fields for {len(cif)} Sybyl types: {', '.join(cif)}", flush=True)
    res = load(out_path)
    res.update({"set": "am_eu", "n": len(heavy), "cif_types": list(cif), "cif_min": CIF_MIN,
                "alphas": list(T.ALPHAS), "lams": list(T.LAMS), "folds": T.FOLDS,
                "activity": {"min": float(y.min()), "max": float(y.max()), "sd": float(y.std())}})
    report_row("classic", best(kernels(heavy, names, CLASSIC), y), res)
    report_row("classic_rdkit_params",
               best(kernels(heavy, names, CLASSIC, param_scheme="rdkit"), y), res)
    report_row("indicator", best(kernels(heavy, names, [], cif_types=cif), y), res)
    report_row("classic+indicator", best(kernels(heavy, names, CLASSIC, cif_types=cif), y), res)
    report_row("classic_with_H", best(kernels(withh, names, CLASSIC), y), res)
    report_row("indicator_with_H", best(kernels(withh, names, [], cif_types=cif), y), res)
    Kc = count_kernel(heavy, names)
    found = None
    for lam in T.LAMS:
        pred = cv_predict(Kc, y, "krr", {"lam": float(lam)}, folds=T.FOLDS, seed=T.SEED)
        q2 = r2_score(y, pred)
        if found is None or q2 > found[0]:
            found = (q2, rmse(y, pred), float("nan"), float(lam))
    report_row("counts", found, res)

    if scramble:
        from cmfqsar.validation import y_scramble
        tag = max((v["q2"], k) for k, v in res.items()
                  if isinstance(v, dict) and "q2" in v)[1]
        print(f"    y-randomising the best model ({tag}), {scramble} shuffles ...", flush=True)
        spec = {"classic": (heavy, CLASSIC, cif[:0], "tripos"),
                "classic_rdkit_params": (heavy, CLASSIC, cif[:0], "rdkit"),
                "indicator": (heavy, [], cif, "tripos"),
                "classic+indicator": (heavy, CLASSIC, cif, "tripos"),
                "classic_with_H": (withh, CLASSIC, cif[:0], "tripos"),
                "indicator_with_H": (withh, [], cif, "tripos")}.get(tag)
        if spec is not None:
            mols, phys, cifs, params = spec
            kerns = kernels(mols, names, phys, cif_types=cifs, param_scheme=params)

            def pick(yy, _k=kerns):
                # the strict test: the whole procedure, hyper-parameter search included, is
                # repeated on every shuffle, so the question is whether the *selected* model
                # can be reached by chance and not merely whether a fixed one can
                return max(r2_score(yy, cv_predict(K, yy, "krr", {"lam": float(lam)},
                                                   folds=T.FOLDS, seed=T.SEED))
                           for K in _k.values() for lam in T.LAMS)

            sc = y_scramble(pick, y, runs=scramble, seed=T.SEED, reselected=True)
            res["y_scramble"] = {"model": tag, "n": int(sc.runs), "q2": sc.observed,
                                 "mean_q2": sc.mean, "sd": sc.sd,
                                 "max_q2": float(np.max(sc.scrambled)),
                                 "p": sc.p_value, "reselected": True}
            print(f"    y-randomisation (selection redone each time): real q2 {sc.observed:+.3f}, "
                  f"{sc.runs} shuffles mean {sc.mean:+.3f} (sd {sc.sd:.3f}), best shuffle "
                  f"{np.max(sc.scrambled):+.3f}, p = {sc.p_value:.4f}", flush=True)
    save(res, out_path)
    return res


def stage_orca(jobs=8, level=None, maxcore=1800, timeout=5400.0, limit=0):
    from cmfqsar import orca as qm
    _heavy, withh, names, _y = prepare()
    if limit:
        withh, names = withh[:limit], names[:limit]
    cfg = qm.Config(orca=qm.find_orca(), level=level or qm.FAST_LEVEL, opt=False, fukui=True,
                    nprocs=1, maxcore=maxcore, timeout=timeout, workdir=WORKDIR)
    sizes = [m.GetNumAtoms() for m in withh]
    print(f"=== {len(withh)} molecules, {int(np.median(sizes))} atoms at the median, "
          f"{max(sizes)} at the largest; level '{cfg.level}', {jobs} at a time", flush=True)
    t0 = time.time()
    outcomes = qm.annotate_molecules(
        withh, names, cfg, jobs=jobs,
        progress=lambda msg, frac: print(f"    {msg}  ({time.time() - t0:.0f} s)", flush=True))
    good = [oc for oc in outcomes if oc.mol is not None]
    sdf = os.path.join(RESULTS, "am_eu_qm.sdf")
    os.makedirs(RESULTS, exist_ok=True)
    w = Chem.SDWriter(sdf)
    for oc in good:
        w.write(oc.mol)
    w.close()
    print(f"    {len(good)}/{len(withh)} molecules annotated -> {sdf}", flush=True)
    for oc in outcomes:
        if oc.mol is None:
            print(f"      FAILED {oc.name}: {oc.error}", flush=True)
    return sdf


def stage_quantum(out_path, cdft_intervals=100, jobs=6):
    """The quantum models, from the annotated SDF the ORCA stage wrote."""
    sdf = os.path.join(RESULTS, "am_eu_qm.sdf")
    if not os.path.isfile(sdf):
        raise SystemExit(f"run --stage orca first: {sdf} is missing")
    heavy, withh, names, y = prepare()
    done = {m.GetProp("_Name"): m for m in Chem.SDMolSupplier(sdf, removeHs=False)
            if m is not None}
    keep = [i for i, n in enumerate(names) if n in done]
    mols = [done[names[i]] for i in keep]
    use_names = [names[i] for i in keep]
    yq = y[keep]
    cif = cif_types_in(heavy, "sybyl", CIF_MIN)
    res = load(out_path)
    res["quantum_n"] = len(mols)
    print(f"=== quantum models on {len(mols)}/{len(names)} molecules", flush=True)
    # the matched baselines, on exactly these molecules
    report_row("classic_matched", best(kernels(mols, use_names, CLASSIC), yq), res)
    report_row("indicator_matched", best(kernels(mols, use_names, [], cif_types=cif), yq), res)
    report_row("quantum_charges",
               best(kernels(mols, use_names, ["steric", "hydrophobic", "hba", "hbd"],
                            props=["q_hirshfeld"]), yq), res)
    report_row("quantum_fukui", best(kernels(mols, use_names, CLASSIC, props=FUKUI), yq), res)
    report_row("quantum_desc", best(kernels(mols, use_names, CLASSIC, descriptors=DESC), yq), res)
    report_row("quantum_all",
               best(kernels(mols, use_names, CLASSIC, cif_types=cif, props=FUKUI,
                            descriptors=DESC), yq), res)
    save(res, out_path)
    return res


def stage_cdft(out_path, intervals=100, jobs=8, fields=("fukui_plus", "dual"), keep_cubes=False):
    """The conceptual-DFT fields themselves: plotted on a grid, fitted, and used as CMF fields.

    Reuses the quantum jobs already in ``orca/``: only the ``orca_plot`` calls are new.  Both
    readings are scored -- the field kept exactly on a common grid with the kernel done by
    quadrature, and the field fitted to atom-centred Gaussians with the kernel in closed form --
    because ``cmfqsar/cdft.py`` reports those two to agree as predictors while differing by half
    the field, and this is a second series on which to check that.
    """
    import run_cdft_kernels as K
    from cmfqsar.cdft import cdft_fields, fit_field, overlap_kernel, valence_window
    from concurrent.futures import ThreadPoolExecutor

    heavy, withh, names, y = prepare()
    dirs = K.job_dirs(WORKDIR)
    keep = [i for i, n in enumerate(names) if n in dirs]
    mols = [withh[i] for i in keep]
    use = [names[i] for i in keep]
    yq = y[keep]
    res = load(out_path)
    t0 = time.time()
    print(f"=== conceptual-DFT fields on {len(mols)}/{len(names)} molecules, "
          f"{intervals}^3 cubes", flush=True)

    def work(k):
        c = K.cubes_for(dirs[use[k]], intervals, keep=keep_cubes)
        F = cdft_fields(c["n"], c["np1"], c["nm1"])
        win = valence_window(c["n"].values)
        return k, c["n"], {f: F[f] * win for f in fields}

    got = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for k, cube, F in ex.map(work, range(len(mols))):
            got[k] = (cube, F)
            if len(got) % 5 == 0 or len(got) == len(mols):
                print(f"    cubes {len(got)}/{len(mols)}  ({time.time() - t0:.0f} s)", flush=True)

    lo, n, points = K.common_grid([got[k][0].positions for k in got])
    print(f"    common grid {tuple(int(v) for v in n)} at {K.SPACING} A "
          f"= {len(points)} points", flush=True)
    base = kernels(mols, use, CLASSIC)
    for fname in fields:
        order = sorted(got)
        X = np.vstack([K.resample(got[k][0], got[k][1][fname], points) for k in order])
        Kg = K.unit_diagonal((X @ X.T) * K.SPACING ** 3)
        fits = [fit_field(got[k][1][fname], got[k][0], widths=K.WIDTHS, ridge=1e-3)
                for k in order]
        Kf = K.unit_diagonal(overlap_kernel([f.coeffs for f in fits],
                                            [got[k][0].positions for k in order], K.WIDTHS))
        off = ~np.eye(len(order), dtype=bool)
        rel = float(np.mean([f.rel_error for f in fits]))
        r_gf = float(np.corrcoef(Kg[off], Kf[off])[0, 1])
        print(f"    --- {fname}: mean fit error {rel:.1%}, grid vs fitted kernel r {r_gf:+.3f}",
              flush=True)
        res[f"cdft_{fname}_fit_error"] = rel
        res[f"cdft_{fname}_r_grid_fitted"] = r_gf
        for tag, Kq in (("grid", Kg), ("fitted", Kf)):
            alone = best({float("nan"): Kq}, yq)
            report_row(f"cdft_{fname}_{tag}", alone, res)
            with_base = best({a: K.unit_diagonal(base[a] + Kq) for a in T.ALPHAS}, yq)
            report_row(f"classic+cdft_{fname}_{tag}", with_base, res)
    res["cdft_n"] = len(mols)
    res["cdft_intervals"] = intervals
    res["cdft_seconds"] = time.time() - t0
    save(res, out_path)
    return res


def report(out_path):
    res = load(out_path)
    if not res:
        raise SystemExit(f"nothing in {out_path}")
    order = ["counts", "classic", "classic_rdkit_params", "indicator", "classic+indicator",
             "classic_with_H", "indicator_with_H", "classic_matched", "indicator_matched",
             "quantum_charges", "quantum_fukui", "quantum_desc", "quantum_all",
             "cdft_fukui_plus_grid", "cdft_fukui_plus_fitted",
             "classic+cdft_fukui_plus_grid", "classic+cdft_fukui_plus_fitted",
             "cdft_dual_grid", "cdft_dual_fitted",
             "classic+cdft_dual_grid", "classic+cdft_dual_fitted"]
    lines = ["# Am/Eu separation: classic, indicator and quantum-chemical fields", "",
             f"{res.get('n', '?')} extractants (Sitnikov et al. 2015), lg SF(Am/Eu) from "
             f"{res['activity']['min']:+.2f} to {res['activity']['max']:+.2f} "
             f"(sd {res['activity']['sd']:.2f}); the structures carry Sybyl types, Gasteiger "
             "charges and no hydrogens, and are already superimposed.", "",
             f"Indicator fields: {', '.join('`%s`' % t for t in res.get('cif_types', []))}.",
             "Same alpha and lambda grids and the same 10-fold CV throughout, each model choosing "
             "its own (alpha, lambda) on the CV it reports.", "",
             "| model | q2 | RMSEcv | alpha | lambda |", "|---|---:|---:|---:|---:|"]
    for tag in order:
        v = res.get(tag)
        if isinstance(v, dict) and "q2" in v:
            a = "-" if not np.isfinite(v.get("alpha", float("nan"))) else f"{v['alpha']:g}"
            lines.append(f"| {tag} | {v['q2']:+.3f} | {v['rmse_cv']:.3f} | {a} | "
                         f"{v['lam']:g} |")
    if "y_scramble" in res:
        s = res["y_scramble"]
        lines += ["", f"y-randomisation of `{s['model']}`: real q2 {s['q2']:+.3f}, "
                      f"{s['n']} shuffles mean {s['mean_q2']:+.3f}, best {s['max_q2']:+.3f}, "
                      f"p = {s['p']:.4f}."]
    text = "\n".join(lines) + "\n"
    path = os.path.join(RESULTS, "table.md")
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"written {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["fields", "orca", "quantum", "cdft", "report"],
                    default="fields")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--maxcore", type=int, default=1800)
    ap.add_argument("--level", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scramble", type=int, default=200)
    ap.add_argument("--intervals", type=int, default=100)
    ap.add_argument("--keep-cubes", action="store_true")
    args = ap.parse_args(argv)
    out_path = os.path.join(RESULTS, "am_eu.json")
    if args.stage == "fields":
        stage_fields(out_path, scramble=args.scramble)
    elif args.stage == "orca":
        stage_orca(jobs=args.jobs, level=args.level, maxcore=args.maxcore, limit=args.limit)
    elif args.stage == "quantum":
        stage_quantum(out_path)
    elif args.stage == "cdft":
        stage_cdft(out_path, intervals=args.intervals, jobs=args.jobs,
                   keep_cubes=args.keep_cubes)
    else:
        report(out_path)


if __name__ == "__main__":
    main()
