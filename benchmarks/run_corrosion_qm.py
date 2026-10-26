#!/usr/bin/env python
"""
Do quantum-chemical fields and quantum-chemical descriptors improve a corrosion model?

The corrosion-inhibitor series are the place to ask. A corrosion inhibitor works by donating
electron density to the iron surface, which is exactly what a frontier orbital energy, a chemical
hardness or a condensed Fukui index is about, and the DFT-descriptor literature on these compounds
is large. The fields, meanwhile, have the hardest possible job here: the structures are
two-dimensional, so the frame is one the program invents (every molecule superimposed on the most
active large member of its series through the scaffold atoms), and ``run_corrosion.py`` measures
that the resulting models are modest -- q2 from about zero to +0.7 depending on the series.

So this is the comparison that matters: against the same series, the same alignment, the same
alpha and lambda grids and the same 10-fold cross-validation,

  base          the five fields of the published table (Gasteiger electrostatic, steric,
                hydrophobic, hba, hbd)
  qm_charges    the same, but the electrostatic field replaced by a field of Hirshfeld charges
                from the quantum calculation
  base+fukui    base plus three per-atom reactivity fields: f+, f- and the dual descriptor
  base+desc     base plus the per-molecule quantum descriptors as a kernel block
  base+both     base plus both
  desc_only     the quantum descriptors alone, no geometry at all: the control that says whether
                the fields are worth their cost

``desc_only`` is the control the DFT-descriptor papers implicitly claim: six numbers per molecule,
no pose, no field. If it matches the field models, the geometry in this series adds nothing.

ORCA is run once per molecule (single point at the molecule's aligned geometry -- never optimised,
which would unalign the series -- plus the N+1 and N-1 states for the Fukui indices), cached under
``benchmarks/results/corrosion_qm/<set>/orca``, and the annotated molecules are written beside it
as an SDF so the models can be rerun without ORCA.

Usage
-----
    python benchmarks/run_corrosion_qm.py --sets 7_c1nncs1 [--limit 2] [--jobs 4]
    python benchmarks/run_corrosion_qm.py                      # every set in SETS
    python benchmarks/run_corrosion_qm.py --report benchmarks/results/corrosion_qm/table.md
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

import kernel_cache as KC  # noqa: E402
import run_corrosion as C  # noqa: E402
import run_topomer as T  # noqa: E402

from cmfqsar import orca as qm  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.descriptors import DescriptorScaler, descriptor_kernel, read_values  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, r2_score, rmse  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "corrosion_qm")

#: series to run: small enough for a quantum calculation, and spanning the range of baseline
#: quality that ``run_corrosion.py`` found (q2 from -0.04 to +0.69)
SETS = ["7_c1nncs1", "12_c1ccoc1", "161_c1ccc2scnc2c1", "13_c1c_nH_nn1"]

#: per-molecule quantum descriptors: the set a DFT corrosion-QSAR paper would use
DESC = ["E_homo", "E_lumo", "gap", "dipole", "hardness", "electrophilicity"]
#: per-atom reactivity fields
FUKUI = ["fukui_plus", "fukui_minus", "dual"]

#: (tag, physchem fields, per-atom property fields, per-molecule descriptors)
MODELS = [
    ("base", T.FIELDS, [], []),
    ("qm_charges", ["steric", "hydrophobic", "hba", "hbd"], ["q_hirshfeld"], []),
    ("base+fukui", T.FIELDS, FUKUI, []),
    ("base+desc", T.FIELDS, [], DESC),
    ("base+both", T.FIELDS, FUKUI, DESC),
]


def prepare(path, limit=None):
    """The molecules of one series, filtered to the scaffold and aligned as the baseline aligns."""
    ds = load_dataset(path)
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y = y[ok]
    core = C.core_of(mols, path)
    keep = [i for i, m in enumerate(mols) if m.GetSubstructMatch(core)]
    mols, names, y = [mols[i] for i in keep], [names[i] for i in keep], y[keep]
    if limit:
        order = np.argsort([-m.GetNumHeavyAtoms() for m in mols])[:limit]
        mols = [mols[i] for i in order]
        names = [names[i] for i in order]
        y = y[list(order)]
    ref_i, rmsds = C.align_on_core(mols, y, core)
    finite = [r for r in rmsds if np.isfinite(r)]
    return mols, names, y, core, ref_i, (float(np.median(finite)) if finite else None)


def run_orca(mols, names, workdir, level, jobs, fukui=True, timeout=3600.0, maxcore=1800):
    """ORCA over the aligned molecules; returns the annotated ones and the failures."""
    cfg = qm.Config(orca=qm.find_orca(), level=level, opt=False, fukui=fukui,
                    nprocs=1, maxcore=maxcore, timeout=timeout, workdir=workdir)
    t0 = time.time()
    done = {"n": 0}

    def progress(msg, frac):
        done["n"] += 1
        print(f"      {msg}  ({time.time() - t0:.0f} s)", flush=True)

    outcomes = qm.annotate_molecules(mols, names, cfg, jobs=jobs, progress=progress)
    good = {oc.index - 1: oc.mol for oc in outcomes if oc.mol is not None}
    failed = [(oc.name, oc.error) for oc in outcomes if oc.mol is None]
    fresh = [oc.seconds for oc in outcomes if oc.mol is not None and not oc.reused]
    if fresh:
        print(f"    {len(fresh)} molecules computed, {np.mean(fresh):.0f} s each on average "
              f"({jobs} at a time, {time.time() - t0:.0f} s in all)", flush=True)
    return good, failed


def kernel_for(mols, names, physchem, props, descriptors, alpha):
    """The joint kernel of one model at one attenuation, descriptor block included."""
    fs = FieldSet.create(physchem=physchem, alpha=alpha, charge_model="gasteiger",
                         param_scheme="rdkit", atom_properties=props, descriptors=descriptors)
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    return KC.joint_training_kernel(model, table)


def best_over_grid(kernels, y, lams=T.LAMS, folds=T.FOLDS, seed=T.SEED):
    """Best (q2, RMSEcv, alpha, lambda) over the grid, scored by the CV it selects on."""
    best = None
    for alpha, K in kernels:
        for lam in lams:
            pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=folds, seed=seed)
            q2 = r2_score(y, pred)
            if best is None or q2 > best[0]:
                best = (q2, rmse(y, pred), float(alpha), float(lam))
    return best


def descriptors_only(mols, y, names_d=DESC, lams=T.LAMS):
    """The quantum descriptors with no geometry at all: the control."""
    Z = np.vstack([read_values(m, names_d) for m in mols])
    scaler = DescriptorScaler.fit(Z, tuple(names_d))
    Zs = scaler.transform(Z)
    K = descriptor_kernel(Zs, Zs)
    K = K / max(np.mean(np.diag(K)), 1e-12)
    best = None
    for lam in lams:
        pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=T.FOLDS, seed=T.SEED)
        q2 = r2_score(y, pred)
        if best is None or q2 > best[0]:
            best = (q2, rmse(y, pred), float("nan"), float(lam))
    return best


def run_set(name, data=C.DATA, results=RESULTS, limit=None, jobs=4, level=qm.FAST_LEVEL,
            force=False, models=MODELS, maxcore=1800):
    path = os.path.join(data, name + ".csv")
    out = os.path.join(results, f"{name}.json")
    if os.path.exists(out) and not force:
        print(f"[skip] {out} exists")
        return json.load(open(out))
    t0 = time.time()
    mols, names, y, core, ref_i, rmsd = prepare(path, limit)
    print(f"\n=== {name}: {len(mols)} molecules with the scaffold {Chem.MolToSmiles(core)}, "
          f"aligned on {names[ref_i]} (median RMSD {rmsd:.2f} A)", flush=True)

    workdir = os.path.join(results, name, "orca")
    good, failed = run_orca(mols, names, workdir, level, jobs, maxcore=maxcore)
    if failed:
        print(f"    {len(failed)} molecule(s) failed: "
              + "; ".join(f"{n}: {e}" for n, e in failed[:4]), flush=True)
    keep = sorted(good)
    if len(keep) < 12:
        raise SystemExit(f"{name}: only {len(keep)} molecules survived the quantum calculation")
    qmols = [good[i] for i in keep]
    qnames = [names[i] for i in keep]
    yq = y[keep]
    sdf = os.path.join(results, name, f"{name}_qm.sdf")
    os.makedirs(os.path.dirname(sdf), exist_ok=True)
    w = Chem.SDWriter(sdf)
    for m in qmols:
        w.write(m)
    w.close()
    print(f"    {len(qmols)} molecules with quantum properties -> {sdf}", flush=True)

    res = {"set": name, "n_series": len(mols), "n": len(qmols), "core": Chem.MolToSmiles(core),
           "reference": names[ref_i], "align_rmsd_median": rmsd, "level": level,
           "descriptors": list(DESC), "fukui_fields": list(FUKUI),
           "alphas": list(T.ALPHAS), "lams": list(T.LAMS), "folds": T.FOLDS,
           "failed": [{"name": n, "error": e} for n, e in failed]}
    for tag, physchem, props, desc in models:
        kernels = [(a, kernel_for(qmols, qnames, physchem, props, desc, a)) for a in T.ALPHAS]
        q2, err, alpha, lam = best_over_grid(kernels, yq)
        res[tag] = {"q2": q2, "rmse_cv": err, "alpha": alpha, "lam": lam, "n": len(yq)}
        print(f"    {tag:12s} q2 {q2:+.3f}  RMSEcv {err:.3f}  (alpha {alpha:g}, lam {lam:g})",
              flush=True)
    q2, err, _a, lam = descriptors_only(qmols, yq)
    res["desc_only"] = {"q2": q2, "rmse_cv": err, "lam": lam, "n": len(yq)}
    print(f"    {'desc_only':12s} q2 {q2:+.3f}  RMSEcv {err:.3f}  (lam {lam:g})", flush=True)
    res["seconds"] = time.time() - t0
    os.makedirs(results, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out}  ({res['seconds']:.0f} s)", flush=True)
    return res


TAGS = ["base", "qm_charges", "base+fukui", "base+desc", "base+both", "desc_only"]


def report(results=RESULTS, out=None):
    rows = []
    for fn in sorted(os.listdir(results)):
        if fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(results, fn))))
    rows.sort(key=lambda r: -r["n"])
    lines = ["# Quantum fields and quantum descriptors on the corrosion-inhibitor series", "",
             "Same series, same alignment (every molecule superimposed on the most active large",
             "member through the scaffold atoms), same alpha and lambda grids, same 10-fold CV,",
             "each model choosing its own (alpha, lambda) on the CV it reports. The quantum",
             "properties come from a single point at the aligned geometry (never optimised), with",
             "the N+1 and N-1 states for the Fukui indices.", ""]
    if rows:
        lines += [f"Level of theory: `{rows[0]['level']}`. Descriptors: "
                  + ", ".join(f"`{d}`" for d in rows[0]["descriptors"]) + ". Fukui fields: "
                  + ", ".join(f"`{d}`" for d in rows[0]["fukui_fields"]) + ".", ""]
    lines += ["| series | n | " + " | ".join(f"q2 {t}" for t in TAGS) + " |",
              "|---|---:|" + "---:|" * len(TAGS)]
    for r in rows:
        cells = [f"{r[t]['q2']:+.3f}" if t in r else "-" for t in TAGS]
        lines.append(f"| {r['set']} | {r['n']} | " + " | ".join(cells) + " |")
    if len(rows) > 1:
        means = [np.mean([r[t]["q2"] for r in rows if t in r]) for t in TAGS]
        lines.append("| **mean** | | " + " | ".join(f"**{m:+.3f}**" for m in means) + " |")
    text = "\n".join(lines) + "\n"
    if out:
        with open(out, "w") as fh:
            fh.write(text)
        print(f"written {out}")
    else:
        print(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--hcl", action="store_true", help="the protonated (1 M HCl) structures")
    ap.add_argument("--limit", type=int, default=0, help="only the N largest molecules (a probe)")
    ap.add_argument("--jobs", type=int, default=4, help="molecules at a time")
    ap.add_argument("--level", default=qm.FAST_LEVEL)
    ap.add_argument("--maxcore", type=int, default=1800, help="MB per ORCA job")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=None)
    ap.add_argument("--report", nargs="?", const="-", default=None)
    args = ap.parse_args(argv)
    data = C.DATA_HCL if args.hcl else C.DATA
    results = args.results or (RESULTS + ("_hcl" if args.hcl else ""))
    if args.report:
        return report(results, None if args.report == "-" else args.report)
    for name in args.sets:
        run_set(name, data=data, results=results, limit=args.limit or None, jobs=args.jobs,
                level=args.level, force=args.force, maxcore=args.maxcore)
    if len(args.sets) > 1:
        report(results, os.path.join(results, "table.md"))


if __name__ == "__main__":
    main()
