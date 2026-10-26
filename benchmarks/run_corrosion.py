#!/usr/bin/env python
"""
The same benchmark on corrosion inhibitors: a frame nobody has curated.

``examples/corrosion`` and ``examples/corrosion_hcl`` hold one CSV per scaffold-tree node of the
iron corrosion-inhibitor database (see ``examples/scaffold_tree_export.py``): SMILES, the MWILA
potency ``ln[theta / (C (1 - theta))]`` as ``activity``, and the node's scaffold in the
``scaffold`` column.  ``corrosion_hcl`` carries the same compounds protonated for 1 M HCl, which
changes the charges the electrostatic field is built from but not the scaffold match.

These sets are the case the Sutherland and Mittal measurements were approaching: the structures are
two-dimensional, so every frame is one the program invents.  The ordinary model gets the
strongest automatic frame available here -- each molecule rigidly superimposed on the most active
large member of its series *through the scaffold atoms*, which is exactly the superposition a
corrosion-QSAR paper would describe -- and it is measured against

  topomer       the same molecules cut at the scaffold, substituents posed by rule
  substituents  the control: the same cut pieces at the same sites, left where the alignment put
                them, which separates the cut from the pose
  invariant     the rotation-invariant field autocorrelation: the conformer, no frame
  invariant, aligned blocks  the same kernel with a mixing coefficient per channel block (one
                block per pair of field types) set by centred kernel alignment inside every
                cross-validation fold, the h_f of the overlap kernel carried over
  counts        atom-type counts: no geometry at all

Same fields, alpha and lambda grids and 10-fold CV for everything, each model choosing (alpha,
lambda) on the CV it reports.

Usage
-----
    python benchmarks/run_corrosion.py [--hcl] [--sets 1_c1ccncc1] [--min-molecules 15] [--force]
    python benchmarks/run_corrosion.py --report benchmarks/results/corrosion/table.md
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
from rdkit.Chem import rdMolAlign  # noqa: E402

import run_topomer as T  # noqa: E402  - the Sutherland/Mittal benchmark, reused in full

from cmfqsar.chemotypes import count_kernel  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.fields import AtomTable  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               feature_matrix)
from cmfqsar.model import (CMFModel, block_cv_predict, block_fold_kernels,  # noqa: E402
                           cv_predict, r2_score, rmse)
from cmfqsar.topomer import (TopomerSpec, cuttable, series_labels,  # noqa: E402
                             topomer_fields, unique_substituents)

DATA = os.path.join(ROOT, "examples", "corrosion")
DATA_HCL = os.path.join(ROOT, "examples", "corrosion_hcl")
RESULTS = os.path.join(HERE, "results", "corrosion")


def core_of(mols, path):
    """The series' scaffold: the ``scaffold`` column if present, else the file name."""
    for m in mols:
        if m is not None and m.HasProp("scaffold"):
            core = Chem.MolFromSmiles(m.GetProp("scaffold"))
            if core is not None:
                return core
    name = os.path.basename(path)[:-4]
    smi = name.split("_", 1)[1].replace("_", "") if "_" in name else name
    return Chem.MolFromSmiles(smi)


def align_on_core(mols, y, core):
    """Superimpose every molecule on the most active large member through the scaffold atoms.

    Returns (reference index, [RMSD over the scaffold atoms]).  A frame has to come from
    somewhere when the structures are two-dimensional; this is the best automatic one available
    for a series defined by a common ring system, and the RMSDs say how well it did.
    """
    cut = np.quantile(y, 0.75)
    best = [i for i in range(len(mols)) if y[i] >= cut]
    ref_i = max(best, key=lambda i: mols[i].GetNumHeavyAtoms())
    ref = mols[ref_i]
    ref_match = ref.GetSubstructMatch(core)
    rmsds = []
    for i, m in enumerate(mols):
        if i == ref_i:
            rmsds.append(0.0)
            continue
        match = m.GetSubstructMatch(core)
        amap = list(zip(match, ref_match))
        rmsds.append(float(rdMolAlign.AlignMol(m, ref, atomMap=amap)) if amap else float("nan"))
    return ref_i, rmsds


def run_set(path, results=RESULTS, force=False, min_molecules=15):
    name = os.path.basename(path)[:-4]
    out = os.path.join(results, f"{name}.json")
    if os.path.exists(out) and not force:
        print(f"[skip] {out} exists")
        return json.load(open(out))
    t0 = time.time()
    ds = load_dataset(path)
    y_all = np.asarray(ds.activity, float)
    ok = np.isfinite(y_all)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y_all = y_all[ok]
    core = core_of(mols, path)
    if core is None:
        raise SystemExit(f"{name}: no scaffold")
    keep_core = [i for i, m in enumerate(mols) if m.GetSubstructMatch(core)]
    mols = [mols[i] for i in keep_core]
    names = [names[i] for i in keep_core]
    y_all = y_all[keep_core]
    print(f"\n=== {name}: {len(mols)} molecules with the scaffold "
          f"{Chem.MolToSmiles(core)}", flush=True)
    if len(mols) < min_molecules:
        raise SystemExit(f"{name}: only {len(mols)} molecules")

    ref_i, rmsds = align_on_core(mols, y_all, core)
    finite = [r for r in rmsds if np.isfinite(r)]
    print(f"    aligned on {names[ref_i]} through the scaffold: median RMSD "
          f"{np.median(finite):.2f} A", flush=True)

    labels = series_labels(mols, core)
    sel = [i for i, m in enumerate(mols) if cuttable(m, core)]
    spec = TopomerSpec(torsion=180.0)
    found = {i: T.posed_as_found(mols[i], core, labels, spec.spacing) for i in sel}
    sel = [i for i in sel if found[i] is not None]
    print(f"    {len(labels)} attachment points, "
          f"{len(unique_substituents([mols[i] for i in sel], core))} distinct substituents, "
          f"{len(sel)}/{len(mols)} molecules cuttable", flush=True)

    aligned, topo, frag, invariant, keep = [], [], [], {}, None
    weighted = None          # the invariant kernel with aligned block weights, per fold
    for alpha in T.ALPHAS:
        fs = FieldSet.create(physchem=T.FIELDS, alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        aligned.append((fs, mf))
        acs = AutocorrelationSpec()
        invariant[alpha] = autocorrelation_kernel(feature_matrix(mf, acs), n_bins=acs.n_bins)
        if weighted is None:
            # the channel features do not depend on alpha, so once is enough: the scale and the
            # block weights are re-derived from the training part of every fold
            wm = CMFModel(fieldset=fs, method="krr", normalize=True,
                          kernel_source="autocorrelation", ac_block_weights="align")
            wm.training_kernel(AtomTable.from_molecules(mf), y=y_all)
            weighted = (wm, block_fold_kernels(wm, y_all, folds=T.FOLDS, seed=T.SEED))
        if len(sel) >= min_molecules:
            tf, _log = topomer_fields([mols[i] for i in sel], [names[i] for i in sel], fs,
                                      core=core, labels=labels, spec=spec)
            k = [sel[j] for j, f in enumerate(tf) if f is not None]
            if keep is None:
                keep = k
            topo.append((fs, [f for f in tf if f is not None]))
            frag.append((fs, [build_fields(found[i], fs, names[i]) for i in keep]))
        print(f"    alpha {alpha:g}: fields built ({time.time() - t0:.0f} s)", flush=True)

    res = {"set": name, "file": os.path.relpath(path, ROOT), "n": len(mols),
           "core": Chem.MolToSmiles(core), "reference": names[ref_i],
           "align_rmsd_median": float(np.median(finite)) if finite else None,
           "sites": len(labels), "n_cuttable": len(sel),
           "fields": T.FIELDS, "alphas": list(T.ALPHAS), "lams": list(T.LAMS), "folds": T.FOLDS}
    q2, err, alpha, lam = T.best_q2(aligned, y_all, T.ALPHAS, T.LAMS)
    res["aligned"] = {"q2": q2, "rmse_cv": err, "alpha": alpha, "lam": lam, "n": len(y_all)}
    print(f"    {'aligned':12s} q2 {q2:+.3f}  RMSEcv {err:.3f}  (alpha {alpha:g}, lam {lam:g})",
          flush=True)
    if keep is not None and len(keep) >= min_molecules:
        y = y_all[keep]
        kept_aligned = [(fs, [f[i] for i in keep]) for fs, f in aligned]
        for tag, data in (("aligned_kept", kept_aligned), ("topomer", topo),
                          ("aligned_frag", frag)):
            q2, err, alpha, lam = T.best_q2(data, y, T.ALPHAS, T.LAMS)
            res[tag] = {"q2": q2, "rmse_cv": err, "alpha": alpha, "lam": lam, "n": len(y)}
            print(f"    {tag:12s} q2 {q2:+.3f}  RMSEcv {err:.3f}  "
                  f"(alpha {alpha:g}, lam {lam:g})", flush=True)
    else:
        print(f"    topomer: only {len(sel)} molecules can be cut at this scaffold", flush=True)
    best = None
    for alpha, K in invariant.items():
        for lam in T.LAMS:
            pred = cv_predict(K, y_all, "krr", {"lam": float(lam)}, folds=T.FOLDS, seed=T.SEED)
            q2 = r2_score(y_all, pred)
            if best is None or q2 > best[0]:
                best = (q2, rmse(y_all, pred), float(lam))
    res["invariant"] = {"q2": best[0], "rmse_cv": best[1], "lam": best[2], "n": len(y_all)}
    print(f"    {'invariant':12s} q2 {best[0]:+.3f}  RMSEcv {best[1]:.3f}", flush=True)
    wm, fold_kernels = weighted
    best = None
    for lam in T.LAMS:
        pred = block_cv_predict(wm, y_all, {"lam": float(lam)}, fold_kernels)
        q2 = r2_score(y_all, pred)
        if best is None or q2 > best[0]:
            best = (q2, rmse(y_all, pred), float(lam))
    res["invariant_aligned"] = {"q2": best[0], "rmse_cv": best[1], "lam": best[2],
                                "n": len(y_all), "block_weights": wm.block_weight_report()}
    print(f"    {'inv.aligned':12s} q2 {best[0]:+.3f}  RMSEcv {best[1]:.3f}", flush=True)
    Kc = count_kernel(mols, names)
    best = None
    for lam in T.LAMS:
        pred = cv_predict(Kc, y_all, "krr", {"lam": float(lam)}, folds=T.FOLDS, seed=T.SEED)
        q2 = r2_score(y_all, pred)
        if best is None or q2 > best[0]:
            best = (q2, rmse(y_all, pred), float(lam))
    res["counts"] = {"q2": best[0], "rmse_cv": best[1], "lam": best[2], "n": len(y_all)}
    print(f"    {'counts':12s} q2 {best[0]:+.3f}  RMSEcv {best[1]:.3f}", flush=True)
    res["seconds"] = time.time() - t0
    os.makedirs(results, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out}  ({res['seconds']:.0f} s)", flush=True)
    return res


def report(path, results=RESULTS, title="on the corrosion-inhibitor series"):
    rows = []
    for fn in sorted(os.listdir(results)) if os.path.isdir(results) else []:
        if fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(results, fn))))
    if not rows:
        raise SystemExit("no results yet")
    rows.sort(key=lambda r: -r["n"])
    out = [f"# Aligned CMF, topomer CMF and the frameless kernels {title}", "",
           "Series: one scaffold-tree node each, activity = MWILA potency. The structures are 2D,",
           "so the alignment is the program's own: every molecule superimposed on the most active",
           "large member of its series through the scaffold atoms. Same fields "
           f"({', '.join(rows[0]['fields'])}),",
           "same alpha and lambda grids, same 10-fold CV, each model choosing its own (alpha,",
           "lambda) on the CV it reports.", "",
           "| series | n | scaffold | fit RMSD | q2 aligned | q2 topomer | q2 invariant | "
           "q2 invariant, aligned blocks | q2 counts | cuttable |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        topo = "%+.3f" % r["topomer"]["q2"] if "topomer" in r else "-"
        inva = "%+.3f" % r["invariant_aligned"]["q2"] if "invariant_aligned" in r else "-"
        out.append(f"| {r['set']} | {r['n']} | `{r['core']}` | "
                   f"{r['align_rmsd_median']:.2f} A | {r['aligned']['q2']:+.3f} | {topo} | "
                   f"{r['invariant']['q2']:+.3f} | {inva} | {r['counts']['q2']:+.3f} | "
                   f"{r['n_cuttable']}/{r['n']} |")
    have = [r for r in rows if "topomer" in r]
    if have:
        a = np.array([r["aligned_kept"]["q2"] for r in have])
        t = np.array([r["topomer"]["q2"] for r in have])
        f = np.array([r["aligned_frag"]["q2"] for r in have])
        out += ["", "On the %d series with enough cuttable molecules, measured on exactly those "
                    "molecules:" % len(have), "",
                "| series | whole molecules | substituents, as aligned | substituents, posed by "
                "rule | the cut costs | the pose costs |", "|---|---|---|---|---|---|"]
        for r in have:
            out.append(f"| {r['set']} | {r['aligned_kept']['q2']:+.3f} | "
                       f"{r['aligned_frag']['q2']:+.3f} | {r['topomer']['q2']:+.3f} | "
                       f"{r['aligned_kept']['q2'] - r['aligned_frag']['q2']:+.3f} | "
                       f"{r['aligned_frag']['q2'] - r['topomer']['q2']:+.3f} |")
        out += ["", "Mean q2: aligned %+.3f, topomer %+.3f (difference %+.3f); topomer wins %d of "
                    "%d. Mean cost of the cut %+.3f, of the pose %+.3f." % (
                        a.mean(), t.mean(), (a - t).mean(), int((t > a).sum()), len(have),
                        (a - f).mean(), (f - t).mean())]
    allr = np.array([r["aligned"]["q2"] for r in rows])
    inv = np.array([r["invariant"]["q2"] for r in rows])
    cnt = np.array([r["counts"]["q2"] for r in rows])
    out += ["", "Over all %d series: aligned %+.3f, invariant %+.3f, counts %+.3f; the invariant "
                "kernel wins %d, the aligned model %d." % (
                    len(rows), allr.mean(), inv.mean(), cnt.mean(), int((inv > allr).sum()),
                    int((allr > inv).sum()))]
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hcl", action="store_true",
                    help="the 1 M HCl set (protonated structures) instead of the neutral one")
    ap.add_argument("--sets", default=None, help="comma-separated file stems; default all")
    ap.add_argument("--min-molecules", type=int, default=15)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", default=None, metavar="PATH")
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    data = DATA_HCL if args.hcl else DATA
    results = RESULTS + "_hcl" if args.hcl else RESULTS
    title = ("on the corrosion-inhibitor series, protonated for 1 M HCl" if args.hcl
             else "on the corrosion-inhibitor series")
    if args.report:
        report(args.report, results=results, title=title)
        return
    stems = args.sets.split(",") if args.sets else sorted(
        f[:-4] for f in os.listdir(data) if f.endswith(".csv"))
    failed = []
    for stem in stems:
        try:
            run_set(os.path.join(data, f"{stem}.csv"), results=results, force=args.force,
                    min_molecules=args.min_molecules)
        except (SystemExit, Exception) as exc:        # noqa: BLE001 - one series, not the run
            failed.append(f"{stem}: {exc}")
            print(f"    [failed] {stem}: {exc}", flush=True)
    for line in failed:
        print("    not measured -- " + line[:160])


if __name__ == "__main__":
    main()
