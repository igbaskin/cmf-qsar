#!/usr/bin/env python
"""
What the alignment is worth: aligned CMF against topomer CMF on the eight Sutherland sets.

The Sutherland et al. (2004) data sets come with one aligned conformation per molecule, produced
by hand-curated superposition -- the frame the ordinary CMF model lives in.  Topomer CMF
(``cmfqsar/topomer.py``, after Cramer's topomer CoMFA) throws that frame away: it cuts every
molecule at the bonds holding its substituents to a common core and rebuilds each substituent in
a pose that is a function of its topology alone, so no superposition is ever computed.  This
script measures the two on the same molecules, with the same fields, the same attenuation and
regularisation grids and the same folds, and adds the alignment-free count baseline
(``chemotypes.count_kernel``) as the floor: a kernel with no geometry in it at all.

Three numbers per set, so the comparison cannot hide anything:

  aligned (kept)   the ordinary model on exactly the molecules the topomer route could cut,
                   which is the only fair comparison
  topomer          the rule-based frame on those molecules
  counts           atom-type counts on those molecules: what is left when geometry goes

and, for context, the ordinary model on the whole set and how much of the set survived the cut.

Both field models select (alpha, lambda) by the same 10-fold CV they are scored on, so both
numbers are optimistic to the same degree; the difference between them is the quantity of
interest, not their absolute size.

Usage
-----
    python benchmarks/run_topomer.py [--sets ACE,THR] [--force]
    python benchmarks/run_topomer.py --report benchmarks/results/topomer/table.md
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

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.chemotypes import chemotype_labels, count_kernel  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.model import AtomTable, CMFModel, cv_predict, r2_score, rmse  # noqa: E402
from cmfqsar.topomer import (TopomerSpec, attachment_label, cuttable,  # noqa: E402
                             series_labels, substituents, topomer_fields, unique_substituents)

RESULTS = os.path.join(HERE, "results", "topomer")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
FOLDS = 10
SEED = 0


def best_q2(mf, y, alphas, lams, folds=FOLDS, seed=SEED):
    """Best (q2, RMSEcv, alpha, lambda) over the grid, scored by the same CV it selects on."""
    best = None
    for alpha, fields in zip(alphas, mf):
        model = CMFModel(fieldset=fields[0], method="krr", normalize=True)
        K = model.joint_kernel(model.prepare_training_kernels(
            AtomTable.from_molecules(fields[1])))
        for lam in lams:
            pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=folds, seed=seed)
            q2 = r2_score(y, pred)
            if best is None or q2 > best[0]:
                best = (q2, rmse(y, pred), float(alpha), float(lam))
    return best


def posed_as_found(mol, core, labels, spacing=100.0):
    """The same substituents the topomer route poses, left in their aligned coordinates.

    This is the control that separates the two things the topomer construction does at once.  It
    cuts the same bonds and keeps the same atoms with the same weights, and it parks every
    substituent at its own site exactly as the topomer molecule does -- but each one keeps the
    position and conformation the published superposition gave it instead of being rebuilt by the
    rule.  Scored against the topomer column it measures the pose rule alone, since the atoms and
    their weights are identical and only the coordinates differ.  Scored against the
    whole-molecule column it measures the cut: the scaffold's own field, and the geometry between
    one substituent and another, which separated sites destroy along with it.
    """
    pieces = []
    for frag in substituents(mol, core):
        label = attachment_label(frag)
        if label not in labels:
            continue
        here = Chem.Mol(frag)
        if not here.GetNumConformers():
            return None
        conf = here.GetConformer()
        shift = spacing * list(labels).index(label)
        for i in range(here.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, [p.x + shift, p.y, p.z])
        pieces.append(here)
    if not pieces:
        return None
    out = pieces[0]
    for piece in pieces[1:]:
        out = Chem.CombineMols(out, piece)
    return out


def pick_core(mols):
    """The scaffold a topomer model is built on: the framework of the largest ring chemotype.

    Taking the maximum common substructure of a whole Sutherland set gives nothing usable -- the
    sets are chemotype mixtures, and the MCS of all of them is a fragment like ``CCO`` that is
    fused to every ring system and cuts off nothing.  A topomer model is a congeneric-series
    method, so the series has to be picked out first: the Bemis-Murcko framework shared by the
    largest group of molecules is the scaffold, and the series is every molecule of the set that
    framework can be cut off cleanly, at acyclic single bonds (so a molecule carrying an extra
    ring on a substituent joins the series, with that ring inside the R-group where it belongs).
    """
    labels, scaffolds = chemotype_labels(mols)
    sizes = [int((labels == g).sum()) for g in range(int(labels.max()) + 1)]
    for g in np.argsort([-s for s in sizes]):
        smi = scaffolds[g] if g < len(scaffolds) else ""
        core = Chem.MolFromSmiles(smi) if smi and smi != "(acyclic)" else None
        if core is not None and core.GetNumAtoms() > 2:
            return core, smi, sizes[g]
    return None, "", 0


def run_set(name, force=False):
    out = os.path.join(RESULTS, f"{name}.json")
    if os.path.exists(out) and not force:
        print(f"[skip] {out} exists")
        return json.load(open(out))
    t0 = time.time()
    ds = load_dataset(os.path.join(DATA, f"{name}.sdf"))
    y_all = np.asarray(ds.activity, float)
    ok = np.isfinite(y_all)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y_all = y_all[ok]
    print(f"\n=== {name}: {len(mols)} molecules", flush=True)

    core, core_smiles, group = pick_core(mols)
    if core is None:
        raise SystemExit(f"{name}: no ring scaffold to build a series on")
    sel = [i for i, m in enumerate(mols) if cuttable(m, core)]
    series = [mols[i] for i in sel]
    labels = series_labels(series, core)
    print(f"    core {core_smiles}  ({core.GetNumAtoms()} atoms, framework of {group} "
          f"molecules, cuts cleanly off {len(sel)})", flush=True)
    print(f"    {len(labels)} attachment points, "
          f"{len(unique_substituents(series, core))} distinct substituents", flush=True)

    spec = TopomerSpec(torsion=180.0)
    found = {i: posed_as_found(mols[i], core, labels, spec.spacing) for i in sel}
    sel = [i for i in sel if found[i] is not None]
    series = [mols[i] for i in sel]
    aligned, topo, frag, keep = [], [], [], None
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit")
        aligned.append((fs, [build_fields(m, fs, n) for m, n in zip(mols, names)]))
        tf, log = topomer_fields(series, [names[i] for i in sel], fs, core=core, labels=labels,
                                 spec=spec)
        k = [sel[j] for j, f in enumerate(tf) if f is not None]
        if keep is None:
            keep = k
            for line in log[:3]:
                print("    " + line[:140], flush=True)
        elif k != keep:                                # the pose rule is alpha-independent
            raise SystemExit(f"{name}: the kept set changed with alpha")
        topo.append((fs, [f for f in tf if f is not None]))
        frag.append((fs, [build_fields(found[i], fs, names[i]) for i in keep]))
        print(f"    alpha {alpha:g}: fields built ({time.time() - t0:.0f} s)", flush=True)
    if len(keep) < 10:
        raise SystemExit(f"{name}: only {len(keep)} molecules could be cut")

    y = y_all[keep]
    kept_aligned = [(fs, [f[i] for i in keep]) for fs, f in aligned]
    res = {"set": name, "n_all": len(mols), "n_kept": len(keep), "core": core_smiles,
           "core_atoms": int(core.GetNumAtoms()), "core_group": group, "sites": len(labels),
           "cuttable": len(sel) / float(len(mols)),
           "unique_substituents": len(unique_substituents(series, core)),
           "fields": FIELDS, "alphas": list(ALPHAS), "lams": list(LAMS), "folds": FOLDS}
    for tag, data, yy in (("aligned_kept", kept_aligned, y), ("topomer", topo, y),
                          ("aligned_frag", frag, y), ("aligned_all", aligned, y_all)):
        q2, err, alpha, lam = best_q2(data, yy, ALPHAS, LAMS)
        res[tag] = {"q2": q2, "rmse_cv": err, "alpha": alpha, "lam": lam, "n": len(yy)}
        print(f"    {tag:13s} q2 {q2:+.3f}  RMSEcv {err:.3f}  (alpha {alpha:g}, lam {lam:g})",
              flush=True)
    Kc = count_kernel([mols[i] for i in keep], [names[i] for i in keep])
    best = None
    for lam in LAMS:
        pred = cv_predict(Kc, y, "krr", {"lam": float(lam)}, folds=FOLDS, seed=SEED)
        q2 = r2_score(y, pred)
        if best is None or q2 > best[0]:
            best = (q2, rmse(y, pred), float(lam))
    res["counts"] = {"q2": best[0], "rmse_cv": best[1], "lam": best[2], "n": len(y)}
    print(f"    {'counts':13s} q2 {best[0]:+.3f}  RMSEcv {best[1]:.3f}  (lam {best[2]:g})",
          flush=True)
    res["seconds"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out}  ({res['seconds']:.0f} s)", flush=True)
    return res


def report(path):
    rows = []
    for name in SETS:
        p = os.path.join(RESULTS, f"{name}.json")
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    if not rows:
        raise SystemExit("no results yet")
    out = ["# The alignment against the topomer rule, on the Sutherland sets", "",
           "Aligned CMF (the published superposition), topomer CMF (substituents posed by rule,",
           "no superposition) and the alignment-free count baseline, all on the same molecules:",
           "the congeneric series each set contains, found as the Bemis-Murcko framework of its",
           "largest chemotype plus every molecule that framework cuts cleanly off. Same fields "
           f"({', '.join(rows[0]['fields'])}),",
           f"same alpha grid ({', '.join('%g' % a for a in rows[0]['alphas'])}), same lambda grid,"
           f" same {rows[0]['folds']}-fold CV, and both field",
           "models choose (alpha, lambda) by the CV they are scored on, so both are optimistic to",
           "the same degree.", "",
           "| set | n | series | core | sites | q2 aligned | q2 topomer | q2 counts | loss |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        loss = r["aligned_kept"]["q2"] - r["topomer"]["q2"]
        out.append(f"| {r['set']} | {r['n_kept']}/{r['n_all']} | {r['cuttable']:.0%} | "
                   f"`{r['core']}` | {r['sites']} | {r['aligned_kept']['q2']:+.3f} | "
                   f"{r['topomer']['q2']:+.3f} | {r['counts']['q2']:+.3f} | "
                   f"{loss:+.3f} |")
    mean = np.mean([r["aligned_kept"]["q2"] - r["topomer"]["q2"] for r in rows])
    out += ["", f"Mean loss from giving up the alignment: {mean:+.3f} q2 over {len(rows)} sets.",
            "",
            "Splitting that loss in two. The middle column keeps the same cut substituents at the",
            "same separated sites but leaves each one exactly where the published superposition",
            "put it, so the step to its left is the price of the cut itself -- the scaffold's own",
            "field and the geometry between one substituent and another, both of which separated",
            "sites destroy -- and the step to its right is the price of the pose rule alone, with",
            "the same atoms and the same weights in both.", "",
            "| set | q2 whole molecules | q2 substituents, as aligned | q2 substituents, posed "
            "by rule | cut costs | pose costs |",
            "|---|---|---|---|---|---|"]
    for r in rows:
        if "aligned_frag" not in r:
            continue
        out.append(f"| {r['set']} | {r['aligned_kept']['q2']:+.3f} | "
                   f"{r['aligned_frag']['q2']:+.3f} | {r['topomer']['q2']:+.3f} | "
                   f"{r['aligned_kept']['q2'] - r['aligned_frag']['q2']:+.3f} | "
                   f"{r['aligned_frag']['q2'] - r['topomer']['q2']:+.3f} |")
    have = [r for r in rows if "aligned_frag" in r]
    if have:
        out += ["", "Mean price of the cut %+.3f, mean price of the pose %+.3f." % (
            np.mean([r["aligned_kept"]["q2"] - r["aligned_frag"]["q2"] for r in have]),
            np.mean([r["aligned_frag"]["q2"] - r["topomer"]["q2"] for r in have]))]
    out += ["", "| set | RMSEcv aligned | RMSEcv topomer | aligned on all n |",
            "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['set']} | {r['aligned_kept']['rmse_cv']:.3f} | "
                   f"{r['topomer']['rmse_cv']:.3f} | {r['aligned_all']['q2']:+.3f} "
                   f"({r['aligned_all']['n']}) |")
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", default=None, metavar="PATH")
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if args.report:
        report(args.report)
        return
    for name in [s for s in args.sets.split(",") if s]:
        run_set(name, force=args.force)


if __name__ == "__main__":
    main()
