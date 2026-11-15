#!/usr/bin/env python
"""
Melting points of molecular crystals: molecular fields in the cell against the inorganic set
(plan item M8).

`cmfqsar/molcrystal.py` perceives the molecules of a crystal (bonds across the cell boundary,
then bond orders and charges), gives every atom of the cell the field weights of its molecule,
and hands the cell to the periodic kernels. The plan asks for the measurement on a sublimation
enthalpy or melting point set against the inorganic field set. The set is
`examples/melting_points` (made by `prepare_molcrystals.py`): compounds of the ONS Open Melting
Point Collection matched to COD crystal structures by the identity of the perceived molecules.

The kernels
-----------
* **inorganic, periodic** -- the periodic autocorrelation with the inorganic field set
  (electrostatic from electronegativity differences, steric, electronegativity), which is what
  the program could do with a CIF before this item.
* **molecular, periodic** -- the same periodic autocorrelation with the five molecular fields
  (Gasteiger electrostatic, steric, hydrophobic, acceptor, donor) on the atoms of the cell: the
  packing seen through the molecule's own fields.
* **molecular, molecule only** -- the field autocorrelation of the molecule as it sits in the
  crystal, without its neighbours: the same fields with no packing.
* **molecular, both** -- the two summed.

The periodic rows are per unit volume, so a crystal of large molecules and one of small
molecules at the same density give rows of the same size. Multiplying a row by the volume per
molecule would make it a row per molecule, but the kernel is cosine-normalised, which divides
any such factor out again: the packing kernel cannot see the size of the molecule either way.
This was measured once (a difference of +0.002) and is not repeated here.

And two references that need no crystal at all, because a melting point is first of all a
property of the molecule: the Tanimoto kernel of ECFP4 fingerprints, and a Gaussian kernel on
eight counts that melting-point models usually start from (molecular weight, donors, acceptors,
rotatable bonds, rings, aromatic rings, polar surface area, fraction of sp3 carbon).

The gate, fixed before the measurement
--------------------------------------
Molecular, periodic minus inorganic, periodic: the difference of cross-validated q² with a 95 %
paired bootstrap interval over the compounds; met where the interval lies wholly above zero.
Five folds repeated three times, the window width and the ridge chosen by the evidence on the
training block. The references are reported beside it and do not decide it.

Usage
-----
    python benchmarks/run_molcrystals.py            # the kernels, the fit and the table
    python benchmarks/run_molcrystals.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar import molcrystal as MC  # noqa: E402
from cmfqsar import periodic  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               block_sizes, channel_scale, feature_matrix)

import run_gridfields as RG  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "melting_points")
RESULTS = os.path.join(HERE, "results", "molcrystals")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
#: the window widths of M4 and five narrower: the narrowest offered was chosen in every fold of
#: three runs (0.35, 0.25, then 0.15, by the inorganic kernel), so the list reaches 0.07
SIGMAS = (0.07, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75)
R_MAX = 10.0
TAPER = 2.0
#: both ends were chosen in a first run with 0.5 to 1.0, then 0.35 and 0.2 in every fold of the
#: next two, so the list reaches 0.05
MOL_SIGMAS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5)
MOL_R_MAX = 12.0
#: 2 was chosen at the edge in every fold of a first run, so the list goes on to 8
COUNT_WIDTHS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load():
    """The cells and the table of `examples/melting_points`, in the same order."""
    from ase.io import read

    cells = read(os.path.join(DATA, "crystals.extxyz"), index=":")
    with open(os.path.join(DATA, "melting_points.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_name = {r["name"]: r for r in rows}
    rows = [by_name[a.info["name"]] for a in cells]
    return cells, rows


def molecular_fieldset() -> FieldSet:
    return FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                           param_scheme="rdkit")


CACHE = os.path.join(HERE, "cache", "molcrystals")


def periodic_features(crystals, tag: str, sigma: float) -> np.ndarray:
    """The periodic autocorrelation rows at one window width, kept in the cache.

    The rows take most of the run (about three minutes per width and field set on 665 cells),
    so they are stored with the names of the cells and computed again only when those change.
    """
    path = os.path.join(CACHE, f"features_{tag}_{sigma:g}.npz")
    names = np.array([c.name for c in crystals])
    if os.path.isfile(path):
        z = np.load(path)
        if z["names"].shape == names.shape and np.all(z["names"] == names):
            return z["F"]
    n_f = crystals[0].weights.shape[1]
    spec = periodic.PeriodicSpec(r_max=R_MAX, step=sigma, sigma=sigma, taper=TAPER,
                                 background=True)
    F = periodic.periodic_feature_matrix(crystals, spec, n_fields=n_f)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, F=F, names=names)
    return F


def periodic_kernels(crystals, tag: str, per: Optional[np.ndarray] = None) -> np.ndarray:
    """The periodic autocorrelation kernel at each window width, as M4 builds it.

    The rows are per unit volume, which is what makes a supercell give the same row. `per`
    multiplies every row by a number of the cell -- the volume per molecule turns a row per unit
    volume into a row per molecule, which keeps the size of the molecule in the description.
    """
    n_f = crystals[0].weights.shape[1]
    out = []
    for s in SIGMAS:
        spec = periodic.PeriodicSpec(r_max=R_MAX, step=s, sigma=s, taper=TAPER, background=True)
        F = periodic_features(crystals, tag, s)
        if per is not None:
            F = F * np.asarray(per, float)[:, None]
        blocks = periodic.periodic_block_sizes(spec, n_f)
        sc = channel_scale(F, spec.n_bins, blocks=blocks)
        out.append(autocorrelation_kernel(F, scale=sc, n_bins=spec.n_bins, blocks=blocks))
    return np.stack(out)


def summed(packing: np.ndarray, molecule: np.ndarray) -> np.ndarray:
    """Every packing width with every molecule width, the two kernels averaged."""
    return np.stack([0.5 * (a + b) for a in packing for b in molecule])


def molecule_only(cell_mols, fs) -> List:
    """One copy of each distinct molecule of the cell, as one molecule's fields."""
    out = []
    for mols in cell_mols:
        first: Dict[str, object] = {}
        for c in mols:
            first.setdefault(c.inchikey, c.mol)
        parts = list(first.values())
        combo = parts[0]
        for p in parts[1:]:
            combo = Chem.CombineMols(combo, p)
        out.append(build_fields(combo, fs, "molecule"))
    return out


def molecule_kernels(mfs, fs) -> np.ndarray:
    out = []
    for s in MOL_SIGMAS:
        spec = AutocorrelationSpec(d_max=MOL_R_MAX, step=s, sigma=s)
        F = feature_matrix(mfs, spec, fieldset=fs)
        blocks = block_sizes(spec, len(fs.specs))
        sc = channel_scale(F, spec.n_bins, blocks=blocks)
        out.append(autocorrelation_kernel(F, scale=sc, n_bins=spec.n_bins, blocks=blocks))
    return np.stack(out)


def counts(mols) -> np.ndarray:
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    rows = []
    for m in mols:
        rows.append([Descriptors.MolWt(m), Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
                     rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
                     rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcTPSA(m),
                     rdMolDescriptors.CalcFractionCSP3(m)])
    X = np.asarray(rows, float)
    return (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)


def count_kernels(X: np.ndarray) -> np.ndarray:
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2) / X.shape[1]
    return np.stack([np.exp(-d2 / (2.0 * w * w)) for w in COUNT_WIDTHS])


def run() -> Dict[str, object]:
    t0 = time.time()
    cells, rows = load()
    y = np.array([float(r["mp_C"]) for r in rows])
    fs = molecular_fieldset()
    inorg = periodic.inorganic_fieldset(elements=[], physchem=periodic.INORGANIC_FIELDS,
                                        charge_model="electronegativity", alpha=1.0)
    cell_mols = [MC.cell_molecules(a) for a in cells]
    log(f"{len(cells)} crystals, molecules perceived ({time.time() - t0:.0f} s)")
    mol_cf = [MC.molecular_crystal_fields(a, fs, r["name"], m)
              for a, r, m in zip(cells, rows, cell_mols)]
    inorg_cf = [periodic.crystal_fields(a, inorg, r["name"]) for a, r in zip(cells, rows)]
    K = {"inorganic, periodic": periodic_kernels(inorg_cf, "inorganic"),
         "molecular, periodic": periodic_kernels(mol_cf, "molecular")}
    log(f"periodic kernels ({time.time() - t0:.0f} s)")
    K["molecular, molecule only"] = molecule_kernels(molecule_only(cell_mols, fs), fs)
    K["molecular, both"] = summed(K["molecular, periodic"], K["molecular, molecule only"])
    smiles_mols = [Chem.MolFromSmiles(r["smiles"]) for r in rows]
    from cmfqsar.descriptors import fingerprint_kernel
    K["ECFP4 fingerprint"] = fingerprint_kernel(smiles_mols)[None]
    K["eight counts"] = count_kernels(counts(smiles_mols))
    sig = [f"sigma {s:g}" for s in SIGMAS]
    msig = [f"sigma {s:g}" for s in MOL_SIGMAS]
    both = [f"sigma {a:g} with {b:g}" for a in SIGMAS for b in MOL_SIGMAS]
    labels = {"inorganic, periodic": sig, "molecular, periodic": sig,
              "molecular, molecule only": msig, "molecular, both": both,
              "ECFP4 fingerprint": ["Tanimoto"],
              "eight counts": [f"width {w:g}" for w in COUNT_WIDTHS]}
    models = {k: ("one", v) for k, v in K.items()}
    res = RG.cross_validate(models, y, labels)
    log(f"cross-validation ({time.time() - t0:.0f} s)")
    pairs = [("molecular, periodic", "inorganic, periodic"),
             ("molecular, both", "molecular, molecule only"),
             ("molecular, periodic", "molecular, molecule only"),
             ("molecular, both", "ECFP4 fingerprint"),
             ("molecular, both", "eight counts"),
             ]
    rows_cmp = RG.compare(res, y, pairs)
    n_mol = [len(m) for m in cell_mols]
    multi = sum(1 for m in cell_mols if len(MC.distinct_molecules(m)) > 1)
    payload = {"crystals": len(cells), "target": "melting point, degrees C",
               "y_mean": float(y.mean()), "y_sd": float(y.std()),
               "y_min": float(y.min()), "y_max": float(y.max()),
               "molecules_per_cell_median": float(np.median(n_mol)),
               "multi_component": multi,
               "models": RG.strip(res), "comparisons": rows_cmp,
               "gate": bool(rows_cmp[0]["lo"] > 0),
               "folds": RG.FOLDS, "repeats": RG.REPEATS, "sigmas": list(SIGMAS),
               "r_max": R_MAX, "seconds": round(time.time() - t0, 1)}
    RG.jdump(payload, os.path.join(RESULTS, "melting.json"))
    for m, r in res.items():
        log(f"  {m:28s} q2 {r['q2_mean']:+.3f}  RMSE {r['rmse_mean']:.1f}  {r['chosen']}")
    for r in rows_cmp:
        log(f"  {r['model']} - {r['against']}: {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    return payload


def report(path: Optional[str] = None) -> str:
    p = os.path.join(RESULTS, "melting.json")
    lines = ["# Melting points of molecular crystals (plan item M8)", ""]
    if os.path.isfile(p):
        d = RG.jload(p)
        lines += [f"{d['crystals']} crystals of `examples/melting_points` (ONS melting points "
                  "matched to COD structures by the perceived molecules); melting point "
                  f"{d['y_min']:.0f} to {d['y_max']:.0f} C, spread {d['y_sd']:.1f} C. Five "
                  "folds repeated three times; the window width or kernel width and the ridge "
                  "chosen by the evidence on the training block.", "",
                  "| Kernel | Needs the crystal | q², mean of 3 repeats | RMSE, C | "
                  "Settings chosen (folds) |", "|---|---|---:|---:|---|"]
        for name, r in d["models"].items():
            crystal = "yes" if ("periodic" in name or "molecule only" in name
                                or "both" in name) else "no"
            ch = ", ".join(f"{k} ({v})" for k, v in r["chosen"].items())
            lines.append(f"| {name} | {crystal} | {r['q2_mean']:+.3f} | {r['rmse_mean']:.1f} | "
                         f"{ch} |")
        lines += ["", "| Difference of q² | Mean [95 % interval] | Interval above zero |",
                  "|---|---|---|"]
        for r in d["comparisons"]:
            lines.append(f"| {r['model']} minus {r['against']} | {RG._iv(r)} | "
                         f"{'yes' if r['above_zero'] else 'no'} |")
        lines += ["", f"Gate (molecular, periodic minus inorganic, periodic, interval above "
                  f"zero): {'met' if d['gate'] else 'not met'}.", ""]
    text = "\n".join(lines)
    path = path or os.path.join(RESULTS, "table.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    if not args.report:
        run()
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
