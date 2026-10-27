#!/usr/bin/env python
"""
Does the field around an atom predict a property of that atom? (plan item LP2)

Every other model in this program predicts one number per molecule.  A carbon shift and an
acidity belong to an atom, so the sample has to be the atom and its surroundings:
`cmfqsar/localenv.py` builds that sample by multiplying the molecule's fields by a Gaussian
envelope about the atom, and the rotation-invariant field autocorrelation of
`cmfqsar/invariant.py` turns it into a descriptor that needs no alignment.

Two properties are measured, because they ask different things of the envelope.

* **Carbon shifts**, 13C spectra of nmrshiftdb2 measured in deuterated chloroform, with every
  signal assigned to an atom.  A shift is a short-range property: the atom, its bonds and the next
  shell decide most of it.
* **Carboxylic acid pKa**, literature and ChEMBL values collected by Baltruschat and Czodrowski
  (F1000Research 9:113, 2020), on molecules with exactly one carboxylic acid group.  Acidity is a
  response to losing a proton, so the sample is the **difference field** of the acid and its anion
  (`difference_field`), which is what `cmfqsar/fielddiff.py` compares in its kernels and what the
  free energy of deprotonation is a field picture of.  The environment of the plain acid is
  measured beside it, to say whether the difference is what carries the signal.

`examples/make_local_examples.py` prepares both files from the free downloads; the prepared sets
are in `examples/local/`.

How it is measured
------------------
* **The reference under rule 1 is an atom-centred count**: the type of the centre itself and the
  counts of atom types in distance shells around it, which is what a per-atom increment scheme is.
  It uses the same molecules, the same samples and the same folds, so the only difference between
  the two numbers is the descriptor.
* **The folds hold out whole molecules** (`AtomTargets.folds`).  Two carbons of one molecule, one
  in the training set and one in the test set, would let the model see the molecule it is being
  tested on.
* **The ridge is searched the same way for both descriptors**, on the same folds.  That search
  makes both numbers optimistic by the same kind of amount; what is compared is the difference.
* **The envelope width rho is reported twice**: at the module's default
  (`localenv.DEFAULT_RHO`, fixed before either set here was measured) and at the best value of the
  scan.  Only the first is a fair comparison with the reference, because the reference has no such
  parameter to tune; the second says how much the width matters.
* **The interval is bootstrapped over molecules, not atoms** (rule 4).  The atoms of one molecule
  are not independent samples, so a bootstrap sample is a sample of molecules with all of their
  atoms.

Usage
-----
    python benchmarks/run_local.py                      # both sets
    python benchmarks/run_local.py --set shifts
    python benchmarks/run_local.py --set pka --rho 0.25 0.5 1.0
    python benchmarks/run_local.py --max-molecules 60   # a quick look
    python benchmarks/run_local.py --report
"""
from __future__ import annotations

import argparse
import io
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

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.localenv import (DEFAULT_RHO, LOCAL_SPEC, count_features, count_kernel,  # noqa: E402
                              choose_lam, difference_field, environment, local_features,
                              local_kernel, paired_interval, types_of)

DATA = os.path.join(ROOT, "examples", "local")
RESULTS = os.path.join(HERE, "results", "local")
#: envelope widths tried, per set.  A shift is a short-range property, so the scan stays narrow;
#: the plan expects an acidity to be felt further away, so that scan reaches a width of 4.5 A.
RHOS = (0.25, 0.5, 1.0, 2.0)
SETS = {
    "shifts": dict(file="shifts_cdcl3.sdf", target="shift", unit="ppm",
                   what="13C shifts in chloroform (nmrshiftdb2)", rhos=RHOS),
    "pka": dict(file="carboxylic_pka.sdf", target="pka", unit="pKa units",
                what="carboxylic acid pKa (Baltruschat and Czodrowski 2020)",
                rhos=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0)),
}


def _say(text):
    print(text, flush=True)


def read_set(key, max_molecules=0):
    """The molecules, their fields and the per-atom target of one set."""
    spec = SETS[key]
    path = os.path.join(DATA, spec["file"])
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing. Run examples/make_local_examples.py first; its "
                         f"docstring says where the free downloads are.")
    ds = load_dataset(path, atom_activity=spec["target"])
    mols = list(ds.mols)
    names = list(ds.names)
    if max_molecules and len(mols) > max_molecules:
        mols, names = mols[:max_molecules], names[:max_molecules]
    fieldset = FieldSet.create()
    fields = [build_fields(m, fieldset, name=n) for m, n in zip(mols, names)]
    targets = AtomTargets.from_molecules(mols, spec["target"], names=names)
    return mols, fields, targets, fieldset


def anion_of(mol, carbon):
    """The carboxylate of an acid: the same geometry with the acid proton taken away."""
    for nb in mol.GetAtomWithIdx(int(carbon)).GetNeighbors():
        if nb.GetSymbol() != "O":
            continue
        hs = [x.GetIdx() for x in nb.GetNeighbors() if x.GetSymbol() == "H"]
        if len(hs) != 1:
            continue
        rw = Chem.RWMol(mol)
        rw.GetAtomWithIdx(nb.GetIdx()).SetFormalCharge(-1)
        rw.GetAtomWithIdx(nb.GetIdx()).SetNoImplicit(True)
        rw.RemoveAtom(hs[0])
        out = rw.GetMol()
        try:
            Chem.SanitizeMol(out)
        except Exception:
            return None
        return out
    return None


def difference_environments(mols, fields, targets, fieldset, rho):
    """The environment of the acid-minus-anion field about every measured carboxyl carbon."""
    out = []
    for r in range(len(targets)):
        k, i = int(targets.molecule[r]), int(targets.atom[r])
        anion = anion_of(mols[k], i)
        if anion is None:
            out.append(environment(fields[k], i, rho))
            continue
        b = build_fields(anion, fieldset, name=f"{fields[k].name}-anion")
        out.append(environment(difference_field(fields[k], b), i, rho))
    return out


def plain_environments(fields, targets, rho):
    return [environment(fields[int(targets.molecule[r])], int(targets.atom[r]), rho)
            for r in range(len(targets))]


def alike(K):
    """The mean kernel value between two different samples, a number in [-1, 1].

    A cosine-normalised kernel of 1 between two samples says the descriptor cannot tell them
    apart.  When that mean sits at almost 1 the model has nothing to work with, whatever the
    regression does afterwards, so this one number explains a result that is no better than the
    mean of the training set.
    """
    n = len(K)
    if n < 2:
        return float("nan")
    return float((K.sum() - np.trace(K)) / (n * (n - 1)))


def one_row(name, envs, targets, folds, seed):
    """Cross-validate one kind of environment."""
    t0 = time.time()
    X = local_features(envs)
    K = local_kernel(X)
    r = choose_lam(K, targets, folds=folds, seed=seed, name=name)
    r.note = f"{X.shape[1]} channels, {time.time() - t0:.0f} s"
    return r, alike(K)


def run_set(key, rhos=None, folds=10, seed=0, max_molecules=0, results=RESULTS):
    spec = SETS[key]
    rhos = list(rhos) if rhos else list(spec["rhos"])
    _say(f"=== {spec['what']}")
    mols, fields, targets, fieldset = read_set(key, max_molecules)
    _say(f"{len(targets)} measured atoms in {targets.n_measured_molecules} molecules")

    types = types_of(fields, min_atoms=10)
    Kc = count_kernel(count_features(fields, targets, types))
    ref = choose_lam(Kc, targets, folds=folds, seed=seed, name="atom-centred counts")
    ref_alike = alike(Kc)
    ref.note = f"{len(types)} atom types in 6 shells"
    _say("  " + ref.summary())

    rows, kinds = [], ["local field"]
    if key == "pka":
        kinds.append("local difference field")
    for rho in rhos:
        for label in kinds:
            if label == "local difference field":
                envs = difference_environments(mols, fields, targets, fieldset, rho)
            else:
                envs = plain_environments(fields, targets, rho)
            r, how_alike = one_row(f"{label}, rho {rho:g}", envs, targets, folds, seed)
            diff, lo, hi = paired_interval(r, ref, targets, runs=2000, seed=seed)
            rows.append(dict(kind=label, rho=float(rho), q2=r.q2, error=r.error, mae=r.mae,
                             lam=r.lam, note=r.note, difference=diff, low=lo, high=hi,
                             alike=how_alike,
                             atoms_per_environment=float(np.mean([e.n_atoms for e in envs]))))
            _say("  " + r.summary() + f"; against the counts {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    out = dict(set=key, what=spec["what"], unit=spec["unit"], target=spec["target"],
               n_samples=len(targets), n_molecules=targets.n_measured_molecules, folds=folds,
               seed=seed, fixed_rho=DEFAULT_RHO, spec=dict(d_max=LOCAL_SPEC.d_max,
                                                           step=LOCAL_SPEC.step,
                                                           sigma=LOCAL_SPEC.sigma),
               reference=dict(name=ref.name, q2=ref.q2, error=ref.error, mae=ref.mae,
                              lam=ref.lam, note=ref.note, alike=ref_alike),
               rows=rows,
               y=dict(low=float(np.min(targets.y)), high=float(np.max(targets.y)),
                      mean=float(np.mean(targets.y)),
                      spread=float(np.std(targets.y))))
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, f"{key}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def _verdict(data):
    """Does the local kernel beat the count reference by more than the interval?"""
    fixed = [r for r in data["rows"] if abs(r["rho"] - data["fixed_rho"]) < 1e-9]
    best = max(data["rows"], key=lambda r: r["q2"])
    pairs = [("at the module's default width", r) for r in fixed]
    if best not in fixed:
        pairs.append(("at the best width of the scan", best))
    out = []
    for label, r in pairs:
        if not np.isfinite(r["low"]):
            word = "cannot be judged: too few molecules for an interval"
        elif r["low"] > 0:
            word = "beats the counts"
        elif r["high"] < 0:
            word = "loses to the counts"
        else:
            word = "is no better than the counts"
        out.append(f"{label} (rho {r['rho']:g}, {r['kind']}) it {word}: "
                   f"{r['difference']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]")
    return out


def report(results=RESULTS):
    out = ["# The field around an atom against a count around an atom (plan item LP2)", "",
           "The sample is an atom in a molecule. The local descriptor is the molecule's fields "
           "under a Gaussian envelope about that atom, turned into the rotation-invariant "
           "autocorrelation, so no alignment is needed. The reference of rule 1 is an "
           "atom-centred count: the type of the centre and the atom types in distance shells "
           "about it, which is what a per-atom increment scheme is. The folds hold out whole "
           "molecules, and the interval is bootstrapped over molecules.", ""]
    for key in ("shifts", "pka"):
        path = os.path.join(results, f"{key}.json")
        if not os.path.exists(path):
            continue
        with io.open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        ref = d["reference"]
        out += [f"## {d['what']}", "",
                f"{d['n_samples']} measured atoms in {d['n_molecules']} molecules; the target "
                f"runs from {d['y']['low']:.2f} to {d['y']['high']:.2f} {d['unit']} with a "
                f"standard deviation of {d['y']['spread']:.2f}. {d['folds']} folds holding out "
                f"whole molecules.", "",
                "| descriptor | envelope width rho | q2 | error | mean absolute error | ridge | "
                "atoms per sample | mean kernel between two samples | difference in q2 against "
                "the counts |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| {ref['name']} ({ref['note']}) | -- | {ref['q2']:+.3f} | {ref['error']:.3f} | "
                f"{ref['mae']:.3f} | {ref['lam']:g} | -- | {ref.get('alike', float('nan')):.3f} | "
                f"-- |"]
        for r in d["rows"]:
            out.append(f"| {r['kind']} | {r['rho']:g} | {r['q2']:+.3f} | {r['error']:.3f} | "
                       f"{r['mae']:.3f} | {r['lam']:g} | {r['atoms_per_environment']:.1f} | "
                       f"{r.get('alike', float('nan')):.3f} | "
                       f"{r['difference']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}] |")
        out += ["", f"The error is in {d['unit']}. A mean kernel close to 1 says the descriptor "
                    f"cannot tell two samples apart, whatever the regression does with it.", ""]
        for line in _verdict(d):
            out.append(f"* {line}")
        out.append("")
    return "\n".join(out)


def write_table(results=RESULTS) -> str:
    """Rebuild ``table.md`` from the result files and return its text."""
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="which", nargs="*", default=["shifts", "pka"],
                    choices=sorted(SETS))
    ap.add_argument("--rho", nargs="*", type=float, default=None,
                    help="envelope width parameters to try, in A^-2; the default is the list the "
                         "set names")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-molecules", type=int, default=0,
                    help="use only the first n molecules of each set")
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    os.makedirs(RESULTS, exist_ok=True)
    for key in a.which:
        run_set(key, rhos=a.rho, folds=a.folds, seed=a.seed, max_molecules=a.max_molecules)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
