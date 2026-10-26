#!/usr/bin/env python
"""
Does an alignment-free model know which group of a guest enters the cavity? (plan item SC6)

The rotation-invariant field autocorrelation is the best kernel on six of the seven cyclodextrin
guest tables (`run_cyclodextrin.py`), and until now it was the one model in the program that could
say nothing about *where* on a molecule binding is gained: it has no frame, so it has no co-field.
It does have a decomposition, though.  Every channel of the autocorrelation is a sum over atom
pairs, and the prediction is a fixed linear functional of those channels, so a prediction splits
over the atoms of a molecule exactly (`cmfqsar/invariant.py`, `CMFModel.atom_contributions`).

This benchmark asks whether that split says anything a chemist would recognise.  Binding to
beta-cyclodextrin is driven by a hydrophobic group entering the cavity; adamantyl is the strongest
such group known, and phenyl and cyclohexyl are the common ones.  Polar groups stay in the water.
So the question is whether the atoms of those groups take a larger share of the prediction than
the rest of the molecule, and whether the polar groups take a smaller one.

How it is measured
------------------
* The model is the invariant kernel with the five fields and the ridge the training
  cross-validation chooses, as in `run_cyclodextrin.py`.
* **The contributions are out of fold.** For each of the ten folds the model is fitted on the
  other nine and used to split the predictions of the held-out guests, so a guest's own measured
  constant never enters its own explanation.
* Hydrogens are reported on the heavy atom they are bonded to, so a number is per heavy atom.
* For every guest carrying a group, the statistic is the mean contribution per heavy atom inside
  the group minus the mean per heavy atom outside it.
* The null is a permutation: the per-atom contributions are shuffled among the heavy atoms of the
  same molecule, which keeps the molecule and the model and destroys only which atom is where.
  A z-score against that null says whether the group stands out among the atoms of its own guest.

Usage
-----
    python benchmarks/run_guest_atoms.py                 # beta, alpha and gamma
    python benchmarks/run_guest_atoms.py --hosts beta
    python benchmarks/run_guest_atoms.py --report
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

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, _kfold_indices, cv_predict, r2_score  # noqa: E402

DATA = os.path.join(ROOT, "examples", "cyclodextrin")
RESULTS = os.path.join(HERE, "results", "guest_atoms")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
FOLDS = 10
SEED = 0
#: how many shuffles the permutation null uses per guest
SHUFFLES = 200
HOSTS = {"beta": "beta_cyclodextrin.csv", "alpha": "alpha_cyclodextrin.csv",
         "gamma": "gamma_cyclodextrin.csv"}

#: the groups the question is about.  The first three are the hydrophobic groups that enter a
#: cyclodextrin cavity, in the order of the affinity they are known to give; the last three are
#: polar groups that stay in the water and are the control.
GROUPS = (
    ("adamantyl", "C12(CC3CC(C1)CC(C3)C2)", "cavity"),
    ("phenyl", "c1ccccc1", "cavity"),
    ("cyclohexyl", "C1CCCCC1", "cavity"),
    ("carboxyl", "[CX3](=O)[OX2H1,OX1-]", "water"),
    ("hydroxyl", "[OX2H1][CX4]", "water"),
    ("amine", "[NX4+,NX3;H2,H1;!$(N-C=[O,N,S])]", "water"),
)


def patterns():
    out = []
    for name, smarts, kind in GROUPS:
        q = Chem.MolFromSmarts(smarts)
        if q is None:
            raise SystemExit(f"the SMARTS of {name} does not parse: {smarts}")
        out.append((name, q, kind))
    return out


def heavy_shares(mol: Chem.Mol, contrib: np.ndarray) -> np.ndarray:
    """Per-atom contributions reported on the heavy atoms: a hydrogen goes on its neighbour."""
    out = np.array(contrib, float)
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 1:
            continue
        nbrs = [n.GetIdx() for n in a.GetNeighbors()]
        if nbrs:
            out[nbrs[0]] += out[a.GetIdx()]
        out[a.GetIdx()] = 0.0
    return out


def group_statistic(mol: Chem.Mol, shares: np.ndarray, query: Chem.Mol, rng) -> dict | None:
    """Mean share per heavy atom inside the group minus outside, and its z against a shuffle."""
    match = mol.GetSubstructMatch(query)
    if not match:
        return None
    heavy = np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1], int)
    inside = np.array(sorted(set(match) & set(heavy.tolist())), int)
    outside = np.array([i for i in heavy if i not in set(inside.tolist())], int)
    if len(inside) == 0 or len(outside) == 0:
        return None
    v = shares[heavy]
    k = len(inside)
    real = float(shares[inside].mean() - shares[outside].mean())
    null = np.empty(SHUFFLES)
    for s in range(SHUFFLES):
        perm = rng.permutation(len(v))
        a = v[perm[:k]].mean()
        b = v[perm[k:]].mean()
        null[s] = a - b
    sd = float(null.std())
    return {"difference": real, "z": float((real - null.mean()) / sd) if sd > 1e-12 else 0.0,
            "inside": int(k), "outside": int(len(outside)),
            "mean_inside": float(shares[inside].mean()),
            "mean_outside": float(shares[outside].mean())}


def out_of_fold_contributions(mf, mols, y, lam):
    """Split every guest's prediction with a model that never saw that guest."""
    n = len(y)
    table = AtomTable.from_molecules(mf)
    fs = mf[0]
    out = [None] * n
    for te in _kfold_indices(n, FOLDS, SEED):
        tr = np.setdiff1d(np.arange(n), te)
        sub = AtomTable.from_molecules([mf[i] for i in tr])
        model = CMFModel(fieldset=FIELDSET, method="krr", normalize=True,
                         params={"lam": float(lam)}, kernel_source="autocorrelation")
        model.fit(sub, y[tr], cv_folds=0)
        for i in te:
            out[i] = model.atom_contributions(mf[i])
    return out


FIELDSET = None            # set in run_host, so the fold models share one field set


def run_host(host, force=False, max_guests=0):
    global FIELDSET
    out_path = os.path.join(RESULTS, f"{host}.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        return json.load(open(out_path))
    t0 = time.time()
    ds = load_dataset(os.path.join(DATA, HOSTS[host]))
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y = y[ok]
    if max_guests and len(mols) > max_guests:
        keep = np.linspace(0, len(mols) - 1, max_guests).astype(int)
        mols = [mols[i] for i in keep]
        names = [names[i] for i in keep]
        y = y[keep]
    print(f"\n=== {host}: {len(mols)} guests", flush=True)
    FIELDSET = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                               param_scheme="rdkit")
    mf = [build_fields(m, FIELDSET, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=FIELDSET, method="krr", normalize=True,
                     kernel_source="autocorrelation")
    K = model.training_kernel(table)
    best = None
    for lam in LAMS:
        pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=FOLDS, seed=SEED)
        q2 = r2_score(y, pred)
        if best is None or q2 > best[0]:
            best = (q2, float(lam))
    q2, lam = best
    print(f"    invariant model: q2 {q2:+.3f} at lambda {lam:g} ({time.time() - t0:.0f} s)",
          flush=True)
    contrib = out_of_fold_contributions(mf, mols, y, lam)
    print(f"    out-of-fold contributions done ({time.time() - t0:.0f} s)", flush=True)
    rng = np.random.default_rng(SEED)
    rows = {name: [] for name, _q, _k in GROUPS}
    for i, mol in enumerate(mols):
        shares = heavy_shares(mol, contrib[i])
        for name, q, _kind in patterns():
            got = group_statistic(mol, shares, q, rng)
            if got is not None:
                got["name"] = names[i]
                got["y"] = float(y[i])
                rows[name].append(got)
    res = {"host": host, "n": len(mols), "q2": q2, "lam": lam, "folds": FOLDS,
           "fields": FIELDS, "shuffles": SHUFFLES, "groups": {}, "seconds": time.time() - t0}
    for name, _q, kind in GROUPS:
        got = rows[name]
        if not got:
            continue
        d = np.array([g["difference"] for g in got])
        z = np.array([g["z"] for g in got])
        res["groups"][name] = {
            "kind": kind, "guests": len(got),
            "difference_mean": float(d.mean()),
            "difference_sem": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
            "z_mean": float(z.mean()), "share_positive": float(np.mean(d > 0)),
            "mean_inside": float(np.mean([g["mean_inside"] for g in got])),
            "mean_outside": float(np.mean([g["mean_outside"] for g in got])),
            "atoms_inside": float(np.mean([g["inside"] for g in got])),
            "activity_mean": float(np.mean([g["y"] for g in got])),
        }
        r = res["groups"][name]
        print(f"    {name:11s} {kind:6s} {r['guests']:4d} guests  per-atom share inside "
              f"{r['mean_inside']:+.4f} outside {r['mean_outside']:+.4f}  difference "
              f"{r['difference_mean']:+.4f} +- {1.96 * r['difference_sem']:.4f}  "
              f"z {r['z_mean']:+.2f}  positive on {r['share_positive']:.0%}", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out_path} ({res['seconds']:.0f} s)", flush=True)
    return res


def report(results=RESULTS):
    rows = []
    if os.path.isdir(results):
        for fn in sorted(os.listdir(results)):
            if fn.endswith(".json"):
                rows.append(json.load(open(os.path.join(results, fn))))
    order = {"beta": 0, "alpha": 1, "gamma": 2}
    rows.sort(key=lambda d: order.get(d["host"], 9))
    out = ["## What an invariant model says about a guest", "",
           "The rotation-invariant kernel has no co-field, but every one of its channels is a sum "
           "over atom pairs, so a prediction splits over the atoms exactly. The question is "
           "whether that split names the group that enters the cyclodextrin cavity. The numbers "
           "below are per heavy atom (hydrogens reported on their neighbour), from models fitted "
           "without the guest being explained, and the difference is the mean share inside the "
           "group minus the mean share outside it.", ""]
    for d in rows:
        out += [f"### {d['host']}-cyclodextrin, {d['n']} guests "
                f"(invariant q2 {d['q2']:+.3f})", "",
                "| group | | guests | share per atom inside | outside | difference | z against a "
                "shuffle | positive on |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, _q, _k in GROUPS:
            g = d["groups"].get(name)
            if not g:
                continue
            out.append(f"| {name} | {g['kind']} | {g['guests']} | {g['mean_inside']:+.4f} | "
                       f"{g['mean_outside']:+.4f} | {g['difference_mean']:+.4f} +- "
                       f"{1.96 * g['difference_sem']:.4f} | {g['z_mean']:+.2f} | "
                       f"{g['share_positive']:.0%} |")
        out.append("")
    if rows:
        beta = next((d for d in rows if d["host"] == "beta"), rows[0])
        cavity = [n for n, _q, k in GROUPS if k == "cavity" and n in beta["groups"]]
        water = [n for n, _q, k in GROUPS if k == "water" and n in beta["groups"]]
        c = float(np.mean([beta["groups"][n]["difference_mean"] for n in cavity]))
        w = float(np.mean([beta["groups"][n]["difference_mean"] for n in water]))
        out += [f"On {beta['host']}-cyclodextrin the three groups that enter the cavity take "
                f"{c:+.4f} more of the prediction per atom than the rest of their molecule, and "
                f"the three polar groups {w:+.4f}. ", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", nargs="*", default=list(HOSTS))
    ap.add_argument("--max-guests", type=int, default=0)
    ap.add_argument("--force", action="store_true")
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
    for host in a.hosts:
        run_host(host, force=a.force, max_guests=a.max_guests)
    print()
    print(report())


if __name__ == "__main__":
    main()
