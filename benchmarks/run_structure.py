#!/usr/bin/env python
"""
Two questions about a set of molecules, asked with the kernel that models it.

**Is this set the same population as that one?**  The maximum mean discrepancy with a permutation
null.  A test is worth nothing unless it is calibrated as well as powerful, so both halves are
measured: a random half-split of one data set, where there is nothing to find and the test must
not find it, and two different targets' ligands, where there is everything to find.  Reported as
the share of splits called significant at 5 %, which should be near 5 % in the first case and
100 % in the second.

**What are this set's real groups?**  Every grouped validation here splits the molecules by their
Bemis-Murcko scaffold, on the assumption that a scaffold is what makes two molecules alike to the
model.  Spectral clustering of the kernel partitions the model's own similarity instead, and the
adjusted Rand index between the two says whether that assumption holds.  Then the question that
matters: leave-one-group-out cross-validation under each partition, beside plain 10-fold, which
says which grouping is the harder test and by how much.

Usage
-----
    python benchmarks/run_structure.py
    python benchmarks/run_structure.py --sets ACE DHFR
    python benchmarks/run_structure.py --report
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

from rdkit import RDLogger  # noqa: E402

from cmfqsar.chemotypes import chemotype_labels, grouped_cv_predict  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.mmd import mmd_test  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, r2_score  # noqa: E402
from cmfqsar.spectral import adjusted_rand, spectral_clusters  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "structure")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 0.5
LAM = 0.1
SPLITS = 20          # random half-splits, for the calibration of the test
PERMUTATIONS = 499


def kernel_of(mols, names, alpha=ALPHA):
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="file")
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    return model.joint_kernel(model.prepare_training_kernels(table)), fs


def _load(name, data=DATA):
    ds = load_dataset(os.path.join(data, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    return ([m for m, k in zip(ds.mols, keep) if k],
            [n for n, k in zip(ds.names, keep) if k], y[keep])


def run_set(name, force=False, data=DATA, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    mols, names, y = _load(name, data)
    n = len(y)
    print(f"=== {name}: {n} molecules", flush=True)
    K, _fs = kernel_of(mols, names)

    # 1. calibration: a random half of the set against the other half
    null_p = []
    for t in range(SPLITS):
        rng = np.random.RandomState(t)
        mask = np.zeros(n, bool)
        mask[rng.choice(n, n // 2, replace=False)] = True
        null_p.append(mmd_test(K, mask, permutations=PERMUTATIONS, seed=t).p_value)

    # 2. the groups. The eigengap says how many clusters the kernel needs; the comparison with
    # the scaffolds is made at the same number of groups as the scaffolds, since holding out one
    # of two clusters removes half the training set and holding out one of nine does not --
    # a difference in granularity would otherwise be read as a difference in difficulty
    scaffolds, _smiles = chemotype_labels(mols, min_size=3)
    n_groups = len(set(np.asarray(scaffolds).tolist()))
    fit_gap = spectral_clusters(K, k_max=8, seed=0)
    fit = spectral_clusters(K, k=n_groups, seed=0)
    ari = adjusted_rand(fit.labels, scaffolds)

    # 3. what each grouping costs a model
    q2_random = r2_score(y, cv_predict(K, y, "krr", {"lam": LAM}, folds=10, seed=0))
    pred_s = grouped_cv_predict(K, y, np.asarray(scaffolds), params={"lam": LAM})
    pred_f = grouped_cv_predict(K, y, fit.labels, params={"lam": LAM})
    ok_s, ok_f = np.isfinite(pred_s), np.isfinite(pred_f)

    out = {"set": name, "n": int(n),
           "null_p": [float(p) for p in null_p],
           "null_rejected": float(np.mean(np.asarray(null_p) <= 0.05)),
           "n_scaffold_groups": int(n_groups),
           "k_field": int(fit.k), "k_gap": int(fit_gap.k), "gap": float(fit_gap.gap),
           "ari": float(ari),
           "q2_random": float(q2_random),
           "q2_scaffold": float(r2_score(y[ok_s], pred_s[ok_s])) if ok_s.any() else float("nan"),
           "q2_field": float(r2_score(y[ok_f], pred_f[ok_f])) if ok_f.any() else float("nan"),
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"  ARI {ari:+.3f}, q2 random {out['q2_random']:+.3f}, by scaffold "
          f"{out['q2_scaffold']:+.3f}, by field cluster {out['q2_field']:+.3f} "
          f"({out['seconds']:.0f} s)", flush=True)
    return out


def power(results=RESULTS, data=DATA, force=False):
    """The other half of the calibration: every pair of different targets, which must be found."""
    path = os.path.join(results, "power.json")
    if os.path.exists(path) and not force:
        return json.load(open(path))
    os.makedirs(results, exist_ok=True)
    rows = []
    for i, a in enumerate(SETS):
        for b in SETS[i + 1:]:
            ma, na, _ = _load(a, data)
            mb, nb, _ = _load(b, data)
            K, _fs = kernel_of(list(ma) + list(mb), list(na) + list(nb))
            mask = np.zeros(len(ma) + len(mb), bool)
            mask[:len(ma)] = True
            r = mmd_test(K, mask, permutations=PERMUTATIONS, seed=0)
            rows.append({"a": a, "b": b, "mmd": r.mmd, "p": r.p_value, "z": r.z})
            print(f"  {a} vs {b}: MMD {r.mmd:.3f}, p {r.p_value:.4f}, z {r.z:+.0f}", flush=True)
    out = {"pairs": rows}
    json.dump(out, open(path, "w"), indent=1)
    return out


def report(results=RESULTS):
    import glob

    files = sorted(f for f in glob.glob(os.path.join(results, "*.json"))
                   if not f.endswith("power.json"))
    if not files:
        return "No structure results yet; run `python benchmarks/run_structure.py`."
    data = [json.load(open(f)) for f in files]
    pw = json.load(open(os.path.join(results, "power.json"))) \
        if os.path.exists(os.path.join(results, "power.json")) else {"pairs": []}

    lines = ["# Two questions about a set, asked with its kernel", "",
             "## Is this the same population as that? (MMD)", "",
             f"{SPLITS} random half-splits of each set, {PERMUTATIONS} permutations each. "
             "There is nothing to find in a half-split, so the share called significant at 5 % "
             "is the test's false-positive rate and should be near 5 %.", "",
             "| set | n | half-splits called significant |", "|---|---:|---:|"]
    for d in data:
        lines.append(f"| {d['set']} | {d['n']} | {d['null_rejected']:.0%} |")
    rej = float(np.mean([d["null_rejected"] for d in data]))
    lines.append(f"| **mean** |  | **{rej:.1%}** |")
    if pw["pairs"]:
        sig = float(np.mean([r["p"] <= 0.05 for r in pw["pairs"]]))
        z = float(np.median([r["z"] for r in pw["pairs"]]))
        lines += ["", f"Against that, all {len(pw['pairs'])} pairs of *different* targets: "
                  f"**{sig:.0%}** are called significant, median z = {z:+.0f}. The test is "
                  "calibrated where there is nothing and saturated where there is."]

    lines += ["", "## What are this set's real groups? (spectral clustering)", "",
              "The adjusted Rand index between the scaffold groups and the clusters of the "
              "kernel, both cut into the same number of groups, and what holding out a whole "
              "group of each kind costs a model. The eigengap column is how many clusters the "
              "kernel would have chosen for itself.", "",
              "| set | groups | k by eigengap | ARI | q² random | q² by scaffold | "
              "q² by field cluster |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in data:
        lines.append(f"| {d['set']} | {d['n_scaffold_groups']} | {d.get('k_gap', d['k_field'])} "
                     f"| {d['ari']:+.3f} "
                     f"| {d['q2_random']:+.3f} | {d['q2_scaffold']:+.3f} | {d['q2_field']:+.3f} |")
    lines.append("| **mean** |  |  | " + " | ".join(
        f"**{np.nanmean([d[k] for d in data]):+.3f}**"
        for k in ("ari", "q2_random", "q2_scaffold", "q2_field")) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-power", action="store_true", help="skip the pairs of different targets")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    for name in a.sets:
        run_set(name, force=a.force)
    if not a.no_power:
        print("\nDifferent targets, which the test must find:")
        power(force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
