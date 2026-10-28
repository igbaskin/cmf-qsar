#!/usr/bin/env python
"""
Data fusion against the one-class descriptions, on the same sets and the same folds.

The program's answer to "several actives and no reliable inactives" has been a one-class
description of the active class: a ball, a centroid, a nearest neighbour, a Gaussian process or a
principal subspace (``cmfqsar/oneclass.py``).  The screening literature's answer is data fusion:
combine the similarities to the individual actives by a fixed rule and never fit anything
(``cmfqsar/fusion.py``).  Two of the rules are descriptions the program already had -- SUM is the
centroid and MAX is the one-nearest-neighbour -- so the comparison is not between two separate
families of methods but between thirteen methods in one family, and the question is only which
combination of a kernel row wins.

The protocol is the one the README's one-class table uses, so that the new rows can be read beside
the old ones:

    actives       the molecules of a Sutherland set whose activity is above its median
    negatives     the inactives of the *same* set, which are already in the same frame and are
                  the legitimate negatives here; a background drawn from another target separates on
                  size and shape before any field model is involved
    folds         five, over the actives, so no active is ever scored by a description that saw it
    fields        the five physico-chemical ones, charges from the file, cosine-normalised per
                  field and averaged, which is the normalisation a one-class model needs
    reported      AUC of the held-out actives against all the inactives, averaged over the folds

Both representations are run, because the rules were established on fingerprints and the claim
worth testing is whether they transfer to an overlap integral of molecular fields:

    cmf     the CMF kernel at the attenuation given
    ecfp4   the Tanimoto coefficient of 2048-bit ECFP4, which is itself a kernel

Nothing is tuned.  Defaults are nu = 0.1, k = 5, five components, threshold 0 and the usual RRF
constant of 60; the sweep varies the attenuation and reports the median over it as well as the
best, because a one-class or fusion score has no cross-validated criterion to tune against and
the median is what a screening run should expect.

Usage
-----
    python benchmarks/run_fusion.py                      # every set, both representations
    python benchmarks/run_fusion.py --sets THR,COX2
    python benchmarks/run_fusion.py --report
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

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.descriptors import fingerprint_kernel  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.fusion import FUSION_RULES, fuse  # noqa: E402
from cmfqsar.kernels import kernel_matrices, normalize_kernels, self_kernels  # noqa: E402
from cmfqsar.oneclass import make as make_oneclass  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "fusion")
SUTHERLAND = os.path.join(ROOT, "examples", "sutherland")

#: the sets of the README's one-class table
SETS = ("COX2", "DHFR", "BZR", "AChE", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.3, 0.5, 1.0)

#: one-class descriptions, as the README's table names them
ONECLASS = [("centroid", "ocmean", {}), ("k-NN", "ocknn", {"k": 5}),
            ("GP", "ocgp", {"lam": 0.1}),
            ("GP/sd", "ocgp", {"lam": 0.1, "score": "mean_over_sd"}),
            ("kernel PCA", "ockpca", {"components": 5})]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC with the convention of the rest of the program: y > 0 is the active class."""
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(set((y > 0).tolist())) < 2:
        return float("nan")
    return float(roc_auc_score((y > 0).astype(int), s))


def cmf_similarity(mols, names, alpha: float) -> np.ndarray:
    """The cosine-normalised CMF kernel, averaged over the fields: similarity in [-1, 1]."""
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="file")
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    K = kernel_matrices(table, table, fs.specs)
    d = self_kernels(table, fs.specs)
    Kn = normalize_kernels(K, d, d)
    return np.asarray(Kn, float).mean(axis=0)


def ecfp4_similarity(mols) -> np.ndarray:
    """The Tanimoto coefficient of 2048-bit ECFP4, which is a positive semi-definite kernel.

    The kernel itself lives in the library (:func:`cmfqsar.descriptors.fingerprint_kernel`),
    since plan item F6 pairs it with the field kernel; this is that function under the name the
    result files were produced with.
    """
    return fingerprint_kernel(mols, radius=2, bits=2048)


def folds_of(n: int, k: int = 5, seed: int = 0):
    """`k` index arrays covering range(n), shuffled once with a fixed seed."""
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]


#: quantiles of a set's own similarity distribution, used as thresholds for ANZ and MNZ
THRESHOLD_QUANTILES = (0.25, 0.5, 0.75, 0.9)

#: rules with no parameter of their own
PLAIN_RULES = tuple(r for r in FUSION_RULES if r not in ("anz", "mnz"))


def thresholds_of(S: np.ndarray) -> dict:
    """What counts as a match, for the two rules that count matches.

    ANZ and MNZ divide and multiply the total by the number of actives a molecule matches, which
    presupposes that some of them it does not match.  That holds for a fingerprint Tanimoto,
    where two molecules sharing no substructure score zero, and fails completely for an overlap
    integral of molecular fields: on these sets no normalised field similarity is below 0.6, so
    at the usual threshold of zero every reference counts for every molecule, the count is a
    constant, and both rules reduce exactly to SUM.  The threshold therefore has to come
    from the similarity distribution rather than from the literature's default, and these are
    quantiles of the set's own off-diagonal similarities.
    """
    off = np.asarray(S, float)
    mask = ~np.eye(off.shape[0], dtype=bool)
    vals = off[mask]
    vals = vals[np.isfinite(vals)]
    out = {"0": 0.0}
    if vals.size:
        for q in THRESHOLD_QUANTILES:
            out[f"q{int(q * 100)}"] = float(np.quantile(vals, q))
    return out


def score_set(S: np.ndarray, actives: np.ndarray, inactives: np.ndarray, folds) -> dict:
    """AUC per method, averaged over the folds, on one similarity matrix."""
    out: dict = {}
    per_fold: dict = {}
    thr = thresholds_of(S)
    for held in folds:
        held = np.asarray(held, int)
        if len(held) == 0:
            continue
        refs = np.array([i for i in actives if i not in set(actives[held].tolist())], int)
        if len(refs) < 2:
            continue
        library = np.concatenate([actives[held], inactives])
        y = np.concatenate([np.ones(len(held)), -np.ones(len(inactives))])
        rows = S[np.ix_(library, refs)]
        for rule in PLAIN_RULES:
            per_fold.setdefault(f"fusion {rule}", []).append(auc(y, fuse(rows, rule)))
        for rule in ("anz", "mnz"):
            for tag, t in thr.items():
                per_fold.setdefault(f"fusion {rule}@{tag}", []).append(
                    auc(y, fuse(rows, rule, threshold=t)))
        Kref = S[np.ix_(refs, refs)]
        diag = np.ones(len(library))
        for label, kind, kw in ONECLASS:
            try:
                sc = make_oneclass(kind, nu=0.1, **kw).fit(Kref)
                per_fold.setdefault(f"oneclass {label}", []).append(
                    auc(y, sc.decision(rows, diag)))
            except Exception:                           # noqa: BLE001 - a degenerate fold
                per_fold.setdefault(f"oneclass {label}", []).append(float("nan"))
    for key, vals in per_fold.items():
        v = np.asarray(vals, float)
        out[key] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out


def run(sets=SETS, alphas=ALPHAS, out_dir=RESULTS):
    os.makedirs(out_dir, exist_ok=True)
    for name in sets:
        path = os.path.join(SUTHERLAND, f"{name}.sdf")
        if not os.path.exists(path):
            print(f"  {name}: no such file, skipped")
            continue
        t0 = time.time()
        ds = load_dataset(path)
        y = np.asarray(ds.activity, float)
        keep = np.flatnonzero(np.isfinite(y))
        mols = [ds.mols[i] for i in keep]
        names = [ds.names[i] for i in keep]
        y = y[keep]
        med = float(np.median(y))
        actives = np.flatnonzero(y > med)
        inactives = np.flatnonzero(y <= med)
        folds = folds_of(len(actives), 5, seed=0)
        record = {"set": name, "n": int(len(y)), "n_actives": int(len(actives)),
                  "n_inactives": int(len(inactives)), "median_activity": med,
                  "alphas": list(alphas), "representations": {}}
        print(f"{name}: {len(y)} molecules, {len(actives)} actives above {med:.2f}")

        ecfp = ecfp4_similarity(mols)
        record["representations"]["ecfp4"] = {
            "default": score_set(ecfp, actives, inactives, folds)}

        per_alpha = {}
        for a in alphas:
            S = cmf_similarity(mols, names, a)
            per_alpha[f"{a:g}"] = score_set(S, actives, inactives, folds)
            print(f"  alpha {a:g}: done")
        record["representations"]["cmf"] = per_alpha
        with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
            json.dump(record, fh, indent=1)
        print(f"  {name} written, {time.time() - t0:.0f} s")


def _load(out_dir=RESULTS):
    rows = {}
    for name in SETS:
        p = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(p):
            rows[name] = json.load(open(p))
    return rows


def report(path=None, out_dir=RESULTS):
    """results/fusion/table.md: the thirteen methods, on both representations."""
    data = _load(out_dir)
    if not data:
        print("no results yet; run the benchmark first")
        return
    sets = [s for s in SETS if s in data]
    # display name -> the key or key prefix its settings are stored under
    methods = [(f"fusion {r}", f"fusion {r}") for r in PLAIN_RULES] + \
              [("fusion anz (threshold swept)", "fusion anz@"),
               ("fusion mnz (threshold swept)", "fusion mnz@")] + \
              [(f"oneclass {lab}", f"oneclass {lab}") for lab, _, _ in ONECLASS]

    def values(entry: dict, key: str):
        """Every setting's AUC for one display method: an exact key, or a family by prefix."""
        if key.endswith("@"):
            return [v for k, v in entry.items() if k.startswith(key)]
        return [entry[key]] if key in entry else []
    lines = ["# Data fusion against the one-class descriptions", "",
             "AUC of the held-out actives against the inactives of the same set, five folds over",
             "the actives, averaged. Nothing tuned. The CMF columns are the attenuation that does",
             "best on that set and the median over the three attenuations tried; a screening run",
             "has no criterion to pick the attenuation with, so the median is what it should",
             "expect.", ""]
    head = "| method | " + " | ".join(sets) + " | mean | median over alpha |"
    lines += [head, "|---|" + "---:|" * (len(sets) + 2)]
    for label, key in methods:
        best, meds = [], []
        for s in sets:
            per = data[s]["representations"]["cmf"]
            vals = [x for a in per for x in values(per[a], key)]
            v = np.asarray(vals, float)
            best.append(float(np.nanmax(v)) if np.isfinite(v).any() else float("nan"))
            meds.append(float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan"))
        lines.append("| " + label + " | " + " | ".join(f"{v:.3f}" for v in best)
                     + f" | **{np.nanmean(best):.3f}** | {np.nanmean(meds):.3f} |")
    lines += ["", "## The same rules on ECFP4", "",
              "The fusion rules are functions of a similarity row, so they run unchanged on a",
              "fingerprint Tanimoto; only the meaning of \"similar\" changes.", ""]
    lines += ["| method | " + " | ".join(sets) + " | mean |",
              "|---|" + "---:|" * (len(sets) + 1)]
    for label, key in methods:
        vals = []
        for s in sets:
            v = np.asarray(values(data[s]["representations"]["ecfp4"]["default"], key), float)
            vals.append(float(np.nanmax(v)) if np.isfinite(v).any() else float("nan"))
        v = np.asarray(vals, float)
        lines.append("| " + label + " | " + " | ".join(f"{x:.3f}" for x in v)
                     + f" | **{np.nanmean(v):.3f}** |")
    lines += ["", "## Sets", "",
              "| set | molecules | actives | inactives | median activity |",
              "|---|---:|---:|---:|---:|"]
    for s in sets:
        d = data[s]
        lines.append(f"| {s} | {d['n']} | {d['n_actives']} | {d['n_inactives']} | "
                     f"{d['median_activity']:.2f} |")
    text = "\n".join(lines) + "\n"
    path = path or os.path.join(out_dir, "table.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--alphas", default=",".join(f"{a:g}" for a in ALPHAS))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    run(sets=tuple(s for s in args.sets.split(",") if s),
        alphas=tuple(float(a) for a in args.alphas.split(",") if a))
    report()


if __name__ == "__main__":
    main()
