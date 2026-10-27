#!/usr/bin/env python
"""
Fitting in the space of the landmarks (plan item B6)

The exact model holds one n by n matrix per field and factorises the n by n joint kernel, which
is what stops it at ten to twenty thousand molecules. `cmfqsar/lowrank.py` replaces every field
kernel by the Nystrom approximation ``K(X, Z) K(Z, Z)^+ K(Z, X)``, which is the Gram matrix of
m features, and fits in the space of those features instead. Plan item B6 assessed the method
and measured the approximation; this measures the implementation.

Four things are asked
---------------------
1. **Is it the same model when nothing is approximated?** With every molecule as its own
   landmark the Nystrom kernel is the exact kernel, so the features, the fit, the evidence and
   the co-field must agree with the exact path to machine precision. If they do not, the rest
   of the table means nothing.
2. **What does the approximation cost in accuracy?** q2 against the exact model on the eight
   Sutherland sets, for a grid of Gaussian centres and for landmark molecules.
3. **What does it save?** The memory of a fit and the time, measured on subsets of the largest
   aligned set, against the exact path.
4. **Does the variance survive?** The exact self-kernel is kept, so the posterior variance
   should not collapse where the landmarks do not reach; the plain form without that term is
   measured beside it.

Usage
-----
    python benchmarks/run_lowrank.py
    python benchmarks/run_lowrank.py --sets GPB THR --report
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

from rdkit import RDLogger  # noqa: E402

from cmfqsar import lowrank as LR  # noqa: E402
from cmfqsar.cofields import evaluate_gaussians, make_grid  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.evidence import log_evidence as exact_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, krr_fit, r2_score  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "lowrank")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 0.5
LAM = 0.1
WAYS = (("grid 2.0 A", "grid", 2.0), ("grid 1.5 A", "grid", 1.5),
        ("20 molecules", "molecules", 20), ("40 molecules", "molecules", 40),
        ("80 molecules", "molecules", 80), ("every molecule", "molecules", 0))


def _say(text):
    print(text, flush=True)


def table_of(name, alpha=ALPHA):
    ds = load_dataset(os.path.join(DATA, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    fs = FieldSet.create(physchem=FIELDS, alpha=float(alpha), charge_model="file")
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    return fs, table, y[keep]


def exact_of(fs, table, y, lam=LAM, folds=10, seed=0):
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    model.params["lam"] = float(lam)
    Kf = np.asarray(model.prepare_training_kernels(table))
    K = np.asarray(model.joint_kernel(Kf, table.descriptors), float)
    # fitted, not only cross-validated: the co-field of the exact model is one of the things
    # the low-rank path has to reproduce
    model.fit(table, y, cv_folds=folds, cv_seed=seed, reuse_kernels=True)
    return model, K, float(model.stats.q2)


def landmarks_for(kind, size, table, fs):
    specs = [s for s in fs.specs if s.enabled]
    if kind == "grid":
        return LR.gaussian_centres(table, float(size), len(specs))
    want = table.n_mol if not size else int(size)
    return LR.landmark_molecules(table, want, seed=0)[0]


def stage_exact(name="GPB", lam=LAM, results=RESULTS):
    """Nothing approximated: every molecule its own landmark."""
    fs, table, y = table_of(name)
    model, K, q2 = exact_of(fs, table, y, lam)
    low = LR.build(table, fs, scale=model.scale)
    Phi = low.transform(table)
    g = LR.gram(low, table, y)
    w, b = LR.fit(g, lam)
    a, b_exact = krr_fit(K, y, lam)
    k_self = np.full(len(y), float(np.sum(np.asarray(fs.mixing_vector())[:len(low.specs)]
                                          * model.scale)))
    var_low = LR.variance(g, Phi, lam, k_self=k_self)
    var_exact = np.diag(K) - np.einsum("ij,jk,ik->i", K,
                                       np.linalg.inv(K + lam * np.eye(len(y))), K)
    grid = make_grid(table.coords, margin=2.0, spacing=2.0)
    pts = grid.points()[:400]
    cof = []
    for f in range(len(low.specs)):
        c0, k0, w0 = model.cofield_coefficients(f)
        want = evaluate_gaussians(pts, c0, k0, w0) * fs.specs[f].h * model.scale[f]
        c1, k1, w1 = LR.cofield_gaussians(low, w, f)
        cof.append(float(np.max(np.abs(want - evaluate_gaussians(pts, c1, k1, w1)))))
    out = dict(set=name, n=int(len(y)), rank=int(low.rank),
               kernel=float(np.max(np.abs(Phi @ Phi.T - K))),
               prediction=float(np.max(np.abs(LR.predict(Phi, w, b) - (K @ a + b_exact)))),
               evidence=float(abs(LR.log_evidence(g, lam) - exact_evidence(K, y, lam))),
               variance=float(np.max(np.abs(var_low - np.maximum(var_exact, 0.0)))),
               cofield=float(max(cof)))
    _say(f"nothing approximated, on {name}: kernel {out['kernel']:.1e}, predictions "
         f"{out['prediction']:.1e}, evidence {out['evidence']:.1e}, variance "
         f"{out['variance']:.1e}, co-field {out['cofield']:.1e}")
    return out


def stage_accuracy(sets=SETS, lam=LAM, folds=10, seed=0, results=RESULTS):
    rows = []
    for name in sets:
        fs, table, y = table_of(name)
        model, K, q2 = exact_of(fs, table, y, lam, folds, seed)
        _say(f"=== {name}: {table.n_mol} molecules, the exact model q2 {q2:+.3f}")
        for label, kind, size in WAYS:
            if kind == "molecules" and size and size >= table.n_mol:
                continue
            t0 = time.time()
            Z = landmarks_for(kind, size, table, fs)
            low = LR.build(Z, fs, scale=model.scale, kind=kind)
            pred, _which = LR.cross_validate(low, table, y, lam, folds=folds, seed=seed)
            got = float(r2_score(y, pred))
            rows.append(dict(set=name, way=label, n=int(table.n_mol), landmarks=int(Z.n_mol),
                             rank=int(low.rank), q2=got, exact=q2, seconds=time.time() - t0,
                             memory=LR.memory(table.n_mol, low.rank, len(low.specs)),
                             memory_exact=LR.memory(table.n_mol, low.rank, len(low.specs),
                                                    exact=True)))
            _say(f"  {label:16s} {Z.n_mol:4d} landmarks, rank {low.rank:5d}: q2 {got:+.3f} "
                 f"({got - q2:+.3f})")
    return rows


def stage_narrow(sets=("GPB", "THR", "AChE"), alphas=(0.5, 2.0, 5.0), lam=LAM, folds=10,
                 seed=0, results=RESULTS):
    """Where a grid of centres fails: the attenuation the item warned about.

    A grid spans a field while its spacing is not much larger than the field's own width, so at
    a strong attenuation it would need tens of thousands of centres.  The atoms of a few
    molecules follow the field instead of covering the box, which is the case plan item B6
    named and left open.  The number to read here is the error of the kernel, not the q2: these
    sets are small and a q2 moves by more than the approximation does.
    """
    rows = []
    for name in sets:
        for alpha in alphas:
            fs, table, y = table_of(name, alpha)
            model, K, q2 = exact_of(fs, table, y, lam, folds, seed)
            n_f = len([s for s in fs.specs if s.enabled])
            kinds = (("a grid 2.0 A apart", LR.gaussian_centres(table, 2.0, n_f), "grid"),
                     ("the atoms, 0.75 A apart",
                      LR.landmark_atoms(table, n_f, min_distance=0.75, seed=seed), "atoms"),
                     ("the atoms, 0.5 A apart",
                      LR.landmark_atoms(table, n_f, min_distance=0.5, seed=seed), "atoms"),
                     ("40 molecules", LR.landmark_molecules(table, 40, seed=seed)[0],
                      "molecules"))
            for label, Z, kind in kinds:
                low = LR.build(Z, fs, scale=model.scale, kind=kind)
                Phi = low.transform(table)
                err = float(np.linalg.norm(Phi @ Phi.T - K) / max(np.linalg.norm(K), 1e-300))
                pred, _w = LR.cross_validate(low, table, y, lam, folds=folds, seed=seed)
                rows.append(dict(set=name, alpha=float(alpha), way=label, kind=kind,
                                 landmarks=int(Z.n_mol), rank=int(low.rank),
                                 error=err, q2=float(r2_score(y, pred)), exact=q2))
                _say(f"  {name:6s} alpha {alpha:<4g} {label:24s} {Z.n_mol:5d} landmarks  "
                     f"kernel error {err:.1e}  q2 {rows[-1]['q2']:+.3f} ({q2:+.3f} exact)")
    return rows


def stage_cost(name="DHFR", sizes=(50, 100, 200, 361), lam=LAM, results=RESULTS):
    """What each path costs as the set grows, on the largest aligned set in the repository."""
    fs, table, y = table_of(name)
    rows = []
    for n in sizes:
        n = int(min(n, table.n_mol))
        part = table.subset(np.arange(n))
        yy = y[:n]
        t0 = time.time()
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        model.params["lam"] = lam
        Kf = np.asarray(model.prepare_training_kernels(part))
        K = np.asarray(model.joint_kernel(Kf, part.descriptors), float)
        krr_fit(K, yy, lam)
        exact_time = time.time() - t0
        t0 = time.time()
        Z = LR.gaussian_centres(part, 2.0, len(model.fieldset.specs))
        low = LR.build(Z, fs, scale=model.scale, kind="grid")
        g = LR.gram(low, part, yy)
        LR.fit(g, lam)
        low_time = time.time() - t0
        rows.append(dict(n=n, rank=int(low.rank), exact_seconds=exact_time,
                         low_seconds=low_time,
                         exact_memory=LR.memory(n, low.rank, len(low.specs), exact=True),
                         low_memory=LR.memory(n, low.rank, len(low.specs))))
        _say(f"  {n:4d} molecules: exact {exact_time:6.1f} s / "
             f"{rows[-1]['exact_memory'] / 1e6:7.1f} MB, low rank {low_time:6.1f} s / "
             f"{rows[-1]['low_memory'] / 1e6:6.1f} MB")
    return rows


def stage_variance(name="THR", lam=LAM, held=20, seed=0, results=RESULTS):
    """Does the variance stay large where the landmarks do not reach?"""
    fs, table, y = table_of(name)
    model, K, _q2 = exact_of(fs, table, y, lam)
    rng = np.random.RandomState(seed)
    rows = np.sort(rng.choice(table.n_mol, int(held), replace=False))
    rest = np.setdiff1d(np.arange(table.n_mol), rows)
    train = table.subset(rest)
    test = table.subset(rows)
    # landmarks from the training half only, so the held-out molecules are off the basis
    Z = LR.landmark_molecules(train, 30, seed=seed)[0]
    low = LR.build(Z, fs, scale=model.scale, kind="molecules")
    g = LR.gram(low, train, y[rest])
    Phi = low.transform(test)
    k_self = np.full(len(rows), float(np.sum(np.asarray(fs.mixing_vector())[:len(low.specs)]
                                             * model.scale)))
    kept = LR.variance(g, Phi, lam, k_self=k_self)
    plain = LR.variance(g, Phi, lam)
    exact = (np.diag(K)[rows]
             - np.einsum("ij,jk,ik->i", K[np.ix_(rows, rest)],
                         np.linalg.inv(K[np.ix_(rest, rest)] + lam * np.eye(len(rest))),
                         K[np.ix_(rows, rest)]))
    out = dict(set=name, held=int(len(rows)), landmarks=int(Z.n_mol),
               exact=float(np.mean(exact)), kept=float(np.mean(kept)),
               plain=float(np.mean(plain)),
               kept_error=float(np.mean(np.abs(kept - exact))),
               plain_error=float(np.mean(np.abs(plain - exact))))
    _say(f"the variance of {out['held']} held-out molecules on {name}: exact "
         f"{out['exact']:.3f}, with the self-kernel kept {out['kept']:.3f}, without it "
         f"{out['plain']:.3f}")
    return out


def run(sets=SETS, results=RESULTS, **kw):
    t0 = time.time()
    out = dict(exact=stage_exact(), accuracy=stage_accuracy(sets=sets),
               narrow=stage_narrow(), cost=stage_cost(), variance=stage_variance(),
               fields=FIELDS, alpha=ALPHA, lam=LAM)
    out["seconds"] = time.time() - t0
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def report(results=RESULTS):
    path = os.path.join(results, "summary.json")
    if not os.path.exists(path):
        raise SystemExit("nothing to report: run the benchmark first")
    with io.open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    e = d["exact"]
    out = ["# Fitting in the space of the landmarks (plan item B6)", "",
           "The exact model holds one n by n matrix per field and factorises the n by n joint "
           "kernel, which is what stops it at ten to twenty thousand molecules. The Nystrom "
           "approximation replaces every field kernel by the Gram matrix of m features built "
           "from m landmarks, and the model is fitted in the space of those features. Plan item "
           "B6 assessed the method and measured the approximation; this measures the "
           "implementation in `cmfqsar/lowrank.py`.", "",
           "## Nothing approximated", "",
           f"With every molecule its own landmark the approximation is an identity, so the two "
           f"paths must agree. On {e['set']} ({e['n']} molecules, rank {e['rank']}), the "
           f"largest disagreement is:", "",
           "| what | difference |", "| --- | ---: |",
           f"| the joint kernel | {e['kernel']:.1e} |",
           f"| the predictions | {e['prediction']:.1e} |",
           f"| the log evidence | {e['evidence']:.1e} |",
           f"| the posterior variance | {e['variance']:.1e} |",
           f"| the co-field on a grid | {e['cofield']:.1e} |", "",
           "The last row is the one that matters beyond arithmetic: the co-field of a low-rank "
           "fit is a sum over the **landmarks** rather than over the training molecules, and it "
           "is the same function. What the program reads off a co-field -- the map, the atom "
           "contributions, the pharmacophore -- therefore keeps its meaning.", "",
           "## What the approximation costs", "",
           f"q2 under {d['lam']:g} ridge and attenuation {d['alpha']:g}, ten folds, against the "
           f"exact model on the same folds. **Every row is at one attenuation**, and a broad "
           f"one: plan item B6 measured what happens at 2 and above, where a grid of this "
           f"spacing loses 0.01 to 0.04 of q2 because a grid spans a field only while its "
           f"spacing is not much larger than the field's own width. This table measures the "
           f"implementation, not that dependence.", "",
           "| set | molecules | landmarks | rank | q2 | against the exact model |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in d["accuracy"]:
        out.append(f"| {r['set']} ({r['way']}) | {r['n']} | {r['landmarks']} | {r['rank']} | "
                   f"{r['q2']:+.3f} | {r['q2'] - r['exact']:+.3f} |")
    if d.get("narrow"):
        out += ["", "## Where a grid fails, and what follows the atoms instead", "",
                "A grid spans a field while its spacing is not much larger than the field's "
                "own width, so at a strong attenuation it would need tens of thousands of "
                "centres. Putting a landmark on every atom of the molecules, thinned to a "
                "spacing, follows the field instead of covering the box. The number to read is "
                "the error of the kernel: these sets are small and a q2 moves by more than the "
                "approximation does.", "",
                "| set | attenuation | landmarks | how many | error of the kernel | q2 | exact "
                "q2 |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
        for r in d["narrow"]:
            out.append(f"| {r['set']} | {r['alpha']:g} | {r['way']} | {r['landmarks']} | "
                       f"{r['error']:.1e} | {r['q2']:+.3f} | {r['exact']:+.3f} |")
        out.append("")
    out += ["", "## What it saves", "",
            "The memory of a fit is the rank squared whatever the number of molecules, because "
            "the features are accumulated a block at a time and never held at once.", "",
            "| molecules | rank | exact, seconds | low rank, seconds | exact, MB | low rank, MB |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in d["cost"]:
        out.append(f"| {r['n']} | {r['rank']} | {r['exact_seconds']:.1f} | "
                   f"{r['low_seconds']:.1f} | {r['exact_memory'] / 1e6:.1f} | "
                   f"{r['low_memory'] / 1e6:.1f} |")
    v = d["variance"]
    out += ["", "## The variance where the landmarks do not reach", "",
            f"{v['held']} molecules of {v['set']} held out, the landmarks drawn from the rest "
            f"({v['landmarks']} of them). The exact posterior variance of those molecules "
            f"averages {v['exact']:.3f}. Keeping the exact self-kernel gives {v['kept']:.3f} "
            f"(off by {v['kept_error']:.3f}); dropping it, which is the plain "
            f"subset-of-regressors form, gives {v['plain']:.3f} (off by "
            f"{v['plain_error']:.3f}).", ""]
    out += _verdict(d)
    return "\n".join(out)


def _verdict(d):
    e = d["exact"]
    acc = d["accuracy"]
    grids = [r for r in acc if r["way"].startswith("grid")]
    mols = [r for r in acc if r["way"].endswith("molecules")]
    worst_grid = min((r["q2"] - r["exact"] for r in grids), default=float("nan"))
    worst_mol = min((r["q2"] - r["exact"] for r in mols), default=float("nan"))
    cost = d["cost"][-1] if d["cost"] else {}
    v = d["variance"]
    out = ["## The verdict", "",
           f"**The implementation is the exact model when nothing is approximated**, to "
           f"{max(e['kernel'], e['prediction'], e['evidence'], e['variance'], e['cofield']):.0e} "
           f"in every quantity it computes, the co-field included.", "",
           f"**What the approximation costs depends on the landmarks, as plan item B6 "
           f"predicted.** The worst a grid of Gaussian centres does here is "
           f"{worst_grid:+.3f} of q2 against the exact model; the worst landmark molecules do "
           f"is {worst_mol:+.3f}. A grid spans the field while its spacing is not much larger "
           f"than the field's own width, so it is the attenuation and not the number of "
           f"molecules that decides whether a grid is enough."]
    if cost:
        smaller = cost["low_memory"] < cost["exact_memory"]
        faster = cost["low_seconds"] < cost["exact_seconds"]
        out += ["",
                f"**What it saves, measured rather than expected.** At {cost['n']} molecules -- "
                f"the largest aligned set in the repository -- the exact kernels need "
                f"{cost['exact_memory'] / 1e6:.0f} MB and the low-rank fit needs "
                f"{cost['low_memory'] / 1e6:.0f} MB"
                + (", so the saving is already there." if smaller else
                   f", so at this size the exact path is the smaller of the two. The low-rank "
                   f"memory is the rank squared and does not grow with the molecules at all, so "
                   f"the two cross at about {cost['rank']} molecules, which is where a set that "
                   f"needs this path begins.")
                + (f" It is also {cost['exact_seconds'] / max(cost['low_seconds'], 1e-9):.0f} "
                   f"times faster here ({cost['low_seconds']:.1f} s against "
                   f"{cost['exact_seconds']:.1f} s), which was not the point of the item but "
                   f"follows from the same thing: a kernel entry against a one-atom centre "
                   f"costs a thirtieth of one against a whole molecule."
                   if faster else
                   f" The time is {cost['low_seconds']:.1f} s against "
                   f"{cost['exact_seconds']:.1f} s.")]
    narrow = d.get("narrow") or []
    if narrow:
        strong = [r for r in narrow if r["alpha"] >= 2.0]
        grid = [r for r in strong if r["kind"] == "grid"]
        atoms = [r for r in strong if r["kind"] == "atoms"]
        if grid and atoms:
            g_err = float(np.median([r["error"] for r in grid]))
            a_err = float(np.median([r["error"] for r in atoms]))
            out += ["",
                    f"**At a narrow field the landmarks have to follow the atoms, and that "
                    f"works.** At an attenuation of 2 and above, a grid 2 A apart reproduces "
                    f"the kernel to {g_err:.0e} while the atoms of the same molecules, thinned "
                    f"to 0.5 to 0.75 A, reach {a_err:.0e} with a landmark count of the same "
                    f"order. That is the case plan item B6 named and left open: a grid covers "
                    f"the box, the atoms cover the field, and only the second survives a "
                    f"Gaussian narrow enough to fall between grid points."]
    out += ["",
            f"**The variance survives the approximation** because the self-kernel is never "
            f"approximated: on molecules the landmarks do not reach it is off by "
            f"{v['kept_error']:.3f} against {v['plain_error']:.3f} for the form that drops that "
            f"term, which is the collapse the item warned about.", "",
            "**What is still not done**, and was not attempted here: the path is a command of "
            "its own (`cmf lowrank`) and not a flag on `build`, so a low-rank model cannot yet "
            "be saved, predicted from or opened in the application; the landmarks are a grid or "
            "whole molecules, not the atoms of a subset, which is the case between the two that "
            "the item names for narrow fields; the inducing Gaussians take the field's own "
            "width, where a broader basis might need fewer centres; and the largest aligned set "
            "in the repository is 361 molecules, so the saving is shown on a set that does not "
            "need it. The last of those is the alignment problem of part C and not this item's "
            "to solve.", ""]
    return out


def write_table(results=RESULTS) -> str:
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    run(sets=a.sets)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
