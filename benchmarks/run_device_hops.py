#!/usr/bin/env python
"""
The co-field as an atom feature on lithium hops: plan item ND14c on the LiTraj sample.

The external check of plan item BT7 on the published LiTraj collection
(``benchmarks/run_litraj.py``) found that a forest fitted on the description of this program
predicts the barrier of a lithium hop better than the kernel ridge regression on the same
description, and concluded that the fitting method is the remaining gap.  Item ND14c names a
change of the fitting method that keeps the fields: read the co-field of the kernel model at
every atom of the environment of the hop, and give the values to a network that reads every
atom, sums over the atoms and maps the sum to the barrier (``cmfqsar/device.py``).  This script
measures it on the same 1,801 hops of 393 structures.

The model whose co-field is read is the best kernel model of BT7: the overlap of the fields of
two hop environments in the frame of the hop (origin at the midpoint, z along the hop), averaged
over the rotation about the hop (``cmfqsar.hops.HopModel``).  The average makes the co-field a
function of the distance from the hop axis and of the place along it.

The protocol
------------
Ten folds that hold out whole structures, those of ``run_litraj.py``.  For every fold, the
other structures are divided into a validation part (15 % of them) and a fitting part.  The
ridge is chosen on five folds of structures inside the fitting part.  The kernel model is
fitted on the fitting part; a hop of the fitting part reads the co-field of a model fitted
without the structures of its inner fold, and a validation or held-out hop reads the co-field
of the model of the whole fitting part.  The atom network is trained on the fitting part,
stopped on the validation part, and scored on the held-out structures.  Every hop is held out
once, so every score is over all 1,801 hops.

The rows
--------
    ridge regression on the fields            the kernel model whose co-field is read
    atom network: field weights of the atom   the weights of the atom in the 17 fields, no place
    atom network: positions                   the weights, the distance from the hop axis and
                                              the place along the hop
    atom network: co-field                    the weights and the co-field of every field
    atom network: co-field, not out-of-fold   the same, fitting hops read the full model
    atom network: co-field and positions

The stored results of ``run_litraj.py`` are quoted beside them.  They used all nine other folds
for the fit and chose the ridge on the folds they report, so they are slightly favoured.

The comparisons, fixed before any run, are numbers 1 to 5 of ``benchmarks/run_device.py``:
the differences of the absolute error on the same hops, with a 95 % bootstrap interval over
structures (the hops of one structure are not independent).

Usage
-----
    python benchmarks/run_device_hops.py --stage kernel      # 10 to 30 minutes, 8 processes
    python benchmarks/run_device_hops.py --stage features    # about the same again
    python benchmarks/run_device_hops.py --stage own         # needs PyTorch
    python benchmarks/run_device_hops.py --report
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

from cmfqsar.localenv import LAMS                                           # noqa: E402
from cmfqsar.model import krr_fit                                           # noqa: E402

RESULTS = os.path.join(HERE, "results", "device_hops")
CACHE = os.path.join(HERE, "cache", "device")
LITRAJ = os.path.join(HERE, "results", "litraj", "public.json")
RHO = 0.25
R_MAX = 6.0
FOLDS = 10
INNER = 5
VALID_SHARE = 0.15
SEED = 0
RIDGE = "ridge regression on the fields"
OWN = {
    "atom network: field weights of the atom": ("weights",),
    "atom network: positions": ("weights", "place"),
    "atom network: co-field": ("weights", "oof"),
    "atom network: co-field, not out-of-fold": ("weights", "full"),
    "atom network: co-field and positions": ("weights", "oof", "place"),
}
#: the sizes were fixed before any run; the rate, the batch and the patience were chosen on
#: the validation rows of the first COMPAS-1D split (rates 0.001 and 0.003, batches 128 to 512)
NETWORK = {"hidden": (64, 64), "collectors": 16, "brain": (64,), "l2": 1e-4, "epochs": 3000,
           "batch": 128, "lr": 3e-3, "patience": 150, "lr_decay": 0.5}
PAIRS = (
    (1, "atom network: co-field", RIDGE, "the gate: the atom network against the ridge "
        "regression on the same fields"),
    (2, "atom network: co-field", "atom network: field weights of the atom",
     "what the co-field adds to the atom network"),
    (3, "atom network: co-field", "atom network: positions",
     "the co-field against the bare place of the atom"),
    (4, "atom network: co-field and positions", "atom network: positions",
     "what the co-field adds to the place of the atom"),
    (5, "atom network: co-field, not out-of-fold", "atom network: co-field",
     "what the out-of-fold reading is worth (positive: it helps)"),
)
#: rows of results/litraj/public.json quoted in the table: (list, key, text)
STORED = (("models", "overlap 0.25", "BT7: the same kernel model, ridge chosen on the reported "
           "folds, fitted on all nine other folds"),
          ("forests", "rf invariant 0.25 raw", "BT7: random forest on the rows of the "
           "rotation-invariant description"),
          ("triangle_forests", "gb three-distance raw", "BT7: boosted forest on the "
           "three-distance rows, the best model of the check"))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------------------------ the hops
def load_hops(max_structures=0):
    """(targets, field set, the environment of every hop in its hop frame)."""
    import run_litraj as RLJ
    from cmfqsar.hops import framed_environments
    hs, _doc = RLJ.load_prepared(RLJ.PREPARED, max_structures)
    fs = RLJ.fieldset_for(hs, "file")
    crystals = hs.crystals(fs)
    t = hs.targets()
    envs = framed_environments(hs, crystals, rows=t.atom, rho=RHO, r_max=R_MAX)
    return t, fs, envs


def blocks_file(tag=""):
    return os.path.join(CACHE, f"hops{tag}_blocks.npz")


def atoms_file(tag=""):
    return os.path.join(CACHE, f"hops{tag}_atoms.npz")


def result_file(tag=""):
    return os.path.join(RESULTS, f"result{tag}.json")


def kernel_stage(jobs=8, max_structures=0, tag="", force=False):
    """The kernel of every field between the hop environments: (fields, hops, hops)."""
    from cmfqsar.hops import cylindrical_blocks
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(blocks_file(tag)) and not force:
        return
    t0 = time.time()
    _t, fs, envs = load_hops(max_structures)
    blocks = cylindrical_blocks(envs, envs, fs.specs, n_jobs=jobs)
    np.savez_compressed(blocks_file(tag), blocks=blocks)
    log(f"  {blocks.shape[0]} field kernels of {blocks.shape[1]} hops ({time.time() - t0:.0f} s)")


def joint_kernel(blocks, h):
    """The joint kernel of ``cmfqsar.hops.HopModel``: every field divided by its diagonal.

    Returns the kernel and the factors ``1 / sqrt(K_f(i, i))``, (fields, hops); a factor is
    zero where an environment has no atom of the field.
    """
    d = np.stack([np.diag(b) for b in blocks])
    inv = np.where(d > 1e-300, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
    K = np.zeros(blocks.shape[1:])
    for f in range(len(blocks)):
        K += h[f] * blocks[f] * inv[f][:, None] * inv[f][None, :]
    return K, inv


def protocol(molecule, seed=SEED, outer=None):
    """For every outer fold: the held-out hops, the validation hops, the fitting hops and the
    inner fold of every fitting hop.  Structures are never divided."""
    molecule = np.asarray(molecule, int)
    plans = []
    for o, held in enumerate(outer):
        held = np.sort(np.asarray(held, int))
        rest = np.setdiff1d(np.arange(len(molecule)), held)
        order = np.random.RandomState(1000 * int(seed) + o).permutation(np.unique(molecule[rest]))
        n_valid = max(1, int(round(VALID_SHARE * len(order))))
        valid = rest[np.isin(molecule[rest], order[:n_valid])]
        fit = rest[np.isin(molecule[rest], order[n_valid:])]
        fold_of = {int(s): k % INNER for k, s in enumerate(order[n_valid:])}
        plans.append({"held": held, "valid": valid, "fit": fit,
                      "inner": np.array([fold_of[int(s)] for s in molecule[fit]], int)})
    return plans


def fit_fold(K, y, plan, lams=LAMS):
    """The ridge chosen on the inner folds, and the six models of one outer fold.

    Returns (ridge, coefficients (INNER + 1, hops) with zeros outside the rows of a model, biases,
    the inner out-of-fold predictions of the fitting hops).
    """
    fit, inner = plan["fit"], plan["inner"]
    best = None
    for lam in lams:
        pred = np.zeros(len(fit))
        for k in range(INNER):
            rows, out = fit[inner != k], fit[inner == k]
            a, b = krr_fit(K[np.ix_(rows, rows)], y[rows], float(lam))
            pred[inner == k] = K[np.ix_(out, rows)] @ a + b
        sse = float(np.sum((pred - y[fit]) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(lam), pred)
    _sse, lam, cv = best
    coef, bias = np.zeros((INNER + 1, len(y))), np.zeros(INNER + 1)
    for k in range(INNER + 1):
        rows = fit if k == INNER else fit[inner != k]
        a, b = krr_fit(K[np.ix_(rows, rows)], y[rows], lam)
        coef[k, rows], bias[k] = a, b
    return lam, coef, bias, cv


def versions(n, plans):
    """(hops, outer folds): the model every hop reads in every outer fold, out-of-fold."""
    v = np.full((n, len(plans)), INNER, dtype=int)
    for o, plan in enumerate(plans):
        v[plan["fit"], o] = plan["inner"]
    return v


def _atom_block(rows, envs, q, specs, groups, coef, version, inv, h, chunk=40000):
    """The co-field values of the atoms of the environments `rows`.

    For every environment: (atoms, fields, 2 x outer folds).  The first half of the last axis
    reads the model `version` names (out-of-fold for a fitting hop), the second half the model
    of the whole fitting part.  The values are not divided by the norm of the environment
    itself: they are the co-field, the same function of the place for every environment.
    """
    from cmfqsar.hops import _cylindrical, _field_widths, _geometric_block
    n_outer = coef.shape[0]
    rq, zq, wq, owner = q["r"], q["z"], q["w"], q["owner"]
    out = []
    for i in rows:
        e = envs[int(i)]
        rp, zp = _cylindrical(e)
        C = np.concatenate([coef[np.arange(n_outer), version[int(i)]], coef[:, -1]], axis=0).T
        vals = np.zeros((e.n_atoms, len(specs), 2 * n_outer))
        for key, fields in groups:
            bp = _field_widths(e, specs[fields[0]], fields[0])
            for c0 in range(0, len(rq), int(chunk)):
                c1 = min(c0 + int(chunk), len(rq))
                G = _geometric_block(rp, zp, bp, rq[c0:c1], zq[c0:c1], q["b"][key][c0:c1])
                own = owner[c0:c1]
                for f in fields:
                    w = wq[c0:c1, f]
                    keep = np.abs(w) > 1e-12
                    if keep.any():
                        Cq = (w[keep] * inv[f][own[keep]])[:, None] * C[own[keep]]
                        vals[:, f, :] += G[:, keep] @ Cq
        vals *= h[None, :, None]
        out.append(vals.astype(np.float32))
    return out


def features_stage(jobs=8, max_structures=0, tag="", force=False):
    from joblib import Parallel, delayed
    from cmfqsar.hops import _q_arrays, _width_groups
    os.makedirs(RESULTS, exist_ok=True)
    if os.path.exists(atoms_file(tag)) and os.path.exists(result_file(tag)) and not force:
        return
    t0 = time.time()
    t, fs, envs = load_hops(max_structures)
    specs = [s for s in fs.specs if s.enabled]
    h = np.array([s.h for s in specs], float)
    y = np.asarray(t.y, float)
    n = len(y)
    K, inv = joint_kernel(np.load(blocks_file(tag))["blocks"], h)
    plans = protocol(t.molecule, SEED, t.folds(FOLDS, SEED))
    coef = np.zeros((len(plans), INNER + 1, n))
    bias = np.zeros((len(plans), INNER + 1))
    lams, ridge = [], np.zeros(n)
    cv_rmse, fit_rmse = [], []
    for o, plan in enumerate(plans):
        lam, coef[o], bias[o], cv = fit_fold(K, y, plan)
        lams.append(lam)
        ridge[plan["held"]] = K[plan["held"]] @ coef[o, -1] + bias[o, -1]
        fitted = K[plan["fit"]] @ coef[o, -1] + bias[o, -1]
        cv_rmse.append(float(np.sqrt(np.mean((cv - y[plan["fit"]]) ** 2))))
        fit_rmse.append(float(np.sqrt(np.mean((fitted - y[plan["fit"]]) ** 2))))
        log(f"  fold {o}: {len(plan['fit'])} fitting, {len(plan['valid'])} validation, "
            f"{len(plan['held'])} held-out hops; ridge {lam:g}; RMSE fitted {fit_rmse[-1]:.3f}, "
            f"inner out-of-fold {cv_rmse[-1]:.3f}")
    version = versions(n, plans)
    groups = _width_groups(specs)
    q = _q_arrays(envs, specs, groups)
    parts = [np.arange(n)[k::4 * jobs] for k in range(4 * jobs)]
    parts = [p for p in parts if len(p)]
    res = Parallel(n_jobs=jobs)(delayed(_atom_block)(p, envs, q, specs, groups, coef, version,
                                                     inv, h) for p in parts)
    values = [None] * n
    for p, block in zip(parts, res):
        for i, v in zip(p, block):
            values[int(i)] = v
    # the check: the values times the weights of the atoms, summed and divided by the norm of
    # the environment in every field, give the prediction back
    worst = 0.0
    for o, plan in enumerate(plans):
        for i in plan["held"][:20]:
            w = np.asarray(envs[int(i)].weights, float)
            per_field = (values[int(i)][:, :, len(plans) + o] * w).sum(axis=0)
            total = float(per_field @ inv[:, int(i)]) + bias[o, -1]
            worst = max(worst, abs(total - ridge[int(i)]))
    counts = np.array([e.n_atoms - 1 for e in envs])              # atom 0 is the midpoint marker
    A, F = int(counts.max()), len(specs)
    out = {"weights": np.zeros((n, A, F), np.float32), "place": np.zeros((n, A, 2), np.float32),
           "mask": np.zeros((n, A), bool),
           "values": np.zeros((n, A, F, 2 * len(plans)), np.float32)}
    for i, e in enumerate(envs):
        k = counts[i]
        c = np.asarray(e.coords, float)[1:]
        out["weights"][i, :k] = np.asarray(e.weights, float)[1:]
        out["place"][i, :k, 0], out["place"][i, :k, 1] = np.hypot(c[:, 0], c[:, 1]), c[:, 2]
        out["mask"][i, :k] = True
        out["values"][i, :k] = values[i][1:]
    np.savez_compressed(atoms_file(tag), **out)
    doc = {"n": n, "structures": int(len(np.unique(t.molecule))), "fields": fs.names,
           "y": y.tolist(), "structure": [int(v) for v in t.molecule],
           "folds": {"held": [p["held"].tolist() for p in plans],
                     "valid": [p["valid"].tolist() for p in plans],
                     "fit": [p["fit"].tolist() for p in plans]},
           "lam": lams, "models": {RIDGE: ridge.tolist()},
           "check": {"sum_of_atoms_max_difference": worst, "fit_rmse": fit_rmse,
                     "inner_out_of_fold_rmse": cv_rmse,
                     "atoms_per_environment": float(counts.mean())},
           "seconds": {"features": time.time() - t0}}
    with open(result_file(tag), "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    e = ridge - y
    log(f"  ridge on the held-out hops: q2 {1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2):+.3f}"
        f", MAE {np.abs(e).mean():.3f} eV; sum check {worst:.1e} ({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- the atom network
def own_stage(tag="", device=None, force=False, models=None):
    from cmfqsar import device as D
    with open(result_file(tag), encoding="utf-8") as fh:
        doc = json.load(fh)
    z = np.load(atoms_file(tag))
    y = np.asarray(doc["y"], float)
    n_outer = len(doc["folds"]["held"])
    for label in (models or OWN):
        if label in doc["models"] and not force:
            continue
        t0 = time.time()
        pred = np.zeros(len(y))
        epochs = []
        for o in range(n_outer):
            blocks = {"weights": z["weights"], "place": z["place"],
                      "oof": z["values"][:, :, :, o], "full": z["values"][:, :, :, n_outer + o]}
            X = np.concatenate([np.asarray(blocks[b], float) for b in OWN[label]], axis=2)
            held = np.asarray(doc["folds"]["held"][o], int)
            res = D.train_device(X, z["mask"], y, doc["folds"]["fit"][o],
                                 doc["folds"]["valid"][o], seed=o, device=device, **NETWORK)
            pred[held] = res.pred[held]
            epochs.append(int(res.epochs))
        doc["models"][label] = pred.tolist()
        doc.setdefault("trained", {})[label] = {"epochs": epochs, "seconds": time.time() - t0}
        e = pred - y
        log(f"  {label}: q2 {1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2):+.3f}, MAE "
            f"{np.abs(e).mean():.3f} eV, epochs {min(epochs)} to {max(epochs)}, "
            f"{time.time() - t0:.0f} s")
        with open(result_file(tag), "w", encoding="utf-8") as fh:
            json.dump(doc, fh)


# ---------------------------------------------------------------------------------- the report
def paired(y, a, b, structure, draws=2000, seed=0):
    """Mean absolute error of `a` minus that of `b`, with a bootstrap interval over structures."""
    d = np.abs(a - y) - np.abs(b - y)
    groups = [np.flatnonzero(structure == s) for s in np.unique(structure)]
    sums = np.array([d[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups], float)
    rng = np.random.default_rng(seed)
    boot = np.zeros(draws)
    for k in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        boot[k] = sums[pick].sum() / sizes[pick].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(d.mean()), "low": float(lo), "high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def write_table(tag=""):
    with open(result_file(tag), encoding="utf-8") as fh:
        doc = json.load(fh)
    y = np.asarray(doc["y"], float)
    structure = np.asarray(doc["structure"], int)
    lines = ["# The co-field as an atom feature on lithium hops (ND14c, LiTraj sample)", "",
             "Written by `benchmarks/run_device_hops.py --report`. "
             f"{doc['n']} hops of {doc['structures']} structures, the barrier of the "
             "bond-valence site-energy model in eV, ten folds that hold out structures. In "
             "every fold 15 % of the other structures are the validation part, and the rest is "
             "the fitting part. \"Atom network\" is the network of `cmfqsar/device.py`: one "
             "network on every atom of the environment of the hop, a sum over the atoms, a "
             "second network. The co-field is that of the ridge regression of the first row. "
             f"The standard deviation of the barriers is {y.std(ddof=1):.3f} eV.", "",
             "| Model | q2 over all hops | Mean absolute error, eV | Root mean square error, eV |",
             "|---|---|---|---|"]
    for label in [RIDGE] + list(OWN):
        if label in doc["models"]:
            e = np.asarray(doc["models"][label], float) - y
            lines.append(f"| {label} | {1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2):+.3f} | "
                         f"{np.abs(e).mean():.3f} | {np.sqrt(np.mean(e ** 2)):.3f} |")
    if os.path.exists(LITRAJ) and not tag:
        with open(LITRAJ, encoding="utf-8") as fh:
            pub = json.load(fh)
        for where, key, text in STORED:
            for m in pub.get(where, []):
                if m.get("key") == key:
                    lines.append(f"| {text} | {m['q2']:+.3f} | {m['mae']:.3f} | {m['rmse']:.3f} |")
    lines += ["", "## Paired comparisons on the same hops", "",
              "The difference is the mean absolute error of the first model minus that of the "
              "second, in eV; a negative number means that the first model is better. The "
              "interval is the 95 % bootstrap interval over structures.", "",
              "| No. | First model | Second model | Difference [95 % interval] | What it says |",
              "|---|---|---|---|---|"]
    for no, a, b, text in PAIRS:
        if a in doc["models"] and b in doc["models"]:
            p = paired(y, np.asarray(doc["models"][a], float), np.asarray(doc["models"][b], float),
                       structure)
            cell = (f"{p['mean']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]"
                    + ("" if p["excludes_zero"] else " (includes zero)"))
            lines.append(f"| {no} | {a} | {b} | {cell} | {text} |")
    c = doc["check"]
    lines += ["", "## Checks", "",
              f"* The largest difference between the prediction of the ridge regression and the "
              f"sum of the co-field values times the weights of the atoms, over 200 held-out "
              f"hops: {c['sum_of_atoms_max_difference']:.1e} eV.",
              f"* The ridge chosen in the ten folds: "
              f"{', '.join(f'{v:g}' for v in sorted(set(doc['lam'])))}.",
              f"* Root mean square error of the kernel model on its fitting part: fitted "
              f"{np.mean(c['fit_rmse']):.3f} eV, out-of-fold on the inner folds "
              f"{np.mean(c['inner_out_of_fold_rmse']):.3f} eV (means over the ten folds).",
              f"* Atoms in the environment of a hop: {c['atoms_per_environment']:.0f} on average.",
              ""]
    if doc.get("trained"):
        lines += ["## Training", "", "| Model | Epochs (smallest to largest of the ten folds) | "
                  "Time for the ten folds, minutes |", "|---|---|---|"]
        for label, tr in doc["trained"].items():
            lines.append(f"| {label} | {min(tr['epochs'])} to {max(tr['epochs'])} | "
                         f"{tr['seconds'] / 60.0:.1f} |")
        lines.append("")
    path = os.path.join(RESULTS, f"table{tag}.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    return path


STAGES = ("kernel", "features", "own")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=STAGES, action="append")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--jobs", type=int, default=8, help="processes for the kernel and the atoms")
    ap.add_argument("--max-structures", type=int, default=0,
                    help="use only the first structures (a quick trial; results get a tag)")
    ap.add_argument("--torch-device", default=None, help="cpu or cuda (default: cuda if any)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    tag = f"_first{a.max_structures}" if a.max_structures else ""
    for stage in a.stage or []:
        if stage == "kernel":
            kernel_stage(a.jobs, a.max_structures, tag, a.force)
        elif stage == "features":
            features_stage(a.jobs, a.max_structures, tag, a.force)
        else:
            own_stage(tag, a.torch_device, a.force)
    if a.report or not a.stage:
        log("wrote " + os.path.relpath(write_table(tag), ROOT))


if __name__ == "__main__":
    main()
