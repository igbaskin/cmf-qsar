#!/usr/bin/env python
"""
CMF as a featuriser: plan item ND1.

A CMF model is a kernel model, and a kernel has an explicit feature map: with landmark
molecules Z and the whitening W of ``cmfqsar.lowrank`` (W W^T = K(Z, Z)^+), the row
``phi(x) = K(x, Z) W`` is a vector whose inner products are the kernel.  That vector can be
given to any method that takes vectors.  Item ND1 asks what is gained by doing so, and names
three things to compare: CMF alone, the neural device alone, and the two combined.

"The device" is the QSPR Neural Device (Baskin, Palyulin, Zefirov, J. Chem. Inf. Comput. Sci.
37 (1997) 715; package ``neurodevice``), an early graph neural network: networks with shared
weights read the atoms, the bonds and the two-bond paths of a molecular graph, their outputs
are summed into a few *collectors*, and a small network (the "brain") maps the collectors to
the property.  The collectors are the descriptors the device has learned.  It is a separate
project; ``benchmarks/featuriser_device.py`` runs it and stores, per split, its prediction and
its collectors for every molecule.

The sets
--------
* **COMPAS-1D**, the set the plan names: the HOMO-LUMO gap of 8,678 cata-condensed polybenzenoid
  hydrocarbons at B3LYP-D3BJ/def2-SVP, in the frame by rule of ``cmfqsar/pbh.py``, with the
  stored group kernels of ``benchmarks/run_compas.py`` (steric, charge, atom class, ring class
  and Clar sextet fields: the groups ABCDE).  Five seeds.
* **The eight Sutherland series** (66 to 361 molecules, superimposed, five fields).  Ten seeds.

The split is the one of the device's own protocol: random, 70 % training, 15 % validation,
15 % test.  Every setting of every model is chosen on the validation rows, and every score is
on the test rows, which no choice has seen.

The models
----------
    CMF kernel             the kernel ridge regression of this program, the reference
    CMF kernel, Gaussian   the same with the Gaussian kernel layer (COMPAS only)
    device, one-shot       the device without message passing
    device, message passing    the device with three rounds of message passing
    fingerprint            ECFP4 counts with a Tanimoto kernel

and three learners (ridge regression, a multilayer network trained by back-propagation, a
random forest) on five inputs: the CMF vector, the collectors of each device, and the CMF
vector joined with the collectors of each device.  A ridge on the CMF vector is the CMF kernel
model again, up to the rank of the landmarks, so it is the check of the vector.  Two simple
combinations of predictions close the list: the mean of the CMF and the device prediction, and
their weighted mean with the weight chosen on the validation rows.

Usage
-----
    python benchmarks/run_featuriser.py --stage tables       # the molecule tables for the device
    python benchmarks/run_featuriser.py --stage device \\
        --device-root C:/path/to/the/device --device-python C:/path/to/its/python
    python benchmarks/run_featuriser.py --stage reference    # the CMF model of every split
    python benchmarks/run_featuriser.py --stage models       # the learners and the scores
    python benchmarks/run_featuriser.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, DataStructs, RDLogger                               # noqa: E402
from rdkit.Chem import AllChem                                              # noqa: E402

from cmfqsar.dataio import load_dataset                                     # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields                # noqa: E402
from cmfqsar.lowrank import whitener                                        # noqa: E402
from cmfqsar.model import CMFModel, krr_fit, mae, r2_score, rmse            # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "featuriser")
DEVICE = os.path.join(RESULTS, "device")
CACHE = os.path.join(HERE, "cache", "featuriser")
SUTHERLAND = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
SETS = ("compas-1D",) + SUTHERLAND
SEEDS = {s: tuple(range(10)) for s in SUTHERLAND}
SEEDS["compas-1D"] = tuple(range(5))
DEVICE_METHODS = ("oneshot", "mp")
DEVICE_LABELS = {"oneshot": "device, one-shot", "mp": "device, message passing"}

#: the groups of the COMPAS field model: steric, charge, atom class, ring class, Clar sextet.
#: Every one of them is made from the structure; the HOMA field of group F reads the bond
#: lengths of the DFT geometry and is left out of a comparison with a model of the graph.
COMPAS_GROUPS = "ABCDE"
LANDMARKS = 1000                    # landmark molecules of the CMF vector on COMPAS
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")
ALPHAS = (0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
LAMS = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
#: the multilayer network: hidden layers and L2 penalties tried, the choice on validation
MLP_LAYERS = ((64,), (128, 64))
MLP_L2 = (1e-4, 1e-2, 1.0)
#: the forest: trees, and the share of the columns tried at a split
TREES = 300
FOREST_FEATURES = ("sqrt", 0.33)
LEARNERS = ("ridge", "network", "forest")
LEARNER_LABELS = {"ridge": "ridge regression", "network": "multilayer network",
                  "forest": "random forest"}
INPUTS = ("CMF vector", "collectors, one-shot", "collectors, message passing",
          "CMF vector + collectors, one-shot", "CMF vector + collectors, message passing")
WEIGHTS = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------------------------ the data
def load_set(name):
    """(names, molecules, activities) in the order every stage uses."""
    if name == "compas-1D":
        import run_compas as RC
        ds, _meta = RC.load_prepared("1D")
    else:
        ds = load_dataset(os.path.join(ROOT, "examples", "sutherland", f"{name}.sdf"))
    return list(ds.names), list(ds.mols), np.asarray(ds.activity, float)


def tables_stage():
    """The name, SMILES and label tables the device reads, in the order of ``load_set``."""
    os.makedirs(CACHE, exist_ok=True)
    for name in SETS:
        names, mols, y = load_set(name)
        path = os.path.join(CACHE, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["name", "smiles", "label"])
            for nm, m, v in zip(names, mols, y):
                w.writerow([nm, Chem.MolToSmiles(Chem.RemoveHs(m)), f"{float(v):.6g}"])
        log(f"  {name}: {len(names)} molecules -> {os.path.relpath(path, ROOT)}")


def device_file(name, method, seed):
    return os.path.join(DEVICE, f"{name}_{method}_{seed}.npz")


def device_stage(device_root, device_python, workers=4, sets=SETS, gpu=False):
    """Train the device for every set, method and seed that has no file yet."""
    os.makedirs(DEVICE, exist_ok=True)
    jobs = [(s, m, k) for s in sets for k in SEEDS[s] for m in DEVICE_METHODS
            if not os.path.exists(device_file(s, m, k))]
    log(f"{len(jobs)} device runs with {workers} processes")
    running = []
    script = os.path.join(HERE, "featuriser_device.py")
    while jobs or running:
        while jobs and len(running) < workers:
            s, m, k = jobs.pop(0)
            cmd = [device_python, script, "--device-root", device_root, "--table",
                   os.path.join(CACHE, f"{s}.csv"), "--method", m, "--seed", str(k),
                   "--out", device_file(s, m, k)] + (["--gpu"] if gpu else [])
            running.append((subprocess.Popen(cmd), s, m, k))
        time.sleep(2.0)
        still = []
        for proc, s, m, k in running:
            if proc.poll() is None:
                still.append((proc, s, m, k))
            elif proc.returncode != 0:
                log(f"  FAILED {s} {m} seed {k} (exit {proc.returncode})")
        running = still


def read_device(name, seed, names, methods=DEVICE_METHODS):
    """The device files of one split: the index sets, and per method prediction and collectors."""
    out = {}
    for m in methods:
        path = device_file(name, m, seed)
        if not os.path.exists(path):
            return None
        z = np.load(path, allow_pickle=False)
        if [str(n) for n in z["names"]] != list(names):
            raise SystemExit(f"{path} lists other molecules than the data set; rerun the "
                             "stages tables and device")
        split = {k: np.asarray(z[k], int) for k in ("train", "valid", "test")}
        if "split" in out and any(not np.array_equal(out["split"][k], split[k]) for k in split):
            raise SystemExit(f"the device files of {name} seed {seed} hold different splits")
        out["split"] = split
        out[m] = {"pred": np.asarray(z["pred"], float),
                  "collectors": np.asarray(z["collectors"], float),
                  "seconds": float(z["seconds"])}
    return out


# --------------------------------------------------------------------------------- the kernels
def ridge_path(K_tr, y_tr, K_others, lams=LAMS):
    """Kernel ridge regression with a bias for every ridge: {lam: [predictions per block]}."""
    out = {}
    for lam in lams:
        a, b = krr_fit(K_tr, y_tr, float(lam))
        out[float(lam)] = [K @ a + b for K in K_others]
    return out


def best_on_validation(preds_by_setting, y_valid, which=0):
    """The setting whose predictions of block `which` have the lowest RMSE on validation."""
    return min(preds_by_setting, key=lambda k: rmse(y_valid, preds_by_setting[k][which]))


def sutherland_kernels(mols, names):
    """{alpha: (n, n) mean of the five cosine-normalised field kernels}, file charges."""
    out = {}
    for a in ALPHAS:
        fs = FieldSet.create(physchem=list(FIELDS), alpha=float(a), charge_model="file")
        table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
        Kf = np.asarray(CMFModel(fieldset=fs, method="krr",
                                 normalize=True).prepare_training_kernels(table))
        out[a] = Kf.mean(axis=0)
    return out


def tanimoto(mols):
    fps = [AllChem.GetHashedMorganFingerprint(Chem.RemoveHs(m), 2, nBits=2048) for m in mols]
    return np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])


def cmf_reference(name, split, y, state):
    """The CMF kernel model on one split and the CMF vector of every molecule.

    Returns ({label: predictions for all molecules}, Phi, a dictionary of what was chosen).
    """
    tr, va = split["train"], split["valid"]
    n = len(y)
    everyone = np.arange(n)
    if name == "compas-1D":
        import run_compas as RC
        store = state["store"]
        r = RC.evaluate_set(store, COMPAS_GROUPS,
                            {"train": tr, "valid": va, "test": split["test"], "seed": state["seed"],
                             "kind": "random", "name": f"featuriser{state['seed']}"},
                            y, layer="gauss")
        lin = "per-group" if "per-group" in r["configs"] else "shared"
        alphas = r["configs"][lin]["alphas"]
        lam = float(r["configs"][lin]["lam"])
        K_tr = store.joint(alphas, tr)
        a, b = krr_fit(K_tr, y[tr], lam)
        preds = {"CMF kernel": store.joint(alphas, everyone, tr) @ a + b}
        g = r["configs"]["gauss"]
        d_tr = store.diag(g["alphas"], tr)
        med = RC.median_d2(K_tr)
        Kg = RC.gauss_layer(store.joint(g["alphas"], tr), d_tr, d_tr, g["gamma"], med)
        ag, bg = krr_fit(Kg, y[tr], float(g["lam"]))
        Kg_all = RC.gauss_layer(store.joint(g["alphas"], everyone, tr),
                                store.diag(g["alphas"], everyone), d_tr, g["gamma"], med)
        preds["CMF kernel, Gaussian layer"] = Kg_all @ ag + bg
        rng = np.random.RandomState(state["seed"])
        land = np.sort(rng.choice(tr, size=min(LANDMARKS, len(tr)), replace=False))
        W = whitener(store.joint(alphas, land))
        Phi = store.joint(alphas, everyone, land) @ W
        chosen = {"alphas": alphas, "lam": lam, "gamma": g["gamma"], "landmarks": int(len(land)),
                  "rank": int(W.shape[1]), "kernel_seconds": float(r["seconds"])}
    else:
        Ks = state["kernels"]
        by = {}
        for a_, K in Ks.items():
            for lam, (pv, pall) in ridge_path(K[np.ix_(tr, tr)], y[tr],
                                              [K[np.ix_(va, tr)], K[:, tr]]).items():
                by[(a_, lam)] = [pv, pall]
        a_, lam = best_on_validation(by, y[va])
        preds = {"CMF kernel": by[(a_, lam)][1]}
        K = Ks[a_]
        W = whitener(K[np.ix_(tr, tr)])
        Phi = K[:, tr] @ W
        chosen = {"alpha": a_, "lam": lam, "landmarks": int(len(tr)), "rank": int(W.shape[1])}
    return preds, Phi, chosen


def reference_file(name, seed):
    return os.path.join(CACHE, f"{name}_{seed}_reference.npz")


def cached_reference(name, split, y, state):
    """`cmf_reference`, kept in the cache folder: on COMPAS the choice of the attenuations is
    most of an hour for one split, and it needs the split only, not the device's results."""
    path = reference_file(name, state["seed"])
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if all(np.array_equal(z[k], split[k]) for k in ("train", "valid", "test")):
            labels = [str(v) for v in z["labels"]]
            return ({k: z["preds"][i] for i, k in enumerate(labels)}, z["Phi"],
                    json.loads(str(z["chosen"])))
    preds, Phi, chosen = cmf_reference(name, split, y, state)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, labels=np.array(list(preds)),
                        preds=np.array([preds[k] for k in preds]), Phi=Phi,
                        chosen=json.dumps(chosen), **split)
    return preds, Phi, chosen


def reference_stage(sets=SETS, seeds=None):
    """The CMF reference of every split whose one-shot device file exists."""
    for name in sets:
        todo = [k for k in (seeds if seeds is not None else SEEDS[name])
                if not os.path.exists(reference_file(name, k))
                and os.path.exists(device_file(name, DEVICE_METHODS[0], k))]
        if not todo:
            continue
        names, mols, y = load_set(name)
        state = {"names": names, "mols": mols, "y": y}
        if name == "compas-1D":
            import run_compas as RC
            state["store"] = RC.KernelStore("1D", max_loaded=4)
        else:
            state["kernels"] = sutherland_kernels(mols, names)
        for k in todo:
            t0 = time.time()
            state["seed"] = k
            split = read_device(name, k, names, DEVICE_METHODS[:1])["split"]
            preds, _Phi, chosen = cached_reference(name, split, y, state)
            te = split["test"]
            log(f"  {name} seed {k}: CMF kernel test R2 "
                f"{r2_score(y[te], preds['CMF kernel'][te]):+.3f}, rank {chosen['rank']} "
                f"({time.time() - t0:.0f} s)")


# -------------------------------------------------------------------------------- the learners
def scaled_blocks(blocks, tr):
    """Join blocks of columns so that no block outweighs another by its size.

    A block is (columns, standardise).  Each block is centred on the training rows; the
    collectors are also divided column by column by their standard deviation there, because
    each collector has a scale of its own.  The CMF vector is not: the length of each of its
    columns is the weight the kernel gives that direction, and making the columns equal would
    raise the directions the kernel holds for noise to the level of the others.  Every block is
    then divided by the root of its mean squared row length on the training rows.
    """
    out = []
    for X, standardise in blocks:
        X = np.asarray(X, float)
        Xc = X - X[tr].mean(axis=0)
        if standardise:
            sd = Xc[tr].std(axis=0)
            Xc = Xc[:, sd > 1e-12] / sd[sd > 1e-12]
        s = float(np.sqrt(np.mean(np.sum(Xc[tr] ** 2, axis=1))))
        out.append(Xc / max(s, 1e-12))
    return np.hstack(out)


def learn_ridge(X, y, split):
    tr, va = split["train"], split["valid"]
    U, s, Vt = np.linalg.svd(X[tr], full_matrices=False)
    mean = float(y[tr].mean())
    Uy = U.T @ (y[tr] - mean)
    by = {}
    for lam in LAMS:
        w = Vt.T @ (s / (s * s + lam) * Uy)
        p = X @ w + mean
        by[float(lam)] = [p[va], p]
    lam = best_on_validation(by, y[va])
    return by[lam][1], {"lam": lam}


def learn_network(X, y, split, seed):
    from sklearn.neural_network import MLPRegressor
    tr, va = split["train"], split["valid"]
    # one factor for all columns, so that the largest column has unit standard deviation: the
    # relative lengths of the columns are kept (see scaled_blocks)
    Z = X / max(float(X[tr].std(axis=0).max()), 1e-12)
    my, sy = float(y[tr].mean()), float(y[tr].std()) or 1.0
    by = {}
    for layers in MLP_LAYERS:
        for l2 in MLP_L2:
            net = MLPRegressor(hidden_layer_sizes=layers, alpha=l2, max_iter=500,
                               early_stopping=True, validation_fraction=0.15,
                               n_iter_no_change=20, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                net.fit(Z[tr], (y[tr] - my) / sy)
            p = net.predict(Z) * sy + my
            by[(layers, l2)] = [p[va], p]
    key = best_on_validation(by, y[va])
    return by[key][1], {"layers": list(key[0]), "l2": key[1]}


def learn_forest(X, y, split, seed, jobs):
    from sklearn.ensemble import RandomForestRegressor
    tr, va = split["train"], split["valid"]
    by = {}
    for feats in FOREST_FEATURES:
        rf = RandomForestRegressor(n_estimators=TREES, max_features=feats, random_state=seed,
                                   n_jobs=jobs)
        rf.fit(X[tr], y[tr])
        p = rf.predict(X)
        by[feats] = [p[va], p]
    key = best_on_validation(by, y[va])
    return by[key][1], {"max_features": key}


def scores(y, pred, test):
    return {"r2": float(r2_score(y[test], pred[test])), "rmse": float(rmse(y[test], pred[test])),
            "mae": float(mae(y[test], pred[test])),
            "abs_errors": np.abs(pred[test] - y[test]).astype(np.float32).tolist()}


def run_split(name, seed, state, jobs=4):
    """Every model on one split of one set."""
    names, mols, y = state["names"], state["mols"], state["y"]
    dev = read_device(name, seed, names)
    if dev is None:
        return None
    split = dev["split"]
    tr, va, te = split["train"], split["valid"], split["test"]
    t0 = time.time()
    state["seed"] = seed
    out = {"set": name, "seed": seed, "n": len(y), "n_train": len(tr), "n_valid": len(va),
           "n_test": len(te), "test": te.tolist(), "models": {}, "chosen": {}}
    preds, Phi, chosen = cached_reference(name, split, y, state)
    out["chosen"]["CMF kernel"] = chosen
    for m in DEVICE_METHODS:
        preds[DEVICE_LABELS[m]] = dev[m]["pred"]
        out["chosen"][DEVICE_LABELS[m]] = {"collectors": int(dev[m]["collectors"].shape[1]),
                                           "seconds": dev[m]["seconds"]}
    T = state["tanimoto"]
    by = {lam: v for lam, v in ridge_path(T[np.ix_(tr, tr)], y[tr],
                                          [T[np.ix_(va, tr)], T[:, tr]]).items()}
    lam = best_on_validation(by, y[va])
    preds["fingerprint"] = by[lam][1]
    c1, c2 = (dev["oneshot"]["collectors"], True), (dev["mp"]["collectors"], True)
    blocks = {"CMF vector": [(Phi, False)], "collectors, one-shot": [c1],
              "collectors, message passing": [c2],
              "CMF vector + collectors, one-shot": [(Phi, False), c1],
              "CMF vector + collectors, message passing": [(Phi, False), c2]}
    for label in INPUTS:
        X = scaled_blocks(blocks[label], tr)
        for learner in LEARNERS:
            key = f"{label} | {learner}"
            if learner == "ridge":
                preds[key], out["chosen"][key] = learn_ridge(X, y, split)
            elif learner == "network":
                preds[key], out["chosen"][key] = learn_network(X, y, split, seed)
            else:
                preds[key], out["chosen"][key] = learn_forest(X, y, split, seed, jobs)
        log(f"    {name} seed {seed}: {label} done ({time.time() - t0:.0f} s)")
    # the two predictions combined: the mean, and a weight chosen on the validation rows
    for m in DEVICE_METHODS:
        a, b = preds["CMF kernel"], preds[DEVICE_LABELS[m]]
        preds[f"mean of CMF kernel and {DEVICE_LABELS[m]}"] = 0.5 * (a + b)
        w = min(WEIGHTS, key=lambda w: rmse(y[va], w * a[va] + (1 - w) * b[va]))
        preds[f"weighted mean of CMF kernel and {DEVICE_LABELS[m]}"] = w * a + (1 - w) * b
        out["chosen"][f"weighted mean of CMF kernel and {DEVICE_LABELS[m]}"] = {"weight": w}
    for key, p in preds.items():
        out["models"][key] = scores(y, np.asarray(p, float), te)
    out["seconds"] = time.time() - t0
    return out


def split_file(name, seed):
    return os.path.join(RESULTS, f"{name}_{seed}.json")


def models_stage(sets=SETS, seeds=None, jobs=4, force=False):
    os.makedirs(RESULTS, exist_ok=True)
    for name in sets:
        todo = [k for k in (seeds if seeds is not None else SEEDS[name])
                if (force or not os.path.exists(split_file(name, k)))
                and all(os.path.exists(device_file(name, m, k)) for m in DEVICE_METHODS)]
        if not todo:
            continue
        names, mols, y = load_set(name)
        state = {"names": names, "mols": mols, "y": y, "tanimoto": tanimoto(mols)}
        if name == "compas-1D":
            import run_compas as RC
            state["store"] = RC.KernelStore("1D", max_loaded=6)
        else:
            state["kernels"] = sutherland_kernels(mols, names)
        for k in todo:
            r = run_split(name, k, state, jobs)
            if r is None:
                continue
            with open(split_file(name, k), "w", encoding="utf-8", newline="\n") as fh:
                json.dump(r, fh)
            m = r["models"]
            log(f"  {name} seed {k}: CMF kernel {m['CMF kernel']['r2']:+.3f}, device one-shot "
                f"{m[DEVICE_LABELS['oneshot']]['r2']:+.3f}, message passing "
                f"{m[DEVICE_LABELS['mp']]['r2']:+.3f} ({r['seconds']:.0f} s)")


# ------------------------------------------------------------------------------------ the table
def collect(name):
    out = []
    for k in SEEDS[name]:
        p = split_file(name, k)
        if os.path.exists(p):
            out.append(json.load(open(p, encoding="utf-8")))
    return out


def paired(runs, a, b, draws=2000, seed=0):
    """The difference of the absolute errors of two models over the test molecules of every
    seed: a molecule that is in several test parts counts once, with its mean difference."""
    diff = {}
    for r in runs:
        da = np.asarray(r["models"][a]["abs_errors"], float)
        db = np.asarray(r["models"][b]["abs_errors"], float)
        for i, v in zip(r["test"], da - db):
            diff.setdefault(int(i), []).append(float(v))
    d = np.array([np.mean(v) for v in diff.values()])
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(draws)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(d.mean()), "low": float(lo), "high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "molecules": int(len(d))}


def _pm(runs, key, what="r2", nd=3):
    v = np.array([r["models"][key][what] for r in runs])
    return f"{v.mean():+.{nd}f} ± {v.std():.{nd}f}" if what == "r2" else \
        f"{v.mean():.{nd}f} ± {v.std():.{nd}f}"


def _pair(p):
    return (f"{p['mean']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]"
            + ("" if p["excludes_zero"] else " (includes zero)"))


DIRECT = ("CMF kernel", "CMF kernel, Gaussian layer", DEVICE_LABELS["oneshot"],
          DEVICE_LABELS["mp"], "fingerprint")
STACKS = tuple(f"{w} of CMF kernel and {DEVICE_LABELS[m]}" for m in DEVICE_METHODS
               for w in ("mean", "weighted mean"))
PAIRS = (
    ("CMF vector | ridge", "CMF kernel"),
    ("CMF vector | network", "CMF vector | ridge"),
    ("CMF vector | forest", "CMF vector | ridge"),
    ("CMF vector + collectors, one-shot | ridge", "CMF vector | ridge"),
    ("CMF vector + collectors, message passing | ridge", "CMF vector | ridge"),
    ("CMF vector + collectors, one-shot | network", DEVICE_LABELS["oneshot"]),
    ("CMF vector + collectors, message passing | network", DEVICE_LABELS["mp"]),
    # added after the first COMPAS seed: a network on the collectors alone already differs
    # from the device, so these two pairs hold the learner fixed and change the input only
    ("CMF vector + collectors, one-shot | network", "collectors, one-shot | network"),
    ("CMF vector + collectors, message passing | network",
     "collectors, message passing | network"),
    ("CMF vector + collectors, message passing | network", "CMF vector | network"),
    (f"weighted mean of CMF kernel and {DEVICE_LABELS['oneshot']}", "CMF kernel"),
    (f"weighted mean of CMF kernel and {DEVICE_LABELS['mp']}", "CMF kernel"),
    (f"weighted mean of CMF kernel and {DEVICE_LABELS['mp']}", DEVICE_LABELS["mp"]),
    (DEVICE_LABELS["mp"], "CMF kernel"),
)


#: what both tables of pairs say about the three pairs added later
ADDED = ("Rows 8 to 10 hold the learner fixed and change only the input. They were added after "
         "the first COMPAS seed had been read; the other rows were fixed before any run.")


def _label(key):
    if " | " in key:
        inp, learner = key.split(" | ")
        return f"{inp}, {LEARNER_LABELS[learner]}"
    return key


def write_table(results=None):
    """The table of the item, from the files of the stage ``models`` in the folder `results`."""
    global RESULTS
    RESULTS = results or RESULTS
    path = os.path.join(RESULTS, "table.md")
    L = ["# CMF as a featuriser (plan item ND1)", "",
         "Test rows of a random split into 70 % training, 15 % validation and 15 % test, the "
         "split of the device's own protocol; every setting is chosen on the validation rows. "
         "R2 is given as the mean and the standard deviation over the seeds.", ""]
    done = {name: collect(name) for name in SETS}
    runs = done["compas-1D"]
    if runs:
        keys = [k for k in runs[0]["models"]]
        L += [f"## COMPAS-1D: the gap of {runs[0]['n']} polybenzenoid hydrocarbons, "
              f"{len(runs)} seeds", "",
              f"{runs[0]['n_train']} training, {runs[0]['n_valid']} validation and "
              f"{runs[0]['n_test']} test molecules. The CMF vector has "
              f"{runs[0]['chosen']['CMF kernel']['landmarks']} landmark molecules "
              f"(rank {runs[0]['chosen']['CMF kernel']['rank']}).", "",
              "| Model | Test R2 | Test RMSE (eV) | Test MAE (eV) |", "|---|---|---|---|"]
        for k in keys:
            L.append(f"| {_label(k)} | {_pm(runs, k)} | {_pm(runs, k, 'rmse')} | "
                     f"{_pm(runs, k, 'mae')} |")
        L += ["", "### Paired differences of the absolute error on COMPAS-1D (eV)", "",
              "A negative value means the first model has the smaller error. The interval is a "
              "95 % bootstrap interval over the test molecules of all seeds. " + ADDED, "",
              "| First model | Second model | Difference |", "|---|---|---|"]
        for a, b in PAIRS:
            L.append(f"| {_label(a)} | {_label(b)} | {_pair(paired(runs, a, b))} |")
        L.append("")
    have = [s for s in SUTHERLAND if done[s]]
    if have:
        keys = [k for k in done[have[0]][0]["models"]]
        L += [f"## The Sutherland series, {len(done[have[0]])} seeds each", "",
              "Test R2, the mean over the seeds. The last column is the mean over the series.",
              "", "| Model | " + " | ".join(have) + " | Mean |",
              "|---|" + "---|" * (len(have) + 1)]
        for k in keys:
            v = [np.mean([r["models"][k]["r2"] for r in done[s]]) for s in have]
            L.append(f"| {_label(k)} | " + " | ".join(f"{x:+.3f}" for x in v)
                     + f" | {np.mean(v):+.3f} |")
        L += ["", "Molecules: " + ", ".join(f"{s} {done[s][0]['n']}" for s in have) + ".", "",
              "### Paired differences of the absolute error on the Sutherland series", "",
              "In the units of each activity (pKi or pIC50). A negative value means the first "
              "model has the smaller error. " + ADDED, "",
              "| First model | Second model | " + " | ".join(have)
              + " | Series where the interval excludes zero |",
              "|---|---|" + "---|" * (len(have) + 1)]
        for a, b in PAIRS:
            ps = [paired(done[s], a, b) for s in have]
            better = sum(p["excludes_zero"] and p["mean"] < 0 for p in ps)
            worse = sum(p["excludes_zero"] and p["mean"] > 0 for p in ps)
            L.append(f"| {_label(a)} | {_label(b)} | "
                     + " | ".join(f"{p['mean']:+.3f}" + ("" if p["excludes_zero"] else " (0)")
                                  for p in ps)
                     + f" | first better in {better}, second better in {worse} |")
        L += ["", "(0) marks a difference whose 95 % interval includes zero.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    return path


STAGES = ("tables", "device", "reference", "models")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=STAGES, action="append")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--device-root", default=None, help="the checkout of the device project")
    ap.add_argument("--device-python", default=sys.executable,
                    help="the Python of the device's environment (default: this one)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4, help="threads of a forest")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="only rewrite the table")
    a = ap.parse_args(argv)
    sets = tuple(a.sets) if a.sets else SETS
    if not a.report:
        for stage in (a.stage or ["models"]):
            if stage == "tables":
                tables_stage()
            elif stage == "device":
                if not a.device_root:
                    raise SystemExit("--stage device needs --device-root, the checkout of the "
                                     "device project")
                device_stage(a.device_root, a.device_python, a.workers, sets, a.gpu)
            elif stage == "reference":
                reference_stage(sets, a.seeds)
            else:
                models_stage(sets, a.seeds, a.jobs, a.force)
    print("wrote", os.path.relpath(write_table(), ROOT))


if __name__ == "__main__":
    main()
