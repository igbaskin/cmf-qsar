#!/usr/bin/env python
"""
The co-field as an atom feature of a network that sums over the atoms: plan item ND14c.

A linear CMF model has a regression-coefficient field (the co-field) for every field type, and a
ridge regression fits it.  Item ND14c keeps the co-fields and changes the last step: the
co-field of every field is read at every atom, in the common frame of the aligned molecules,
and the values become atom features of a network that reads every atom with shared weights,
sums over the atoms and maps the sum to the activity with a second network.  The question is
whether this change of the fitting method lowers the error of the ridge regression on the same
fields.

Two such networks are measured.

* **The atom network** of this program (``cmfqsar/device.py``): one network on every atom, a
  sum, a second network.  It sees no bonds; an atom is its features.
* **The device**: the QSPR Neural Device (Baskin, Palyulin, Zefirov, J. Chem. Inf. Comput. Sci.
  37 (1997) 715; package ``neurodevice``), the separate project that plan item ND1 ran.  It
  reads atoms, bonds and paths of two bonds of the molecular graph.  A sensor of the device is
  one number per atom; the co-field values enter as further atom sensors
  (``benchmarks/featuriser_device.py --atom-features``).

What is read, and from which model
----------------------------------
The value at an atom is the overlap of the co-field with the Gaussian that the field puts on
the atom, for a weight of one (``cmfqsar.device.atom_cofield``).  The linear model is the one of
plan item ND2 without any training: the attenuations and the ridge that the grid search of ND1
chose for the split, 1,000 landmark molecules on COMPAS-1D and every training molecule on a
Sutherland series.  A training molecule reads the co-field of a model that was fitted without
its fold (five folds inside the training part), so that its features do not hold its own
activity; a validation or test molecule reads the co-field of the whole training part.  One row
of the table reads the co-field of the whole training part for the training molecules too, to
show what the folds are worth.

The sets and the splits are those of ND1 and ND2: COMPAS-1D (8,678 polybenzenoid hydrocarbons,
the HOMO-LUMO gap, nine fields, five seeds) and the eight Sutherland series (five fields, ten
seeds each); random 70 % training, 15 % validation, 15 % test, read from the device files of
ND1.  The atoms that are read are the non-hydrogen atoms.

The rows
--------
    ridge regression on the fields            the linear model whose co-field is read
    atom network: field weights of the atom   the weights of the atom in the fields, no place
    atom network: positions                   the weights and the coordinates in the frame
    atom network: co-field                    the weights and the co-field of every field
    atom network: co-field, not out-of-fold   the same, training molecules read the full model
    atom network: co-field and positions      the weights, the co-field and the coordinates
    device, one-shot: with the co-field       the device with the co-field as atom sensors
    device, one-shot: with positions          the device with the coordinates as atom sensors

and, read from the result files of ND1: the device without these sensors, the kernel model on
every training molecule and the best model of ND1.  "One-shot" is the device's name for its
network without message passing.  The sizes of the atom network were fixed before any run: two
hidden layers of 64 units on the atom, 16 sums (collectors), one hidden layer of 64 units after
the sum.  The learning rate (0.003), the batch (128 molecules) and the patience of the early
stopping (150 epochs) were chosen on the validation molecules of the first COMPAS-1D split.

What was not run
----------------
The device with message passing was not trained with the further sensors (``--methods mp``
does it).  The device is fast because many atoms have the same sensor values, and it computes
each distinct row once.  A co-field value or a coordinate is different on every atom, so this
saving is lost: one run without message passing took 1.5 to several hours where ND1 needed 2.5
to 10 minutes, and a run with message passing would take days.  Comparisons 9 and 10 below
therefore have no numbers.

The comparisons, fixed before any run
-------------------------------------
Paired differences of the absolute error on the same test molecules, 95 % bootstrap interval:

    1  atom network: co-field against the ridge regression on the fields     the gate of the item
    2  atom network: co-field against field weights of the atom    what the co-field adds
    3  atom network: co-field against positions                    the co-field or the bare place
    4  atom network: co-field and positions against positions      what the co-field adds to them
    5  atom network: co-field, not out-of-fold against co-field    what the folds are worth
    6  device, one-shot: with the co-field against the ridge regression on the fields
    7  device, one-shot: with the co-field against the device, one-shot
    8  device, one-shot: with the co-field against with positions
    9  device, message passing: with the co-field against the device, message passing
    10 device, message passing: with the co-field against with positions

Usage
-----
    python benchmarks/run_device.py --stage features     # needs PyTorch; uses a GPU if any
    python benchmarks/run_device.py --stage own
    python benchmarks/run_device.py --stage device --device-root C:/path/to/the/device
    python benchmarks/run_device.py --report

The stage ``features`` needs the packed atoms of ND2 (``run_learnable.py --stage pack``) and
the device files of ND1, which are in the repository.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar import device as D                                             # noqa: E402
from cmfqsar import learnable as L                                          # noqa: E402

import run_learnable as RL                                                  # noqa: E402

RESULTS = os.path.join(HERE, "results", "device")
BPZ = os.path.join(RESULTS, "device")
CACHE = os.path.join(HERE, "cache", "device")
ND1_TABLES = os.path.join(HERE, "cache", "featuriser")
SETS = RL.SETS
SEEDS = dict(RL.SEEDS)
SEEDS["compas-1D"] = tuple(range(5))
FOLDS = 5
RIDGE = "ridge regression on the fields"
#: the atom network: label -> the blocks of atom features
OWN = {
    "atom network: field weights of the atom": ("weights",),
    "atom network: positions": ("weights", "xyz"),
    "atom network: co-field": ("weights", "oof"),
    "atom network: co-field, not out-of-fold": ("weights", "full"),
    "atom network: co-field and positions": ("weights", "oof", "xyz"),
}
#: the sizes were fixed before any run; the rate, the batch and the patience were chosen on
#: the validation rows of the first COMPAS-1D split (rates 0.001 and 0.003, batches 128 to 512)
NETWORK = {"hidden": (64, 64), "collectors": 16, "brain": (64,), "l2": 1e-4, "epochs": 3000,
           "batch": 128, "lr": 3e-3, "patience": 150, "lr_decay": 0.5}
DEVICE_METHODS = ("oneshot", "mp")
DEVICE_NAMES = {"oneshot": "device, one-shot", "mp": "device, message passing"}
DEVICE_INPUTS = {"cofield": "with the co-field", "positions": "with positions"}
#: (number, model, reference, what the comparison says)
PAIRS = (
    (1, "atom network: co-field", RIDGE, "the gate: the atom network against the ridge "
        "regression on the same fields"),
    (2, "atom network: co-field", "atom network: field weights of the atom",
     "what the co-field adds to the atom network"),
    (3, "atom network: co-field", "atom network: positions",
     "the co-field against the bare coordinates"),
    (4, "atom network: co-field and positions", "atom network: positions",
     "what the co-field adds to the coordinates"),
    (5, "atom network: co-field, not out-of-fold", "atom network: co-field",
     "what the out-of-fold reading is worth (positive: it helps)"),
    (6, "device, one-shot: with the co-field", RIDGE,
     "the gate with the device without message passing"),
    (7, "device, one-shot: with the co-field", "device, one-shot",
     "what the co-field adds to the device without message passing"),
    (8, "device, one-shot: with the co-field", "device, one-shot: with positions",
     "the co-field against the bare coordinates, device without message passing"),
    (9, "device, message passing: with the co-field", "device, message passing",
     "what the co-field adds to the device with message passing"),
    (10, "device, message passing: with the co-field", "device, message passing: with positions",
     "the co-field against the bare coordinates, device with message passing"),
)
ND1_ROWS = (("device, one-shot", "ND1: the device without message passing"),
            ("device, message passing", "ND1: the device with message passing"),
            ("CMF kernel", "ND1: the kernel model on every training molecule"),
            ("CMF vector + collectors, message passing | network",
             "ND1: multilayer network on the CMF vector joined with the collectors"))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------- the atoms that are read
def points_file(name):
    return os.path.join(CACHE, f"{name}_points.npz")


def smiles_order(mol):
    """(SMILES, atoms of `mol` in the order in which RDKit reads that SMILES, their degrees).

    The SMILES is the one the device table of ND1 holds.  The order is checked: every bond of
    the molecule that is read back must join the same two atoms of `mol`.
    """
    from rdkit import Chem
    mh = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mh)
    order = [int(v) for v in re.findall(r"\d+", mh.GetProp("_smilesAtomOutputOrder"))]
    back = Chem.MolFromSmiles(smiles)
    if back is None or back.GetNumAtoms() != mh.GetNumAtoms() or len(order) != mh.GetNumAtoms():
        raise ValueError("the SMILES does not give the molecule back")
    for bond in back.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mh.GetBondBetweenAtoms(order[i], order[j]) is None:
            raise ValueError("the atom order of the SMILES is not the one that was expected")
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if len(heavy) != mh.GetNumAtoms():
        raise ValueError("a hydrogen atom was kept; the atom order is not known")
    return smiles, [heavy[k] for k in order], [back.GetAtomWithIdx(i).GetDegree()
                                               for i in range(back.GetNumAtoms())]


def points_stage(sets=SETS, force=False):
    """The non-hydrogen atoms of every molecule: place, radius, field weights, bond count."""
    import run_featuriser as RF
    os.makedirs(CACHE, exist_ok=True)
    for name in sets:
        if os.path.exists(points_file(name)) and not force:
            continue
        names, mols, _y = RF.load_set(name)
        table = RL.build_table(name)
        coords, W = np.asarray(table.coords, float), np.asarray(table.weights, float)
        radii = np.asarray(table.radii, float)
        with open(os.path.join(ND1_TABLES, f"{name}.csv"), encoding="utf-8") as fh:
            listed = {r["name"]: r["smiles"] for r in csv.DictReader(fh)}
        rows = []
        for i, (nm, mol) in enumerate(zip(names, mols)):
            smiles, order, degree = smiles_order(mol)
            if listed.get(nm) != smiles:
                raise SystemExit(f"{name}: the SMILES of {nm} is not the one of the device "
                                 "table; run run_featuriser.py --stage tables")
            sl = table.slice(i)
            xyz = mol.GetConformer().GetPositions()
            if np.abs(coords[sl][:mol.GetNumAtoms()] - xyz).max() > 1e-6:
                raise SystemExit(f"{name}: the atoms of {nm} are not the first rows of its "
                                 "table")
            at = sl.start + np.asarray(order, int)
            rows.append((coords[at], radii[at], W[at], np.asarray(degree, int)))
        A = max(len(r[0]) for r in rows)
        n, F = len(rows), W.shape[1]
        out = {"xyz": np.zeros((n, A, 3)), "radii": np.ones((n, A)),
               "weights": np.zeros((n, A, F)), "mask": np.zeros((n, A), bool),
               "degree": np.zeros((n, A), int)}
        for i, (x, r, w, d) in enumerate(rows):
            k = len(x)
            out["xyz"][i, :k], out["radii"][i, :k], out["weights"][i, :k] = x, r, w
            out["degree"][i, :k], out["mask"][i, :k] = d, True
        np.savez_compressed(points_file(name), names=np.array(names), **out)
        log(f"  {name}: {n} molecules, at most {A} non-hydrogen atoms, {int(out['mask'].sum())} "
            "in all")


def load_points(name):
    z = np.load(points_file(name), allow_pickle=False)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ the co-field at the atoms
def result_file(name, seed):
    return os.path.join(RESULTS, f"{name}_{seed}.json")


def atoms_file(name, seed):
    return os.path.join(CACHE, f"{name}_{seed}_atoms.npz")


def sensors_file(name, seed, what):
    return os.path.join(CACHE, f"{name}_{seed}_sensors_{what}.npz")


def sensor_name(field):
    return "COF_" + re.sub(r"[^A-Za-z0-9]+", "_", field).strip("_").upper()


def write_sensors(path, pts, sensors, values):
    """The file ``featuriser_device.py --atom-features`` reads: one row per real atom."""
    mask = pts["mask"]
    np.savez_compressed(path, names=pts["names"], counts=mask.sum(1).astype(np.int32),
                        degree=pts["degree"][mask].astype(np.int16),
                        sensors=np.array(sensors), values=values[mask].astype(np.float32))


def sum_check(packed, M, rows, pred, device=None, dtype=None):
    """The largest difference between the prediction of the linear model and the sum of the
    co-field values times the weights of the atoms, over the molecules `rows`."""
    total = np.full(len(rows), float(M.bias[-1]))
    for f in range(len(packed["names"])):
        w = packed["w"][f][rows]
        v = D.atom_cofield(packed, M, rows, packed["xyz"][f][rows], mask=w != 0.0,
                           radii=1.0 / np.sqrt(packed["c"][f][rows]), molecule_norm=True,
                           device=device, dtype=dtype)
        total += (v[:, :, f] * w).sum(axis=1)
    return float(np.abs(total - pred).max())


def features_stage(sets=SETS, seeds=None, device=None, force=False):
    import torch
    # double precision also on a GPU: a co-field value is a sum of large terms of both signs
    f64 = torch.float64
    os.makedirs(RESULTS, exist_ok=True)
    points_stage(sets)
    for name in sets:
        todo = [k for k in (seeds if seeds is not None else SEEDS[name])
                if force or not (os.path.exists(atoms_file(name, k))
                                 and os.path.exists(result_file(name, k)))]
        if not todo:
            continue
        packed, y, molecules, groups = RL.load_atoms(name)
        pts = load_points(name)
        if [str(v) for v in pts["names"]] != molecules:
            raise SystemExit(f"{name}: the atoms and the packed fields list other molecules")
        everyone = np.arange(len(y))
        for seed in todo:
            t0 = time.time()
            split = RL.nd1_split(name, seed, molecules)["split"]
            tr, va, te = split["train"], split["valid"], split["test"]
            land = RL.landmarks_of(name, seed, tr)
            start = RL.start_of(name, seed, groups, len(tr))
            M = D.fold_models(packed, y, tr, land, alpha=start["alpha"], h=start["h"],
                              lam=start["lam"], normalize=start["normalize"], folds=FOLDS,
                              seed=seed, device=device, dtype=f64)
            pred = np.full(len(y), np.nan)
            rest = np.concatenate([va, te])
            pred[rest] = L._predict_rows(packed, rest, land, M.alpha, M.h, M.normalize, None,
                                         None, 1.0, M.coef[-1], float(M.bias[-1]),
                                         L._device(device), f64, L.BUDGET)
            pred[tr] = M.fit
            kw = dict(mask=pts["mask"], radii=pts["radii"], device=device, dtype=f64)
            full = D.atom_cofield(packed, M, everyone, pts["xyz"], **kw)
            oof = D.atom_cofield(packed, M, everyone, pts["xyz"],
                                 version=M.versions(everyone), **kw)
            some = te[:200]
            check = sum_check(packed, M, some, pred[some], device, f64)
            np.savez_compressed(atoms_file(name, seed), full=full.astype(np.float32),
                                oof=oof.astype(np.float32), fields=np.array(packed["names"]),
                                fold_of=M.fold_of, train=tr)
            write_sensors(sensors_file(name, seed, "cofield"), pts,
                          [sensor_name(f) for f in packed["names"]], oof)
            write_sensors(sensors_file(name, seed, "positions"), pts,
                          ["POS_X", "POS_Y", "POS_Z"], pts["xyz"])

            def rmse(a, b):
                return float(np.sqrt(np.mean((a - b) ** 2)))
            moved = full[tr][pts["mask"][tr]] - oof[tr][pts["mask"][tr]]
            out = {"set": name, "seed": seed, "n": len(y), "n_train": len(tr),
                   "n_valid": len(va), "n_test": len(te), "test": te.tolist(),
                   "fields": packed["names"], "landmarks": int(len(land)),
                   "settings": {"alpha": [float(a) for a in M.alpha],
                                "h": [float(a) for a in M.h], "lam": float(M.lam),
                                "normalize": M.normalize, "folds": int(M.folds)},
                   "models": {RIDGE: RL.scores(y, pred, te)},
                   "check": {
                       "sum_of_atoms_max_difference": check,
                       "train_rmse_fitted": rmse(M.fit, y[tr]),
                       "train_rmse_out_of_fold": rmse(M.cv, y[tr]),
                       "cofield_sd_full": float(full[tr][pts["mask"][tr]].std()),
                       "cofield_rms_full_minus_out_of_fold": float(np.sqrt((moved ** 2).mean()))},
                   "seconds": {"features": time.time() - t0}}
            if os.path.exists(result_file(name, seed)) and not force:
                with open(result_file(name, seed), encoding="utf-8") as fh:
                    old = json.load(fh)
                old["models"].update(out["models"])
                old["check"], old["settings"] = out["check"], out["settings"]
                old.setdefault("seconds", {})["features"] = out["seconds"]["features"]
                out = old
            with open(result_file(name, seed), "w", encoding="utf-8") as fh:
                json.dump(out, fh)
            log(f"  {name} seed {seed}: ridge test MAE {out['models'][RIDGE]['mae']:.4f}, "
                f"R2 {out['models'][RIDGE]['r2']:+.3f}; training RMSE fitted "
                f"{out['check']['train_rmse_fitted']:.4f}, out-of-fold "
                f"{out['check']['train_rmse_out_of_fold']:.4f}; sum check {check:.1e} "
                f"({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- the atom network
def own_stage(sets=SETS, seeds=None, device=None, force=False, models=None):
    for name in sets:
        pts = y = molecules = None
        for seed in (seeds if seeds is not None else SEEDS[name]):
            if not os.path.exists(result_file(name, seed)):
                continue
            with open(result_file(name, seed), encoding="utf-8") as fh:
                out = json.load(fh)
            todo = [m for m in (models or OWN) if force or m not in out["models"]]
            if not todo:
                continue
            if pts is None:
                pts = load_points(name)
                _packed, y, molecules, _groups = RL.load_atoms(name)
            split = RL.nd1_split(name, seed, molecules)["split"]
            tr, va, te = split["train"], split["valid"], split["test"]
            z = np.load(atoms_file(name, seed), allow_pickle=False)
            blocks = {"weights": pts["weights"], "xyz": pts["xyz"],
                      "oof": np.asarray(z["oof"], float), "full": np.asarray(z["full"], float)}
            for label in todo:
                X = np.concatenate([blocks[b] for b in OWN[label]], axis=2)
                res = D.train_device(X, pts["mask"], y, tr, va, seed=seed, device=device,
                                     **NETWORK)
                out["models"][label] = RL.scores(y, res.pred, te)
                out.setdefault("trained", {})[label] = {
                    "features": int(X.shape[2]), "epochs": int(res.epochs),
                    "valid_rmse": float(res.valid_rmse), "seconds": float(res.seconds)}
                log(f"    {name} seed {seed}: {label}: test R2 "
                    f"{out['models'][label]['r2']:+.3f}, MAE {out['models'][label]['mae']:.4f}, "
                    f"{res.epochs} epochs, {res.seconds:.0f} s")
                with open(result_file(name, seed), "w", encoding="utf-8") as fh:
                    json.dump(out, fh)


# ---------------------------------------------------------------------------------- the device
def device_file(name, method, what, seed):
    return os.path.join(BPZ, f"{name}_{method}_{what}_{seed}.npz")


def device_stage(device_root, device_python=None, workers=4, sets=("compas-1D",), seeds=None,
                 methods=DEVICE_METHODS, inputs=("cofield",), backend=None, stagger=0.0):
    """Train the device with the further atom sensors, for every run that has no file yet.

    A run needs most memory while it prepares its input (2.6 GB on COMPAS-1D, 1 GB after
    that), so `stagger` seconds are left between two starts.
    """
    os.makedirs(BPZ, exist_ok=True)
    jobs = [(s, m, w, k) for m in methods for w in inputs for s in sets
            for k in (seeds if seeds is not None else SEEDS[s])
            if not os.path.exists(device_file(s, m, w, k))
            and os.path.exists(sensors_file(s, k, w))]
    log(f"{len(jobs)} device runs with {workers} processes")
    script = os.path.join(HERE, "featuriser_device.py")
    running = []
    while jobs or running:
        while jobs and len(running) < workers:
            s, m, w, k = jobs.pop(0)
            cmd = [device_python or sys.executable, script, "--device-root", device_root,
                   "--table", os.path.join(ND1_TABLES, f"{s}.csv"), "--method", m,
                   "--seed", str(k), "--out", device_file(s, m, w, k),
                   "--atom-features", sensors_file(s, k, w)]
            if backend:
                cmd += ["--backend", backend]
            running.append((subprocess.Popen(cmd), s, m, w, k))
            time.sleep(float(stagger))
        time.sleep(2.0)
        still = []
        for proc, s, m, w, k in running:
            if proc.poll() is None:
                still.append((proc, s, m, w, k))
            elif proc.returncode != 0:
                log(f"  FAILED {s} {m} {w} seed {k} (exit {proc.returncode})")
        running = still


def read_device(name, seed, y, test):
    """{label: scores} of the device runs of one split, with what the device used."""
    out, used = {}, {}
    for m in DEVICE_METHODS:
        for w, tail in DEVICE_INPUTS.items():
            path = device_file(name, m, w, seed)
            if not os.path.exists(path):
                continue
            z = np.load(path, allow_pickle=False)
            if not np.array_equal(np.asarray(z["test"], int), np.asarray(test, int)):
                raise SystemExit(f"{path} holds another split than ND1")
            label = f"{DEVICE_NAMES[m]}: {tail}"
            out[label] = RL.scores(y, np.asarray(z["pred"], float), np.asarray(test, int))
            used[label] = {"atom_sensors": [str(v) for v in z["atom_sensors"]],
                           "added": [str(v) for v in z["added_sensors"]],
                           "seconds": float(z["seconds"])}
    return out, used


# ---------------------------------------------------------------------------------- the report
def collect(name):
    runs = []
    y = None
    for seed in SEEDS[name]:
        if not os.path.exists(result_file(name, seed)):
            continue
        with open(result_file(name, seed), encoding="utf-8") as fh:
            r = json.load(fh)
        if y is None:
            y = RL.load_atoms(name)[1]
        models, used = read_device(name, seed, y, r["test"])
        r["models"].update(models)
        r["device"] = used
        nd1 = RL.nd1_result(name, seed)
        if nd1["test"] != r["test"]:
            raise SystemExit(f"{name} seed {seed}: the test rows differ from those of ND1")
        for key, _text in ND1_ROWS:
            if key in nd1["models"]:
                r["models"].setdefault(key, nd1["models"][key])
        runs.append(r)
    return runs


def _row(runs, key, text=None):
    have = [r for r in runs if key in r["models"]]
    if not have:
        return None
    mae = np.array([r["models"][key]["mae"] for r in have])
    r2 = np.array([r["models"][key]["r2"] for r in have])
    return (f"| {text or key} | {mae.mean():.3f} ± {mae.std():.3f} | {r2.mean():+.3f} ± "
            f"{r2.std():.3f} | {len(have)} |")


def write_table():
    lines = ["# The co-field as an atom feature of a network that sums over the atoms (ND14c)",
             "",
             "Written by `benchmarks/run_device.py --report`. The splits are those of plan "
             "items ND1 and ND2: random, 70 % training, 15 % validation, 15 % test. Every "
             "score is on the test molecules. \"Atom network\" is the network of "
             "`cmfqsar/device.py`: one network on every atom, a sum over the atoms, a second "
             "network; it sees no bonds. \"Device\" is the QSPR Neural Device, which reads the "
             "atoms, the bonds and the paths of two bonds. The co-field is that of the ridge "
             "regression of the first row; a training molecule reads the co-field of a model "
             "fitted without its fold, unless the row says \"not out-of-fold\".", ""]
    order = [RIDGE] + list(OWN) + [f"{DEVICE_NAMES[m]}: {t}" for m in DEVICE_METHODS
                                   for t in DEVICE_INPUTS.values()]
    for name in ("compas-1D",):
        runs = collect(name)
        if not runs:
            continue
        unit = "eV"
        lines += [f"## COMPAS-1D: the HOMO-LUMO gap, {len(runs)} splits", "",
                  f"| Model | Mean absolute error, {unit} (mean ± sd over the splits) | "
                  "Test R2 (mean ± sd) | Splits |", "|---|---|---|---|"]
        lines += [row for row in (_row(runs, k) for k in order) if row]
        lines += [row for row in (_row(runs, k, t) for k, t in ND1_ROWS) if row]
        lines += ["", "### Paired comparisons on the same test molecules", "",
                  "The difference is the mean absolute error of the first model minus that of "
                  "the second, in eV; a negative number means that the first model is better. "
                  "The interval is the 95 % bootstrap interval over the test molecules.", "",
                  "| No. | First model | Second model | Difference [95 % interval] | Splits | "
                  "What it says |", "|---|---|---|---|---|---|"]
        for no, a, b, text in PAIRS:
            have = [r for r in runs if a in r["models"] and b in r["models"]]
            if have:
                lines.append(f"| {no} | {a} | {b} | {RL._pair(RL.paired(have, a, b), 4)} | "
                             f"{len(have)} | {text} |")
        lines += ["", "### Checks of the co-field values", "",
                  "| Split | Largest difference between the prediction and the sum over the "
                  "atoms, eV | Training RMSE of the fitted model, eV | Training RMSE "
                  "out-of-fold, eV | Root mean square change of a co-field value between the "
                  "full model and the fold model, as a share of the spread of the values |",
                  "|---|---|---|---|---|"]
        for r in runs:
            c = r["check"]
            share = c["cofield_rms_full_minus_out_of_fold"] / max(c["cofield_sd_full"], 1e-300)
            lines.append(f"| {r['seed']} | {c['sum_of_atoms_max_difference']:.1e} | "
                         f"{c['train_rmse_fitted']:.4f} | {c['train_rmse_out_of_fold']:.4f} | "
                         f"{share:.3f} |")
        times = {}
        for r in runs:
            for label, t in r.get("trained", {}).items():
                times.setdefault(label, []).append(t["seconds"])
            for label, t in r.get("device", {}).items():
                times.setdefault(label, []).append(t["seconds"])
        if times:
            lines += ["", "### Training time", "", "| Model | Mean time for one split, minutes |",
                      "|---|---|"]
            lines += [f"| {k} | {np.mean(v) / 60.0:.1f} |" for k, v in times.items()]
        used = {}
        for r in runs:
            for label, t in r.get("device", {}).items():
                used.setdefault(label, set()).add(
                    (len(t["atom_sensors"]), sum(s in t["atom_sensors"] for s in t["added"]),
                     len(t["added"])))
        if used:
            lines += ["", "### Atom sensors of the device", "",
                      "| Model | Atom sensors in use | Of them read from the file | Offered "
                      "in the file |", "|---|---|---|---|"]
            for label, v in used.items():
                for total, taken, offered in sorted(v):
                    lines.append(f"| {label} | {total} | {taken} | {offered} |")
        lines.append("")
    series = [s for s in SETS if s != "compas-1D"]
    rows = {s: collect(s) for s in series}
    if any(rows.values()):
        keys = [RIDGE] + list(OWN)
        lines += ["## The eight Sutherland series: test R2, mean over the splits", "",
                  "| Series | Splits | " + " | ".join(keys) + " |",
                  "|---|---|" + "---|" * len(keys)]
        means = {k: [] for k in keys}
        for s in series:
            runs = rows[s]
            if not runs:
                continue
            cells = []
            for k in keys:
                v = [r["models"][k]["r2"] for r in runs if k in r["models"]]
                cells.append(f"{np.mean(v):+.3f}" if v else "-")
                if v:
                    means[k].append(np.mean(v))
            lines.append(f"| {s} | {len(runs)} | " + " | ".join(cells) + " |")
        lines.append("| Mean over the series | | "
                     + " | ".join(f"{np.mean(v):+.3f}" if v else "-" for v in means.values())
                     + " |")
        lines += ["", "### Paired comparisons, Sutherland series", "",
                  "The difference of the absolute errors in the units of the activity "
                  "(pIC50 or pKi), first model minus second; negative means that the first "
                  "model is better.", "",
                  "| Series | " + " | ".join(f"{no}: {a} against {b}" for no, a, b, _t in PAIRS[:5])
                  + " |", "|---|" + "---|" * 5]
        for s in series:
            runs = rows[s]
            if not runs:
                continue
            cells = []
            for _no, a, b, _t in PAIRS[:5]:
                have = [r for r in runs if a in r["models"] and b in r["models"]]
                cells.append(RL._pair(RL.paired(have, a, b)) if have else "-")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        lines.append("")
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    return path


STAGES = ("features", "own", "device")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=STAGES, action="append")
    ap.add_argument("--report", action="store_true", help="write results/device/table.md")
    ap.add_argument("--sets", nargs="+", default=None, choices=SETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--torch-device", default=None, help="cpu or cuda (default: cuda if any)")
    ap.add_argument("--device-root", default=None, help="the checkout of the device project")
    ap.add_argument("--device-python", default=None,
                    help="the Python that runs the device (default: this one)")
    ap.add_argument("--workers", type=int, default=4, help="device runs at the same time")
    ap.add_argument("--methods", nargs="+", default=list(DEVICE_METHODS), choices=DEVICE_METHODS)
    ap.add_argument("--inputs", nargs="+", default=["cofield"], choices=list(DEVICE_INPUTS))
    ap.add_argument("--backend", default=None, choices=("numpy", "torch-cpu", "torch-cuda"),
                    help="where the device computes, in place of its own choice")
    ap.add_argument("--stagger", type=float, default=0.0,
                    help="seconds between the starts of two device runs")
    a = ap.parse_args(argv)
    sets = tuple(a.sets) if a.sets else SETS
    for stage in a.stage or []:
        if stage == "features":
            features_stage(sets, a.seeds, a.torch_device, a.force)
        elif stage == "own":
            own_stage(sets, a.seeds, a.torch_device, a.force)
        else:
            if not a.device_root:
                ap.error("--stage device needs --device-root")
            device_stage(a.device_root, a.device_python, a.workers,
                         tuple(a.sets) if a.sets else ("compas-1D",), a.seeds,
                         tuple(a.methods), tuple(a.inputs), a.backend, a.stagger)
    if a.report or not a.stage:
        log("wrote " + os.path.relpath(write_table(), ROOT))


if __name__ == "__main__":
    main()
