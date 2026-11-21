#!/usr/bin/env python
"""
A learnable CMF layer: plan item ND2.

A CMF model fixes the attenuation of every field (the width of the Gaussian on every atom) and
the weight of every field in the joint kernel before it sees an activity.  The program chooses
the attenuation from a grid and leaves the weights equal.  ``cmfqsar/learnable.py`` makes both
continuous and trains them by gradient, with the ridge.  Item ND2 asks for exactly this, and
asks that the co-field stays exportable.  This script measures what the training gains.

The sets and the splits are those of plan item ND1 (``benchmarks/run_featuriser.py``):
COMPAS-1D (8,678 polybenzenoid hydrocarbons, the HOMO-LUMO gap, nine fields in the five groups
ABCDE, 1,000 landmark molecules, five seeds) and the eight Sutherland series (five fields,
every training molecule a landmark, ten seeds each).  The split is random, 70 % training, 15 %
validation, 15 % test, read from the device files of ND1.  The start of every training is the
setting the grid search of ND1 chose for the same split, read from the result files of ND1.

The models with a closed-form ridge (``--stage ridge``)
-------------------------------------------------------
Every row is the kernel model over the landmarks.  What differs is what was trained:

    grid                      nothing: the attenuations and the ridge of the grid search
    ridge                     the ridge only, as a continuous number
    weights                   the field weights and the ridge
    widths                    the attenuation of every field and the ridge
    widths + weights          all three: the learnable layer of the plan
    widths + weights, to the minimum   the same without early stopping

and, on COMPAS-1D, with the Gaussian kernel layer on top: ``grid, layer`` (nothing trained),
``ridge, layer`` (the ridge and the layer width) and ``widths + weights, layer`` (everything).
The criterion is the squared error of five-fold out-of-fold predictions inside the training
part.  With early stopping, the evaluation with the lowest error on the validation rows is
kept; "to the minimum" keeps the lowest criterion.  The test rows are used for the scores only.

A network on the layer (``--stage network``, COMPAS-1D)
-------------------------------------------------------
A multilayer network on the kernel rows of the layer, trained by stochastic gradient descent,
with the layer fixed (the control) or trained with the network; and the same with the 15
collectors of the neural device with message passing joined to the input (the collectors are
those stored by ND1 and do not change).  The network has the size ND1 chose most often.

The comparisons, fixed before any run
-------------------------------------
Paired differences of the absolute error on the same test molecules, 95 % bootstrap interval:

    1  ridge against grid                           what a continuous ridge alone gains
    2  weights against ridge                        what the field weights gain
    3  widths against ridge                         what the attenuations gain
    4  widths + weights against widths              what the weights add to the attenuations
    5  widths + weights against grid                the layer against the grid search
    6  to the minimum against widths + weights      what early stopping is worth
    7  ridge, layer against grid, layer             (COMPAS) the same under the Gaussian layer
    8  widths + weights, layer against ridge, layer
    9  network, trained fields against network, fixed fields
    10 network + collectors, trained fields against network + collectors, fixed fields
    11 network + collectors, trained fields against the device with message passing (ND1)

Usage
-----
    python benchmarks/run_learnable.py --stage pack        # the atoms of every set, once
    python benchmarks/run_learnable.py --stage ridge       # needs PyTorch; uses a GPU if any
    python benchmarks/run_learnable.py --stage network
    python benchmarks/run_learnable.py --stage cofield     # the co-field of one trained model
    python benchmarks/run_learnable.py --report

The stages ``ridge`` and ``network`` need the result and device files of ND1 under
``benchmarks/results/featuriser/``, which are in the repository.
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

from cmfqsar import learnable as L                                          # noqa: E402

RESULTS = os.path.join(HERE, "results", "learnable")
CACHE = os.path.join(HERE, "cache", "learnable")
ND1 = os.path.join(HERE, "results", "featuriser")
SUTHERLAND = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
SETS = ("compas-1D",) + SUTHERLAND
SEEDS = {s: tuple(range(10)) for s in SUTHERLAND}
SEEDS["compas-1D"] = tuple(range(5))
COMPAS_GROUPS = "ABCDE"
LANDMARKS = 1000
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")
STEPS = 30                              # iterations of the quasi-Newton method
FOLDS = 5

#: the models with a closed-form ridge: label -> (what is trained, layer, selection)
RIDGE_MODELS = {
    "grid": ((), None, "valid"),
    "ridge": (("lam",), None, "valid"),
    "weights": (("weights", "lam"), None, "valid"),
    "widths": (("alpha", "lam"), None, "valid"),
    "widths + weights": (("alpha", "weights", "lam"), None, "valid"),
    "grid, layer": ((), "gauss", "valid"),
    "ridge, layer": (("lam", "gamma"), "gauss", "valid"),
    "widths + weights, layer": (("alpha", "weights", "lam", "gamma"), "gauss", "valid"),
}
LAYER_MODELS = tuple(k for k, v in RIDGE_MODELS.items() if v[1] == "gauss")
#: the model at the lowest criterion of the training of "widths + weights" (no early stopping)
TO_MINIMUM = "widths + weights, to the minimum"
#: the networks: label -> (what is trained in the layer, collectors joined, layers, L2)
NETWORK_MODELS = {
    "network, fixed fields": ((), False, (128, 64), 1e-4),
    "network, trained fields": (("alpha", "weights"), False, (128, 64), 1e-4),
    "network + collectors, fixed fields": ((), True, (128, 64), 1e-2),
    "network + collectors, trained fields": (("alpha", "weights"), True, (128, 64), 1e-2),
}
DEVICE_MP = "device, message passing"
PAIRS = (
    ("ridge", "grid"),
    ("weights", "ridge"),
    ("widths", "ridge"),
    ("widths + weights", "widths"),
    ("widths + weights", "grid"),
    ("widths + weights, to the minimum", "widths + weights"),
    ("ridge, layer", "grid, layer"),
    ("widths + weights, layer", "ridge, layer"),
    ("network, trained fields", "network, fixed fields"),
    ("network + collectors, trained fields", "network + collectors, fixed fields"),
    ("network + collectors, trained fields", DEVICE_MP),
)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def r2(y, p):
    return 1.0 - float(np.sum((y - p) ** 2)) / float(np.sum((y - y.mean()) ** 2))


def scores(y, pred, test):
    e = pred[test] - y[test]
    return {"r2": r2(y[test], pred[test]), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "abs_errors": np.abs(e).astype(np.float32).tolist()}


# ----------------------------------------------------------------------------------- the atoms
def atoms_file(name):
    return os.path.join(CACHE, f"{name}_atoms.npz")


def pack_stage(sets=SETS):
    """The atoms of every molecule of every set, one padded block per field."""
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    os.makedirs(CACHE, exist_ok=True)
    for name in sets:
        if name == "compas-1D":
            import run_compas as RC
            ds, meta = RC.load_prepared("1D")
            fs = RC.make_fieldset(COMPAS_GROUPS, meta)
            table = RC.build_table(ds, fs, ring_cache={},
                                   carbon_only=meta.get("carbon_only", True))
            groups = [g for g in COMPAS_GROUPS for _n in RC.group_fields(g, meta)]
        else:
            ds = load_dataset(os.path.join(ROOT, "examples", "sutherland", f"{name}.sdf"))
            fs = FieldSet.create(physchem=list(FIELDS), alpha=0.5, charge_model="file")
            table = AtomTable.from_molecules([build_fields(m, fs, n)
                                              for m, n in zip(ds.mols, ds.names)])
            groups = ["-"] * len(fs.specs)
        specs = [s for s in fs.specs if s.enabled]
        packed = L.pack(table, specs)
        L.save_packed(atoms_file(name), packed, y=np.asarray(ds.activity, float),
                      molecules=np.array(list(ds.names)), groups=np.array(groups))
        log(f"  {name}: {packed['n']} molecules, fields "
            + ", ".join(f"{n} ({w.shape[1]} atoms)" for n, w in zip(packed["names"], packed["w"])))


def load_atoms(name):
    packed, extra = L.load_packed(atoms_file(name))
    return (packed, np.asarray(extra["y"], float), [str(v) for v in extra["molecules"]],
            [str(v) for v in extra["groups"]])


# ------------------------------------------------------------------- the splits and the starts
def nd1_split(name, seed, molecules):
    """The index sets of ND1 for one split, and the collectors of the device."""
    out = {}
    for method in ("oneshot", "mp"):
        z = np.load(os.path.join(ND1, "device", f"{name}_{method}_{seed}.npz"),
                    allow_pickle=False)
        if [str(n) for n in z["names"]] != list(molecules):
            raise SystemExit(f"the device file of {name} seed {seed} lists other molecules")
        out[method] = np.asarray(z["collectors"], float)
        out["split"] = {k: np.asarray(z[k], int) for k in ("train", "valid", "test")}
    return out


def nd1_result(name, seed):
    with open(os.path.join(ND1, f"{name}_{seed}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def start_of(name, seed, groups, n_train_rows):
    """The start of a training: the settings the grid search of ND1 chose for this split."""
    chosen = nd1_result(name, seed)["chosen"]["CMF kernel"]
    F = len(groups)
    if name == "compas-1D":
        # the stored kernel of ND1 is the sum of the field kernels, each divided by its mean
        # self-kernel: weight 1 for every field under the "mean" convention
        return {"alpha": np.array([float(chosen["alphas"][g]) for g in groups]),
                "h": np.ones(F), "lam": float(chosen["lam"]), "gamma": float(chosen["gamma"]),
                "normalize": "mean"}
    # ND1 used the mean of the five cosine-normalised field kernels: weight 1/5 each
    return {"alpha": np.full(F, float(chosen["alpha"])), "h": np.full(F, 1.0 / F),
            "lam": float(chosen["lam"]), "gamma": 1.0, "normalize": "cosine"}


def landmarks_of(name, seed, train):
    """The landmark molecules: those of ND1 on COMPAS, every training molecule elsewhere."""
    if name != "compas-1D":
        return np.asarray(train, int)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(train, size=min(LANDMARKS, len(train)), replace=False))


# ------------------------------------------------------------------------- the closed-form rows
def ridge_file(name, seed):
    return os.path.join(RESULTS, f"{name}_{seed}.json")


def models_file(name, seed):
    """The linear models of one split (coefficients over the landmarks), for the co-fields."""
    return os.path.join(CACHE, f"{name}_{seed}_models.npz")


LINEAR_KEPT = ("grid", "widths + weights", TO_MINIMUM)


def ridge_models(name):
    return [k for k in RIDGE_MODELS if name == "compas-1D" or k not in LAYER_MODELS]


def ridge_stage(sets=SETS, seeds=None, device=None, force=False, steps=STEPS,
                verbose=False):
    os.makedirs(RESULTS, exist_ok=True)
    for name in sets:
        todo = [k for k in (seeds if seeds is not None else SEEDS[name])
                if force or not os.path.exists(ridge_file(name, k))]
        if not todo:
            continue
        packed, y, molecules, groups = load_atoms(name)
        for seed in todo:
            t0 = time.time()
            split = nd1_split(name, seed, molecules)["split"]
            tr, va, te = split["train"], split["valid"], split["test"]
            land = landmarks_of(name, seed, tr)
            start = start_of(name, seed, groups, len(tr))
            kept = {"landmarks": land}
            out = {"set": name, "seed": seed, "n": len(y), "n_train": len(tr),
                   "n_valid": len(va), "n_test": len(te), "test": te.tolist(),
                   "fields": packed["names"], "groups": groups, "landmarks": int(len(land)),
                   "start": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in start.items()}, "models": {}, "trained": {}}
            for label in ridge_models(name):
                learn, layer, select = RIDGE_MODELS[label]
                both = label == "widths + weights"
                res = L.train(packed, y, tr, land, valid=va, alpha=start["alpha"],
                              h=start["h"], lam=start["lam"],
                              gamma=start["gamma"] if layer else None, learn=learn,
                              normalize=start["normalize"], layer=layer, steps=steps,
                              folds=FOLDS, seed=seed, select=select, predict=te,
                              keep_criterion=both, device=device, log=log if verbose else None)
                cvs = [h["cv"] for h in res.history]
                for key, r, how in ((label, res, select),) + (
                        ((TO_MINIMUM, res.by_criterion, "cv"),) if both else ()):
                    out["models"][key] = scores(y, r.pred, te)
                    if key in LINEAR_KEPT:
                        for what in ("alpha", "h", "coef", "norms"):
                            kept[f"{key}|{what}"] = np.asarray(getattr(r, what), float)
                        kept[f"{key}|bias"] = np.array(r.bias)
                    out["trained"][key] = {
                        **r.settings(), "learned": list(r.learned), "select": how,
                        "cv_start": float(r.start["cv"]), "cv_lowest": float(min(cvs)),
                        "valid_rmse_start": float(r.start["valid_rmse"]),
                        "valid_rmse": float(min(h["valid_rmse"] for h in r.history))
                        if how == "valid" else float(r.history[int(np.argmin(cvs))]
                                                     ["valid_rmse"]),
                        "evaluations": len(r.history), "passes": int(r.passes),
                        "stop": str(r.start.get("stop", "")),
                        "seconds": float(r.seconds),
                        "kernel_seconds": float(r.kernel_seconds)}
                    log(f"    {name} seed {seed}: {key}: test R2 "
                        f"{out['models'][key]['r2']:+.3f}, MAE {out['models'][key]['mae']:.4f}, "
                        f"{len(r.history)} evaluations, {r.seconds:.0f} s")
            out["seconds"] = time.time() - t0
            np.savez_compressed(models_file(name, seed), **kept)
            with open(ridge_file(name, seed), "w", encoding="utf-8") as fh:
                json.dump(out, fh)
            log(f"  {name} seed {seed} done ({out['seconds']:.0f} s)")


# ------------------------------------------------------------------------------- the networks
def network_file(name, seed):
    return os.path.join(RESULTS, f"{name}_{seed}_network.json")


def scaled_block(X, tr, standardise=True):
    """One block of columns as ND1 scales it: centred on the training rows, the columns
    divided by their standard deviation there, the block by its root mean squared row length."""
    X = np.asarray(X, float)
    Xc = X - X[tr].mean(axis=0)
    if standardise:
        sd = Xc[tr].std(axis=0)
        Xc = Xc[:, sd > 1e-12] / sd[sd > 1e-12]
    return Xc / max(float(np.sqrt(np.mean(np.sum(Xc[tr] ** 2, axis=1)))), 1e-12)


def network_stage(sets=("compas-1D",), seeds=None, device=None, force=False, epochs=300,
                  models=None):
    os.makedirs(RESULTS, exist_ok=True)
    for name in sets:
        packed, y, molecules, groups = load_atoms(name)
        for seed in (seeds if seeds is not None else SEEDS[name]):
            path = network_file(name, seed)
            out = {"set": name, "seed": seed, "models": {}, "trained": {}}
            if os.path.exists(path) and not force:
                with open(path, encoding="utf-8") as fh:
                    out = json.load(fh)
            nd1 = nd1_split(name, seed, molecules)
            split = nd1["split"]
            tr, va, te = split["train"], split["valid"], split["test"]
            out["test"] = te.tolist()
            land = landmarks_of(name, seed, tr)
            start = start_of(name, seed, groups, len(tr))
            collectors = scaled_block(nd1["mp"], tr)
            for label in (models or NETWORK_MODELS):
                if label in out["models"]:
                    continue
                learn, joined, hidden, l2 = NETWORK_MODELS[label]
                res = L.train_network(packed, y, tr, va, land, alpha=start["alpha"],
                                      h=start["h"], extra=collectors if joined else None,
                                      learn=learn, normalize=start["normalize"], hidden=hidden,
                                      l2=l2, epochs=epochs, seed=seed, predict=te,
                                      device=device, log=log)
                out["models"][label] = scores(y, res.pred, te)
                out["trained"][label] = {**res.settings(), "learned": list(res.learned),
                                         "hidden": list(hidden), "l2": l2,
                                         "epochs": int(res.epochs), "passes": int(res.passes),
                                         "valid_rmse": float(res.valid_rmse),
                                         "seconds": float(res.seconds),
                                         "kernel_seconds": float(res.kernel_seconds)}
                log(f"    {name} seed {seed}: {label}: test R2 "
                    f"{out['models'][label]['r2']:+.3f}, MAE {out['models'][label]['mae']:.4f}, "
                    f"{res.epochs} epochs, {res.passes} passes, {res.seconds:.0f} s")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(out, fh)


# -------------------------------------------------------------------------------- the co-fields
COFIELD_FILE = os.path.join(RESULTS, "cofield.json")
SPACING = 1.0                           # grid spacing of the co-field comparison, angstrom


def build_table(name):
    """The atom table of a set and its field specifications, as ``pack_stage`` builds them."""
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    if name == "compas-1D":
        import run_compas as RC
        ds, meta = RC.load_prepared("1D")
        fs = RC.make_fieldset(COMPAS_GROUPS, meta)
        return RC.build_table(ds, fs, ring_cache={}, carbon_only=meta.get("carbon_only", True))
    ds = load_dataset(os.path.join(ROOT, "examples", "sutherland", f"{name}.sdf"))
    fs = FieldSet.create(physchem=list(FIELDS), alpha=0.5, charge_model="file")
    return AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)])


def stored_model(name, seed, label, packed):
    """A stored linear model as the object ``cmfqsar.learnable.cofield_gaussians`` reads."""
    z = np.load(models_file(name, seed), allow_pickle=False)
    return L.Trained(names=list(packed["names"]), kinds=list(packed["kinds"]),
                     alpha=z[f"{label}|alpha"], h=z[f"{label}|h"], lam=0.0, gamma=None,
                     med=1.0, normalize="", layer=None, landmarks=np.asarray(z["landmarks"], int),
                     coef=z[f"{label}|coef"], bias=float(z[f"{label}|bias"]),
                     pred=np.zeros(0), learned=(), norms=z[f"{label}|norms"])


def cofield_stage(sets=SETS, seeds=None, cube=("compas-1D", 0)):
    """The co-field of the trained layer as a 3D map, compared with that of the grid model.

    For every split whose linear models are stored, the co-field of every field is evaluated
    on a grid over the landmark molecules for the grid model and for the two trained models,
    and the correlation of the trained map with the grid map is recorded: how far the reading
    of the model moves when the settings are trained.  For one split (`cube`) the maps of the
    trained layer are also written as Gaussian cube files under the cache folder.
    """
    from cmfqsar.cofields import evaluate_gaussians, export_cube, make_grid
    out = {}
    if os.path.exists(COFIELD_FILE):
        with open(COFIELD_FILE, encoding="utf-8") as fh:
            out = json.load(fh)
    for name in sets:
        table = packed = None
        for seed in (seeds if seeds is not None else SEEDS[name]):
            key = f"{name}_{seed}"
            if key in out or not os.path.exists(models_file(name, seed)):
                continue
            if table is None:
                table = build_table(name)
                packed = L.load_packed(atoms_file(name))[0]
            models = {m: stored_model(name, seed, m, packed) for m in LINEAR_KEPT}
            Z = table.subset(models["grid"].landmarks)
            grid = make_grid(np.asarray(Z.coords, float), margin=2.0, spacing=SPACING)
            points = grid.points()
            rec = {"points": int(len(points)), "fields": {}}
            for f, field in enumerate(packed["names"]):
                maps = {m: evaluate_gaussians(points, *L.cofield_gaussians(mod, Z, f))
                        for m, mod in models.items()}
                row = {}
                for m in LINEAR_KEPT[1:]:
                    a, b = maps["grid"], maps[m]
                    ok = a.std() > 0 and b.std() > 0
                    row[m] = float(np.corrcoef(a, b)[0, 1]) if ok else None
                rec["fields"][field] = row
                if (name, seed) == tuple(cube):
                    path = os.path.join(CACHE, f"cofield_{name}_{seed}_{f}.cube")
                    export_cube(path, grid, maps["widths + weights"].reshape(grid.shape))
                    rec.setdefault("cubes", []).append(os.path.basename(path))
            out[key] = rec
            shown = {k: v["widths + weights"] for k, v in rec["fields"].items()}
            log(f"  {key}: " + ", ".join(f"{k} {'-' if v is None else round(v, 3)}"
                                         for k, v in shown.items()))
            with open(COFIELD_FILE, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)


def cofield_reference(sets=SETS, seeds=None):
    """The reference of the co-field comparison: the grid models of two splits.

    For every split, the co-field of its grid model is compared with the co-field of the grid
    model of the next split of the same set, on the points of the first.  The two models differ
    in their training molecules (and in what the grid search chose for them), and in nothing
    that was trained.  The correlation says how far the map moves when the training molecules
    are drawn again, which is what the change made by the training must be read against.
    """
    from cmfqsar.cofields import evaluate_gaussians, make_grid
    with open(COFIELD_FILE, encoding="utf-8") as fh:
        out = json.load(fh)
    for name in sets:
        table = packed = None
        all_seeds = [k for k in SEEDS[name] if f"{name}_{k}" in out]
        for seed in (seeds if seeds is not None else all_seeds):
            key = f"{name}_{seed}"
            if key not in out or "reference" in out[key] or len(all_seeds) < 2:
                continue
            other = all_seeds[(all_seeds.index(seed) + 1) % len(all_seeds)]
            if table is None:
                table = build_table(name)
                packed = L.load_packed(atoms_file(name))[0]
            pair = [stored_model(name, k, "grid", packed) for k in (seed, other)]
            tables = [table.subset(m.landmarks) for m in pair]
            points = make_grid(np.asarray(tables[0].coords, float), margin=2.0,
                               spacing=SPACING).points()
            row = {}
            for f, field in enumerate(packed["names"]):
                a, b = (evaluate_gaussians(points, *L.cofield_gaussians(m, Z, f))
                        for m, Z in zip(pair, tables))
                ok = a.std() > 0 and b.std() > 0
                row[field] = float(np.corrcoef(a, b)[0, 1]) if ok else None
            out[key]["reference"] = row
            out[key]["reference_seed"] = int(other)
            log(f"  {key} against the grid model of seed {other}: "
                + ", ".join(f"{k} {'-' if v is None else round(v, 3)}" for k, v in row.items()))
            with open(COFIELD_FILE, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)


# ---------------------------------------------------------------------------------- the report
def collect(name):
    """One record per seed with the rows of both stages and the device row of ND1."""
    runs = []
    for seed in SEEDS[name]:
        if not os.path.exists(ridge_file(name, seed)):
            continue
        with open(ridge_file(name, seed), encoding="utf-8") as fh:
            r = json.load(fh)
        if os.path.exists(network_file(name, seed)):
            with open(network_file(name, seed), encoding="utf-8") as fh:
                net = json.load(fh)
            r["models"].update(net["models"])
            r["trained"].update(net["trained"])
        nd1 = nd1_result(name, seed)
        if nd1["test"] != r["test"]:
            raise SystemExit(f"{name} seed {seed}: the test rows differ from those of ND1")
        for key in ("CMF kernel", "CMF vector | ridge", "CMF vector | network", DEVICE_MP,
                    "CMF kernel, Gaussian layer",
                    "CMF vector + collectors, message passing | network"):
            if key in nd1["models"]:
                r["models"].setdefault(f"ND1: {key}" if key != DEVICE_MP else key,
                                       nd1["models"][key])
        runs.append(r)
    return runs


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
    sign = "+" if what == "r2" else ""
    return f"{v.mean():{sign}.{nd}f} ± {v.std():.{nd}f}"


def _pair(p, nd=3):
    return (f"{p['mean']:+.{nd}f} [{p['low']:+.{nd}f}, {p['high']:+.{nd}f}]"
            + ("" if p["excludes_zero"] else " (includes zero)"))


ND1_ROWS = (("ND1: CMF kernel", "the kernel model of ND1 on every training molecule"),
            ("ND1: CMF vector | ridge", "ND1: ridge regression on the CMF vector"),
            ("ND1: CMF kernel, Gaussian layer", "ND1: the kernel model with the Gaussian layer"),
            ("ND1: CMF vector | network", "ND1: multilayer network on the CMF vector"),
            (DEVICE_MP, "ND1: the device with message passing"),
            ("ND1: CMF vector + collectors, message passing | network",
             "ND1: multilayer network on the CMF vector joined with the collectors"))
TRAINED_ROWS = ("ridge", "weights", "widths", "widths + weights", TO_MINIMUM,
                "ridge, layer", "widths + weights, layer")


def _geo(values):
    return float(np.exp(np.mean(np.log(np.asarray(values, float)))))


def write_table():
    """The table of plan item ND2, from the result files; returns the path."""
    lines = ["# A learnable CMF layer (plan item ND2)", "",
             "The attenuation of every field (the width of the Gaussian on every atom), the "
             "weight of every field in the joint kernel and the ridge are trained by gradient "
             "(`cmfqsar/learnable.py`). The coefficients of the model stay the closed-form "
             "ridge solution, so the model stays the kernel model and keeps its co-field. The "
             "criterion is the squared error of five-fold out-of-fold predictions inside the "
             "training part. The sets, the splits (random, 70 % training, 15 % validation, "
             "15 % test) and the start of every training (the settings of the grid search) are "
             "those of plan item ND1. Every score is on the test rows.", "",
             "What each row trained: **grid** nothing; **ridge** the ridge only; **weights** "
             "the field weights and the ridge; **widths** the attenuations and the ridge; "
             "**widths + weights** all three. These rows keep the evaluation with the lowest "
             "error on the validation rows (early stopping). **To the minimum** keeps the "
             "evaluation with the lowest criterion of the same training. **Layer** rows have "
             "the Gaussian kernel layer on top and train its width with the ridge.", ""]
    compas = collect("compas-1D")
    if compas:
        k = len(compas)
        lines += [f"## COMPAS-1D ({compas[0]['n']} molecules, {k} seeds, "
                  f"{compas[0]['landmarks']} landmark molecules)", "",
                  "The gap between the highest occupied and the lowest unoccupied orbital, in "
                  "eV. Mean ± standard deviation over the seeds.", "",
                  "| Model | Test R² | Mean absolute error, eV |", "|---|---:|---:|"]
        have = compas[0]["models"]
        for key in list(RIDGE_MODELS)[:5] + [TO_MINIMUM] + list(LAYER_MODELS) \
                + list(NETWORK_MODELS):
            if all(key in r["models"] for r in compas):
                lines.append(f"| {key} | {_pm(compas, key)} | {_pm(compas, key, 'mae')} |")
        for key, label in ND1_ROWS:
            if key in have:
                lines.append(f"| {label} | {_pm(compas, key)} | {_pm(compas, key, 'mae')} |")
        lines += ["", "The rows of ND1 are copied from its result files for the same test "
                  "rows. The first is the kernel model on all 6,074 training molecules; every "
                  "row of this item is a model over the 1,000 landmarks.", ""]
        lines += ["### Paired differences on COMPAS-1D", "",
                  "Difference of the absolute error on the same test molecules, in eV, with a "
                  "95 % bootstrap interval. A molecule in the test part of several seeds "
                  "counts once. A negative value means that the first model has the smaller "
                  "error.", "", "| Comparison | Difference, eV |", "|---|---:|"]
        for a, b in PAIRS:
            if all(a in r["models"] and b in r["models"] for r in compas):
                lines.append(f"| {a} against {b} | {_pair(paired(compas, a, b), 4)} |")
        lines += ["", "### What the training changed on COMPAS-1D", "",
                  "The criterion is the out-of-fold squared error inside the training part, "
                  "divided by the variance of the activity. The validation error is that of "
                  "the model the row keeps.", "",
                  "| Model | Criterion at the start | Lowest criterion | Validation RMSE at "
                  "the start, eV | Validation RMSE of the kept model, eV | Evaluations | "
                  "Minutes |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key in TRAINED_ROWS:
            if not all(key in r["trained"] for r in compas):
                continue
            tr = [r["trained"][key] for r in compas]
            lines.append(
                f"| {key} | {np.mean([x['cv_start'] for x in tr]):.4f} | "
                f"{np.mean([x['cv_lowest'] for x in tr]):.4f} | "
                f"{np.mean([x['valid_rmse_start'] for x in tr]):.4f} | "
                f"{np.mean([x['valid_rmse'] for x in tr]):.4f} | "
                f"{np.mean([x['evaluations'] for x in tr]):.0f} | "
                f"{np.mean([x['seconds'] for x in tr]) / 60:.0f} |")
        fields, groups = compas[0]["fields"], compas[0]["groups"]
        lines += ["", "### The attenuations and the weights on COMPAS-1D", "",
                  "The attenuation of the grid search is one of 0.1, 0.2, 0.5, 1, 2 and 5 "
                  "for a group of fields. The trained values are geometric means over the "
                  "seeds, with the smallest and the largest. The share is the weight of the "
                  "field divided by the sum of the weights; equal weights give "
                  f"{1 / len(fields):.3f}.", "",
                  "| Field (group) | Grid attenuation of the five seeds | Widths + weights: "
                  "attenuation | Widths + weights: share of the weight | To the minimum: "
                  "attenuation | To the minimum: share of the weight |",
                  "|---|---|---|---:|---|---:|"]
        for f, (name, g) in enumerate(zip(fields, groups)):
            row = [f"{name} ({g})",
                   ", ".join(f"{r['start']['alpha'][f]:g}" for r in compas)]
            for key in ("widths + weights", TO_MINIMUM):
                if not all(key in r["trained"] for r in compas):
                    row += ["-", "-"]
                    continue
                al = [r["trained"][key]["alpha"][name] for r in compas]
                hs = [r["trained"][key]["h"][name] / sum(r["trained"][key]["h"].values())
                      for r in compas]
                row += [f"{_geo(al):.3g} ({min(al):.3g} to {max(al):.3g})",
                        f"{np.mean(hs):.3f}"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    series = {s: collect(s) for s in SUTHERLAND}
    series = {s: r for s, r in series.items() if r}
    if series:
        keys = [k for k in list(RIDGE_MODELS)[:5] + [TO_MINIMUM]
                if all(k in r["models"] for runs in series.values() for r in runs)]
        lines += [f"## The Sutherland series ({len(series)} series, "
                  f"{len(next(iter(series.values())))} seeds each)", "",
                  "Every training molecule is a landmark, so every row is the exact kernel "
                  "model. Test R², mean over the seeds. The column before the last is the "
                  "mean over the series. The last column is the median over all splits of "
                  "all series, which one failed split does not move.", "",
                  "| Model | " + " | ".join(series) + " | Mean over the series | Median over "
                  "all splits |",
                  "|---|" + "---:|" * (len(series) + 2)]
        for key in keys:
            vals = [np.mean([r["models"][key]["r2"] for r in runs]) for runs in series.values()]
            every = [r["models"][key]["r2"] for runs in series.values() for r in runs]
            lines.append(f"| {key} | " + " | ".join(f"{v:+.3f}" for v in vals)
                         + f" | {np.mean(vals):+.3f} | {np.median(every):+.3f} |")
        lines += ["", "### Paired differences on the Sutherland series", "",
                  "For every comparison: the number of series in which the first model has "
                  "the smaller absolute error with an interval that excludes zero, the number "
                  "in which it has the larger one, and the mean difference of the test R² over "
                  "the series.", "",
                  "| Comparison | Series better | Series worse | Mean difference of R² |",
                  "|---|---:|---:|---:|"]
        for a, b in PAIRS:
            if a not in keys or b not in keys:
                continue
            ps = [paired(runs, a, b) for runs in series.values()]
            better = sum(1 for q in ps if q["excludes_zero"] and q["mean"] < 0)
            worse = sum(1 for q in ps if q["excludes_zero"] and q["mean"] > 0)
            d = np.mean([np.mean([r["models"][a]["r2"] - r["models"][b]["r2"] for r in runs])
                         for runs in series.values()])
            lines.append(f"| {a} against {b} | {better} | {worse} | {d:+.3f} |")
        lines += ["", "### The criterion and the validation error on the Sutherland series", "",
                  "Means over all series and seeds. The criterion is the out-of-fold squared "
                  "error inside the training part divided by the variance of the activity.", "",
                  "| Model | Criterion at the start | Lowest criterion | Share of the trainings "
                  "in which early stopping kept the start |", "|---|---:|---:|---:|"]
        for key in ("ridge", "weights", "widths", "widths + weights"):
            tr = [r["trained"][key] for runs in series.values() for r in runs]
            kept = np.mean([abs(x["valid_rmse"] - x["valid_rmse_start"]) < 1e-12 for x in tr])
            lines.append(f"| {key} | {np.mean([x['cv_start'] for x in tr]):.3f} | "
                         f"{np.mean([x['cv_lowest'] for x in tr]):.3f} | {100 * kept:.0f} % |")
        lines.append("")
    both = {"COMPAS-1D": compas, "Sutherland series": [r for runs in series.values()
                                                       for r in runs]}
    if any(both.values()):
        lo, hi = L.ALPHA_BOUNDS
        lines += ["## Where the training of the attenuations ended", "",
                  f"The attenuation of a field may move between {lo:g} and {hi:g}. A trained "
                  "attenuation is counted as at a limit when it is within 1 % of one of the two "
                  "values. The share is over the fields and the splits. The last three columns "
                  "are shares of the trainings, by the reason the optimiser (L-BFGS-B) gave "
                  "for its end: the criterion or its gradient no longer changed; all the "
                  "allowed steps were used; the search along the last direction found no "
                  "lower value of the criterion.", "",
                  "| Set | Model | Attenuations at the lower limit | Attenuations at the upper "
                  "limit | Trainings ended by the criterion | Trainings ended by the number "
                  "of steps | Trainings ended by a search that found no lower value |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for label, runs in both.items():
            for key in ("widths", "widths + weights", TO_MINIMUM, "widths + weights, layer"):
                tr = [r["trained"][key] for r in runs if key in r["trained"]]
                if not tr:
                    continue
                al = np.array([a for x in tr for a in x["alpha"].values()], float)
                stop = [x.get("stop", "") for x in tr]
                lines.append(
                    f"| {label} | {key} | {100 * np.mean(al <= lo * 1.01):.0f} % | "
                    f"{100 * np.mean(al >= hi / 1.01):.0f} % | "
                    f"{100 * np.mean([m.startswith('CONVERGENCE') for m in stop]):.0f} % | "
                    f"{100 * np.mean([m.startswith('STOP') for m in stop]):.0f} % | "
                    f"{100 * np.mean([m.startswith('ABNORMAL') for m in stop]):.0f} % |")
        lines.append("")
    if os.path.exists(COFIELD_FILE):
        with open(COFIELD_FILE, encoding="utf-8") as fh:
            cof = json.load(fh)
        lines += ["## The co-field of the trained layer against that of the grid model", "",
                  "The co-field of every field was evaluated on a grid of points "
                  f"{SPACING:g} Å apart over the landmark molecules, for the grid model and "
                  "for the trained models of the same split. The number is the correlation "
                  "coefficient of the two maps. Where early stopping kept the start, the "
                  "trained model is the grid model and the two maps are equal; these splits "
                  "are counted and left out of the mean. The mean is over the fields and the "
                  "remaining splits, with the smallest value of any field and split. The "
                  "last two columns are the reference: the same correlation between the grid "
                  "models of two splits of the set, which differ in their training molecules "
                  "only.", "",
                  "| Set | Splits | Widths + weights: splits with a changed layer | Widths + "
                  "weights: mean correlation | Widths + weights: smallest correlation | To the "
                  "minimum: splits with a changed layer | To the minimum: mean correlation | "
                  "To the minimum: smallest correlation | Grid models of two splits: mean "
                  "correlation | Grid models of two splits: smallest correlation |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name in SETS:
            keys = [k for k in cof if k.rsplit("_", 1)[0] == name]
            if not keys:
                continue
            row = [name, str(len(keys))]
            for key in ("widths + weights", TO_MINIMUM):
                v, changed = [], 0
                for k in keys:
                    with open(os.path.join(RESULTS, k + ".json"), encoding="utf-8") as fh:
                        tr = json.load(fh)["trained"]
                    a, g = tr[key], tr["grid"]
                    same = (all(np.isclose(a[q][f], g[q][f], rtol=1e-9)
                                for q in ("alpha", "h") for f in g[q])
                            and np.isclose(a["lam"], g["lam"], rtol=1e-9))
                    if same:
                        continue
                    changed += 1
                    v += [x[key] for x in cof[k]["fields"].values() if x[key] is not None]
                row += [str(changed)] + ([f"{np.mean(v):.3f}", f"{np.min(v):.3f}"] if v
                                         else ["-", "-"])
            ref = [x for k in keys for x in cof[k].get("reference", {}).values()
                   if x is not None]
            row += [f"{np.mean(ref):.3f}", f"{np.min(ref):.3f}"] if ref else ["-", "-"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", action="append", default=[],
                    choices=["pack", "ridge", "network", "cofield"])
    ap.add_argument("--sets", default=None, help="comma-separated; default: all")
    ap.add_argument("--seeds", default=None, help="comma-separated; default: all of a set")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if there is one)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--models", default=None, help="network stage: labels separated by ;")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print every evaluation")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    sets = tuple(args.sets.split(",")) if args.sets else SETS
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else None
    for stage in args.stage:
        if stage == "pack":
            pack_stage(sets)
        elif stage == "ridge":
            ridge_stage(sets, seeds, args.device, args.force, args.steps, args.verbose)
        elif stage == "network":
            network_stage(tuple(s for s in sets if s == "compas-1D"), seeds, args.device,
                          args.force, args.epochs,
                          args.models.split(";") if args.models else None)
        elif stage == "cofield":
            cofield_stage(sets, seeds)
            cofield_reference(sets, seeds)
    if args.report:
        print(write_table())


if __name__ == "__main__":
    main()
