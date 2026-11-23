#!/usr/bin/env python
"""
Atom amplitudes of the fields from a network on the atoms: plan item ND14a.

A field of CMF puts a Gaussian on every atom, and the amplitude of the Gaussian is a fixed
number: a partial charge, a well depth, a hydrogen-bond strength, a one in an indicator field.
Item ND14a lets a small network produce the amplitudes (``cmfqsar/amplitudes.py``).  The
network is the first level of the neural device of Baskin, Palyulin and Zefirov (J. Chem. Inf.
Comput. Sci. 37 (1997) 715): one network with shared weights that reads every atom of the
molecular graph.  Its output is a correction to the fixed amplitude, it starts at zero, and a
penalty holds it near zero.  The model stays the kernel model: the coefficients are the
closed-form ridge solution for the current amplitudes, so the co-field is kept.  The question
of the item is the question of ND4a: do learned atom values improve on the fixed ones inside
the same kernel?

The sets, the splits and the start of every training are those of plan items ND1 and ND2:
COMPAS-1D (8,678 polybenzenoid hydrocarbons, the HOMO-LUMO gap, nine fields, 1,000 landmark
molecules, five seeds) and the eight Sutherland series (five fields, every training molecule a
landmark, ten seeds each); random 70 % training, 15 % validation, 15 % test, read from the
device files of ND1.  The attenuations, the field weights and the start of the ridge are the
settings that the grid search of ND1 chose for the split.  They are not trained here.

On a Sutherland series the amplitudes of all five fields are corrected.  On COMPAS-1D the
amplitudes of the four fields on the atoms are corrected (steric, the xTB charges, the two
carbon classes); the five ring fields sit on ring centres, which are not atoms of the graph,
and stay fixed.

The rows
--------
    fixed amplitudes             nothing but the ridge is trained: the reference
    learned, the atom alone      the network reads one atom (no neighbours)
    learned, with neighbours     every atom also reads its bonded neighbours, two rounds
    learned, to the minimum      the same training without early stopping

and, on COMPAS-1D, ``fixed amplitudes, layer`` and ``learned, with neighbours, layer``: the
same two models with the Gaussian kernel layer on top (a model with the layer has no
co-field).  The criterion is the squared error of five-fold out-of-fold predictions inside the
training part, plus the penalty.  With early stopping, the evaluation with the lowest error on
the validation rows is kept.  The test rows are used for the scores only.

The sizes were fixed before any run: 16 hidden units, two rounds of reading the neighbours,
30 iterations of the quasi-Newton method.  The penalty (0.1) and the length of the first step
(0.3) were chosen on the validation rows of the first split of DHFR, among the penalties 0.1,
1 and 10 and the first steps 0.1 and 0.3.  The choice was not repeated on COMPAS-1D, where one
training takes 40 minutes.

The comparisons, fixed before any run
-------------------------------------
Paired differences of the absolute error on the same test molecules, 95 % bootstrap interval:

    1  learned, with neighbours against fixed amplitudes      the gate of the item
    2  learned, the atom alone against fixed amplitudes
    3  learned, with neighbours against the atom alone        what the neighbours add
    4  learned, to the minimum against with neighbours        what early stopping is worth
    5  learned, with neighbours, layer against fixed amplitudes, layer      (COMPAS-1D)

The gate is comparison 1 on COMPAS-1D and on one Sutherland series.  The series is DHFR, named
before any run because it is the largest (397 molecules).  The other seven are reported too.

Usage
-----
    python benchmarks/run_amplitudes.py --stage pack       # the graph of every set, once
    python benchmarks/run_amplitudes.py --stage train      # needs PyTorch; uses a GPU if any
    python benchmarks/run_amplitudes.py --stage reading    # what the training changed
    python benchmarks/run_amplitudes.py --report

The stage ``train`` needs the result and device files of ND1 under
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

from cmfqsar import amplitudes as A                                         # noqa: E402
from cmfqsar import learnable as L                                          # noqa: E402

import run_learnable as RL                                                  # noqa: E402

RESULTS = os.path.join(HERE, "results", "amplitudes")
CACHE = os.path.join(HERE, "cache", "amplitudes")
SUTHERLAND = RL.SUTHERLAND
SETS = RL.SETS
SEEDS = dict(RL.SEEDS)
GATE_SERIES = "DHFR"
FOLDS = RL.FOLDS
#: the sizes, fixed before any run; the penalty and the first step were chosen on the
#: validation rows of the first split of DHFR
NETWORK = {"hidden": 16, "steps": 30, "penalty": 0.1, "first_step": 0.3}
FIXED = "fixed amplitudes"
ALONE = "learned, the atom alone"
NEIGHBOURS = "learned, with neighbours"
TO_MINIMUM = "learned, to the minimum"
FIXED_LAYER = "fixed amplitudes, layer"
NEIGHBOURS_LAYER = "learned, with neighbours, layer"
#: label -> (the network is trained, rounds of reading the neighbours, layer)
MODELS = {
    FIXED: (False, 0, None),
    ALONE: (True, 0, None),
    NEIGHBOURS: (True, 2, None),
    FIXED_LAYER: (False, 0, "gauss"),
    NEIGHBOURS_LAYER: (True, 2, "gauss"),
}
LAYER_MODELS = (FIXED_LAYER, NEIGHBOURS_LAYER)
PAIRS = (
    (1, NEIGHBOURS, FIXED, "the gate of the item"),
    (2, ALONE, FIXED, "what a network on the atom alone gains"),
    (3, NEIGHBOURS, ALONE, "what the neighbours add"),
    (4, TO_MINIMUM, NEIGHBOURS, "what early stopping is worth"),
    (5, NEIGHBOURS_LAYER, FIXED_LAYER, "the same as 1 under the Gaussian kernel layer"),
)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------------- the graphs
def graph_file(name):
    return os.path.join(CACHE, f"{name}_graph.npz")


def table_of(name):
    """The atom table of a set and the specifications of its fields, as ND2 builds them."""
    table = RL.build_table(name)
    if name == "compas-1D":
        import run_compas as RC
        _ds, meta = RC.load_prepared("1D")
        fs = RC.make_fieldset(RL.COMPAS_GROUPS, meta)
    else:
        from cmfqsar.fields import FieldSet
        fs = FieldSet.create(physchem=list(RL.FIELDS), alpha=0.5, charge_model="file")
    return table, [s for s in fs.specs if s.enabled]


def pack_stage(sets=SETS):
    """The graph of every set: the packed atoms of ND2 and the input of the network."""
    os.makedirs(CACHE, exist_ok=True)
    for name in sets:
        table, specs = table_of(name)
        graph = A.pack_graph(table, specs)
        packed, y, molecules, _groups = RL.load_atoms(name)
        for f in range(len(specs)):
            if not np.array_equal(graph["w"][f], packed["w"][f]):
                raise SystemExit(f"{name}: the atoms of field {specs[f].name} differ from "
                                 "the packed atoms of ND2")
        A.save_graph(graph_file(name), graph, y=y, molecules=np.array(molecules))
        log(f"  {name}: {graph['n']} molecules, {graph['x'].shape[0]} atoms, "
            f"{graph['x'].shape[1]} input columns ({len(graph['types'])} atom types), "
            f"{graph['edges'].shape[1] // 2} bonds")


def load_graph(name):
    graph, extra = A.load_graph(graph_file(name))
    return graph, np.asarray(extra["y"], float), [str(v) for v in extra["molecules"]]


def learned_fields(name, graph):
    """The fields whose amplitudes are corrected: those on atoms of the molecular graph."""
    return [f for f in graph["names"] if not f.startswith("ring:")]


# ---------------------------------------------------------------------------------- the training
def result_file(name, seed):
    return os.path.join(RESULTS, f"{name}_{seed}.json")


def models_file(name, seed):
    """The trained models of one split (network, coefficients, amplitudes of the landmarks)."""
    return os.path.join(CACHE, f"{name}_{seed}_models.npz")


def models_of(name):
    return [k for k in MODELS if name == "compas-1D" or k not in LAYER_MODELS]


def precision(name):
    """The precision of the kernels: double on a Sutherland series, where it costs little;
    on COMPAS-1D the default of the device (single precision on a GPU, as in ND2)."""
    import torch
    return None if name == "compas-1D" else torch.float64


def record(res, how):
    cvs = [h["cv"] for h in res.history]
    at = int(np.argmin([h["valid_rmse"] if how == "valid" else h["objective"]
                        for h in res.history]))
    return {**res.settings(), "learned": list(res.learned), "select": how,
            "cv_start": float(res.start["cv"]), "cv_lowest": float(min(cvs)),
            "cv_kept": float(cvs[at]), "kept_evaluation": at,
            "valid_rmse_start": float(res.start["valid_rmse"]),
            "valid_rmse": float(res.history[at]["valid_rmse"]),
            "evaluations": int(res.evaluations), "weights": int(res.start["weights"]),
            "stop": str(res.start.get("stop", "")), "seconds": float(res.seconds),
            "pass_seconds": float(res.pass_seconds)}


def keep(kept, key, res):
    """Store what the co-field stage and a later reading need of one linear model."""
    kept[f"{key}|coef"] = np.asarray(res.coef, float)
    kept[f"{key}|bias"] = np.array(res.bias)
    kept[f"{key}|norms"] = np.asarray(res.norms, float)
    kept[f"{key}|lam"] = np.array(res.lam)
    for f, a in enumerate(res.land["a"]):
        kept[f"{key}|a_{f}"] = np.asarray(a, np.float32)
    for k, v in res.state.items():
        kept[f"{key}|net|{k}"] = np.asarray(v, float)


def train_stage(sets=SETS, seeds=None, device=None, force=False, models=None, verbose=False,
                network=None):
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    net = dict(NETWORK, **(network or {}))
    for name in sets:
        todo = [k for k in (seeds if seeds is not None else SEEDS[name])]
        graph = None
        for seed in todo:
            path = result_file(name, seed)
            out = None
            if os.path.exists(path) and not force:
                with open(path, encoding="utf-8") as fh:
                    out = json.load(fh)
            wanted = [m for m in (models or models_of(name))
                      if out is None or m not in out["models"]]
            if not wanted:
                continue
            if graph is None:
                graph, y, molecules = load_graph(name)
                _packed, _y, _molecules, groups = RL.load_atoms(name)
                fields = learned_fields(name, graph)
            split = RL.nd1_split(name, seed, molecules)["split"]
            tr, va, te = split["train"], split["valid"], split["test"]
            land = RL.landmarks_of(name, seed, tr)
            start = RL.start_of(name, seed, groups, len(tr))
            if out is None:
                out = {"set": name, "seed": seed, "n": len(y), "n_train": len(tr),
                       "n_valid": len(va), "n_test": len(te), "test": te.tolist(),
                       "fields": graph["names"], "learned_fields": fields,
                       "landmarks": int(len(land)), "network": net,
                       "start": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in start.items()}, "models": {}, "trained": {}}
            kept = {}
            if os.path.exists(models_file(name, seed)):
                with np.load(models_file(name, seed), allow_pickle=False) as z:
                    kept = {k: z[k] for k in z.files}
            kept["landmarks"] = land
            for label in wanted:
                trained, rounds, layer = MODELS[label]
                both = label == NEIGHBOURS
                res = A.train(graph, y, tr, land, valid=va, fields=fields if trained else [],
                              alpha=start["alpha"], h=start["h"], lam=start["lam"],
                              gamma=start["gamma"] if layer else None, hidden=net["hidden"],
                              rounds=rounds, penalty=net["penalty"], normalize=start["normalize"],
                              layer=layer, steps=net["steps"], folds=FOLDS, seed=seed,
                              predict=te, keep_criterion=both, first_step=net["first_step"],
                              device=device, dtype=precision(name),
                              log=log if verbose else None)
                for key, r, how in ((label, res, "valid"),) + (
                        ((TO_MINIMUM, res.by_criterion, "cv"),) if both else ()):
                    out["models"][key] = RL.scores(y, r.pred, te)
                    out["trained"][key] = record(r, how)
                    if layer is None:
                        keep(kept, key, r)
                    log(f"    {name} seed {seed}: {key}: test R2 "
                        f"{out['models'][key]['r2']:+.3f}, MAE {out['models'][key]['mae']:.4f}, "
                        f"{r.evaluations} evaluations, {r.seconds:.0f} s, correction "
                        + ", ".join(f"{k} {v:.2f}" for k, v in r.correction.items()))
                np.savez_compressed(models_file(name, seed), **kept)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(out, fh)
            log(f"  {name} seed {seed} done")


# ---------------------------------------------------------------------------------- the reading
READING_FILE = os.path.join(RESULTS, "reading.json")
SPACING = RL.SPACING                    # grid spacing of the co-field comparison, angstrom
READ = (ALONE, NEIGHBOURS, TO_MINIMUM)


def _gaussians(kept, label, graph, land, start, f, amplitude):
    """The co-field of field `f` of a stored linear model as atom-centred Gaussians."""
    a = np.asarray(amplitude, float)
    use = a != 0.0
    per_landmark = (np.asarray(kept[f"{label}|coef"], float) * float(start["h"][f])
                    * np.asarray(kept[f"{label}|norms"], float)[f])
    return (np.asarray(graph["xyz"][f], float)[land][use], (per_landmark[:, None] * a)[use],
            float(start["alpha"][f]) * np.asarray(graph["c"][f], float)[land][use])


def reading_stage(sets=SETS, seeds=None):
    """What the training changed: the amplitudes and the co-field of every stored model.

    For every split and every corrected field: the correlation of the trained amplitudes with
    the fixed ones over the atoms of the landmark molecules, the mean and the root mean square
    of the correction for every atom type (in units of the field scale), and the correlation
    of the co-field of the trained model with the co-field of the model with the fixed
    amplitudes, on a grid of points over the landmark molecules.
    """
    from cmfqsar.cofields import evaluate_gaussians, make_grid
    out = {}
    if os.path.exists(READING_FILE):
        with open(READING_FILE, encoding="utf-8") as fh:
            out = json.load(fh)
    for name in sets:
        graph = None
        for seed in (seeds if seeds is not None else SEEDS[name]):
            key = f"{name}_{seed}"
            if key in out or not os.path.exists(models_file(name, seed)):
                continue
            with np.load(models_file(name, seed), allow_pickle=False) as z:
                kept = {k: z[k] for k in z.files}
            if not all(f"{m}|coef" in kept for m in (FIXED,) + READ):
                continue
            if graph is None:
                graph, _y, _molecules = load_graph(name)
                _packed, _y2, _m2, groups = RL.load_atoms(name)
                n_types = len(graph["types"])
            land = np.asarray(kept["landmarks"], int)
            start = RL.start_of(name, seed, groups, 0)
            centres = np.concatenate([np.asarray(graph["xyz"][f], float)[land]
                                      [np.asarray(graph["w"][f])[land] != 0.0]
                                      for f in range(len(graph["names"]))])
            points = make_grid(centres, margin=2.0, spacing=SPACING).points()
            rec = {"points": int(len(points)), "fields": {}}
            for f, field in enumerate(graph["names"]):
                if field not in learned_fields(name, graph):
                    continue
                w = np.asarray(graph["w"][f], float)[land]
                use = w != 0.0
                atom = np.asarray(graph["atom"][f])[land][use]
                kind = np.argmax(graph["x"][atom, :n_types], axis=1)
                fixed_map = evaluate_gaussians(points, *_gaussians(kept, FIXED, graph, land,
                                                                   start, f, w))
                row = {}
                for label in READ:
                    a = np.asarray(kept[f"{label}|a_{f}"], float)
                    d = (a[use] - w[use]) / float(graph["scale"][f])
                    trained_map = evaluate_gaussians(points, *_gaussians(kept, label, graph,
                                                                         land, start, f, a))
                    ok = fixed_map.std() > 0 and trained_map.std() > 0
                    spread = w[use].std() > 0 and a[use].std() > 0
                    row[label] = {
                        "amplitude_correlation": float(np.corrcoef(a[use], w[use])[0, 1])
                        if spread else None,
                        "correction_rms": float(np.sqrt(np.mean(d ** 2))),
                        "cofield_correlation": float(np.corrcoef(fixed_map, trained_map)[0, 1])
                        if ok else None,
                        "by_type": {graph["types"][t]: [int(np.sum(kind == t)),
                                                        float(d[kind == t].mean()),
                                                        float(np.sqrt(np.mean(d[kind == t] ** 2)))]
                                    for t in np.unique(kind)}}
                rec["fields"][field] = row
            out[key] = rec
            log(f"  {key}: " + ", ".join(
                f"{field} {row[NEIGHBOURS]['cofield_correlation']:.3f}"
                for field, row in rec["fields"].items()
                if row[NEIGHBOURS]["cofield_correlation"] is not None))
            with open(READING_FILE, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)


# ---------------------------------------------------------------------------------- the report
def collect(name):
    """One record per seed, with the rows of ND2 for the same test rows."""
    runs = []
    for seed in SEEDS[name]:
        if not os.path.exists(result_file(name, seed)):
            continue
        with open(result_file(name, seed), encoding="utf-8") as fh:
            r = json.load(fh)
        if not all(m in r["models"] for m in models_of(name) + [TO_MINIMUM]):
            continue
        with open(RL.ridge_file(name, seed), encoding="utf-8") as fh:
            nd2 = json.load(fh)
        if nd2["test"] != r["test"]:
            raise SystemExit(f"{name} seed {seed}: the test rows differ from those of ND2")
        for key in ("grid", "ridge", "widths + weights", "grid, layer",
                    "widths + weights, layer"):
            if key in nd2["models"]:
                r["models"][f"ND2: {key}"] = nd2["models"][key]
        runs.append(r)
    return runs


ND2_ROWS = (("ND2: grid", "ND2: nothing trained (the grid search)"),
            ("ND2: ridge", "ND2: the ridge trained"),
            ("ND2: widths + weights", "ND2: attenuations, field weights and ridge trained"),
            ("ND2: grid, layer", "ND2: nothing trained, Gaussian kernel layer"),
            ("ND2: widths + weights, layer", "ND2: everything trained, Gaussian kernel layer"))
ROWS = (FIXED, ALONE, NEIGHBOURS, TO_MINIMUM, FIXED_LAYER, NEIGHBOURS_LAYER)


def _mean_of(runs, key, what):
    return float(np.mean([r["trained"][key][what] for r in runs]))


def _changes(runs, keys):
    lines = ["| Model | Weights of the network | Criterion at the start | Criterion of the "
             "kept model | Lowest criterion | Validation RMSE at the start | Validation RMSE "
             "of the kept model | Share of the trainings in which early stopping kept the "
             "start | Evaluations | Seconds |", "|---|" + "---:|" * 9]
    for key in keys:
        if not all(key in r["trained"] for r in runs):
            continue
        tr = [r["trained"][key] for r in runs]
        kept = np.mean([x["kept_evaluation"] == 0 for x in tr])
        lines.append(
            f"| {key} | {tr[0]['weights']} | {np.mean([x['cv_start'] for x in tr]):.4f} | "
            f"{np.mean([x['cv_kept'] for x in tr]):.4f} | "
            f"{np.mean([x['cv_lowest'] for x in tr]):.4f} | "
            f"{np.mean([x['valid_rmse_start'] for x in tr]):.4f} | "
            f"{np.mean([x['valid_rmse'] for x in tr]):.4f} | {100 * kept:.0f} % | "
            f"{np.mean([x['evaluations'] for x in tr]):.0f} | "
            f"{np.mean([x['seconds'] for x in tr]):.0f} |")
    return lines


def _corrections(runs, keys):
    fields = runs[0]["learned_fields"]
    lines = ["| Model | " + " | ".join(fields) + " |", "|---|" + "---:|" * len(fields)]
    for key in keys:
        if not all(key in r["trained"] for r in runs):
            continue
        lines.append(f"| {key} | " + " | ".join(
            f"{np.mean([r['trained'][key]['correction'].get(f, 0.0) for r in runs]):.2f}"
            for f in fields) + " |")
    return lines


def _reading(name, reading, runs):
    keys = [k for k in reading if k.rsplit("_", 1)[0] == name]
    if not keys:
        return []
    fields = list(reading[keys[0]]["fields"])
    reference = {}
    if os.path.exists(RL.COFIELD_FILE):
        with open(RL.COFIELD_FILE, encoding="utf-8") as fh:
            nd2 = json.load(fh)
        for f in fields:
            v = [nd2[k]["reference"][f] for k in keys
                 if k in nd2 and nd2[k].get("reference", {}).get(f) is not None]
            if v:
                reference[f] = float(np.mean(v))
    lines = ["| Model | Field | Correlation of the trained with the fixed amplitudes | "
             "Correlation of the co-field with that of the fixed amplitudes | Reference: "
             "correlation of the co-fields of two splits, fixed amplitudes (ND2) |",
             "|---|---|---:|---:|---:|"]
    for label in READ:
        for f in fields:
            rows = [reading[k]["fields"][f][label] for k in keys]
            amp = [x["amplitude_correlation"] for x in rows
                   if x["amplitude_correlation"] is not None]
            cof = [x["cofield_correlation"] for x in rows
                   if x["cofield_correlation"] is not None]
            lines.append(f"| {label} | {f} | " + (f"{np.mean(amp):.3f}" if amp else "-")
                         + " | " + (f"{np.mean(cof):.3f}" if cof else "-") + " | "
                         + (f"{reference[f]:.3f}" if f in reference else "-") + " |")
    return lines


def _by_type(name, reading, label=NEIGHBOURS, least=0.01):
    keys = [k for k in reading if k.rsplit("_", 1)[0] == name]
    if not keys:
        return []
    fields = list(reading[keys[0]]["fields"])
    types = sorted({t for k in keys for f in fields
                    for t in reading[k]["fields"][f][label]["by_type"]})
    count = {t: sum(reading[k]["fields"][f][label]["by_type"].get(t, [0])[0]
                    for k in keys for f in fields) for t in types}
    total = sum(count.values())
    types = [t for t in types if count[t] >= least * total]
    lines = ["| Atom type | " + " | ".join(fields) + " |", "|---|" + "---:|" * len(fields)]
    for t in types:
        cells = []
        for f in fields:
            v = [reading[k]["fields"][f][label]["by_type"][t] for k in keys
                 if t in reading[k]["fields"][f][label]["by_type"]]
            n = sum(x[0] for x in v)
            cells.append("-" if not n else
                         f"{sum(x[0] * x[1] for x in v) / n:+.2f} "
                         f"({np.sqrt(sum(x[0] * x[2] ** 2 for x in v) / n):.2f})")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return lines


def write_table():
    """The table of plan item ND14a, from the result files; returns the path."""
    P = {k: (a, b, what) for k, a, b, what in PAIRS}
    reading = {}
    if os.path.exists(READING_FILE):
        with open(READING_FILE, encoding="utf-8") as fh:
            reading = json.load(fh)
    lines = ["# Atom amplitudes of the fields from a network on the atoms (plan item ND14a)", "",
             "A field puts a Gaussian on every atom, and the amplitude of the Gaussian is a "
             "fixed number: a partial charge, a well depth, a hydrogen-bond strength, a one in "
             "an indicator field. Here a small network on the atoms of the molecular graph "
             "produces a correction to every amplitude (`cmfqsar/amplitudes.py`). The network "
             "has shared weights for all atoms, it starts at zero, and a penalty holds it near "
             "zero. The model stays the kernel model: its coefficients are the closed-form "
             "ridge solution for the current amplitudes, and it keeps its co-field. The "
             "attenuations and the field weights are those of the grid search of plan item "
             "ND1 and are not trained. The sets and the splits (random, 70 % training, 15 % "
             "validation, 15 % test) are those of ND1 and ND2. Every score is on the test "
             "rows.", "",
             f"What each row trained: **{FIXED}** the ridge only (the reference); "
             f"**{ALONE}** the ridge and a network that reads one atom; **{NEIGHBOURS}** the "
             "ridge and a network in which every atom also reads its bonded neighbours, two "
             "rounds. These rows keep the evaluation with the lowest error on the validation "
             f"rows (early stopping). **{TO_MINIMUM}** keeps the evaluation with the lowest "
             f"objective of the training of *{NEIGHBOURS}*. **Layer** rows have the Gaussian "
             "kernel layer on top. The network has "
             f"{NETWORK['hidden']} hidden units; the penalty is {NETWORK['penalty']:g} times "
             "the mean squared correction, in units of the field scale.", ""]
    compas = collect("compas-1D")
    if compas:
        k = len(compas)
        lines += [f"## COMPAS-1D ({compas[0]['n']} molecules, {k} seeds, "
                  f"{compas[0]['landmarks']} landmark molecules)", "",
                  "The gap between the highest occupied and the lowest unoccupied orbital, in "
                  "eV. Mean ± standard deviation over the seeds. The amplitudes of "
                  + ", ".join(f"`{f}`" for f in compas[0]["learned_fields"])
                  + " are corrected; the ring fields are not.", "",
                  "| Model | Test R² | Mean absolute error, eV |", "|---|---:|---:|"]
        for key in ROWS:
            if all(key in r["models"] for r in compas):
                lines.append(f"| {key} | {RL._pm(compas, key)} | {RL._pm(compas, key, 'mae')} |")
        for key, label in ND2_ROWS:
            if all(key in r["models"] for r in compas):
                lines.append(f"| {label} | {RL._pm(compas, key)} | "
                             f"{RL._pm(compas, key, 'mae')} |")
        lines += ["", "The rows of ND2 are copied from its result files for the same test "
                  "rows.", "", "### Paired differences on COMPAS-1D", "",
                  "Difference of the absolute error on the same test molecules, in eV, with a "
                  "95 % bootstrap interval. A molecule in the test part of several seeds "
                  "counts once. A negative value means that the first model has the smaller "
                  "error.", "", "| Number | Comparison | Difference, eV |", "|---:|---|---:|"]
        for num, a, b, _what in PAIRS:
            if all(a in r["models"] and b in r["models"] for r in compas):
                lines.append(f"| {num} | {a} against {b} | "
                             f"{RL._pair(RL.paired(compas, a, b), 4)} |")
        lines += ["", "### What the training changed on COMPAS-1D", "",
                  "The criterion is the out-of-fold squared error inside the training part, "
                  "divided by the variance of the activity. The validation error is in eV.",
                  ""] + _changes(compas, ROWS)
        lines += ["", "Root mean square of the correction over the corrected atoms of the "
                  "training molecules, in units of the field scale (the root mean square of "
                  "the fixed amplitude):", ""] + _corrections(compas, ROWS[1:4] + ROWS[5:])
        part = _reading("compas-1D", reading, compas)
        if part:
            lines += ["", "The amplitudes and the co-field of the trained models against "
                      "those of the model with the fixed amplitudes, over the atoms of the "
                      f"landmark molecules and on a grid of points {SPACING:g} Å apart; means "
                      "over the seeds:", ""] + part
            lines += ["", f"Mean correction of *{NEIGHBOURS}* for every atom type, in units "
                      "of the field scale, with the root mean square in brackets:", ""] \
                + _by_type("compas-1D", reading)
        lines.append("")
    series = {s: collect(s) for s in SUTHERLAND}
    series = {s: r for s, r in series.items() if r}
    if series:
        keys = [key for key in ROWS[:4]
                if all(key in r["models"] for runs in series.values() for r in runs)]
        nd2 = [(key, label) for key, label in ND2_ROWS[:3]
               if all(key in r["models"] for runs in series.values() for r in runs)]
        lines += [f"## The Sutherland series ({len(series)} series, "
                  f"{len(next(iter(series.values())))} seeds each)", "",
                  "Every training molecule is a landmark. The amplitudes of all five fields "
                  "are corrected. Test R², mean over the seeds. The column before the last is "
                  "the mean over the series. The last column is the median over all splits "
                  f"of all series. The series of the gate is {GATE_SERIES}, named before any "
                  "run.", "",
                  "| Model | " + " | ".join(series) + " | Mean over the series | Median over "
                  "all splits |", "|---|" + "---:|" * (len(series) + 2)]
        for key, label in [(key, key) for key in keys] + nd2:
            vals = [np.mean([r["models"][key]["r2"] for r in runs]) for runs in series.values()]
            every = [r["models"][key]["r2"] for runs in series.values() for r in runs]
            lines.append(f"| {label} | " + " | ".join(f"{v:+.3f}" for v in vals)
                         + f" | {np.mean(vals):+.3f} | {np.median(every):+.3f} |")
        lines += ["", "### Paired differences on the Sutherland series", "",
                  "Difference of the absolute error on the same test molecules, in units of "
                  "the activity (the negative decimal logarithm of a concentration), with a "
                  "95 % bootstrap interval. A negative value means that the first model has "
                  "the smaller error.", "",
                  "| Number | Comparison | " + " | ".join(series) + " | Series better | "
                  "Series worse |", "|---:|---|" + "---:|" * (len(series) + 2)]
        for num, a, b, _what in PAIRS:
            if a not in keys or b not in keys:
                continue
            ps = [RL.paired(runs, a, b) for runs in series.values()]
            better = sum(1 for q in ps if q["excludes_zero"] and q["mean"] < 0)
            worse = sum(1 for q in ps if q["excludes_zero"] and q["mean"] > 0)
            lines.append(f"| {num} | {a} against {b} | "
                         + " | ".join(f"{q['mean']:+.3f} [{q['low']:+.3f}, {q['high']:+.3f}]"
                                      for q in ps) + f" | {better} | {worse} |")
        every = [r for runs in series.values() for r in runs]
        lines += ["", "### What the training changed on the Sutherland series", "",
                  "Means over all series and seeds. The criterion is the out-of-fold squared "
                  "error inside the training part, divided by the variance of the activity.",
                  ""] + _changes(every, keys)
        lines += ["", "Root mean square of the correction over the corrected atoms of the "
                  "training molecules, in units of the field scale:", ""] \
            + _corrections(every, keys[1:])
        part = _reading(GATE_SERIES, reading, series.get(GATE_SERIES, []))
        if part:
            lines += ["", f"The amplitudes and the co-field of the trained models of "
                      f"{GATE_SERIES} against those of the model with the fixed amplitudes; "
                      "means over the seeds:", ""] + part
            lines += ["", f"Mean correction of *{NEIGHBOURS}* on {GATE_SERIES} for every atom "
                      "type that holds at least 1 % of the corrected atoms, in units of the "
                      "field scale, with the root mean square in brackets:", ""] \
                + _by_type(GATE_SERIES, reading)
        lines.append("")
    gate = []
    for label, runs in (("COMPAS-1D", compas), (GATE_SERIES, series.get(GATE_SERIES, []))):
        if runs:
            q = RL.paired(runs, NEIGHBOURS, FIXED)
            gate.append((label, q))
    if gate:
        lines += ["## The gate", "",
                  f"The item continues only if *{NEIGHBOURS}* has a smaller absolute error "
                  f"than *{FIXED}* on COMPAS-1D and on {GATE_SERIES}, with an interval that "
                  "excludes zero on both.", "",
                  "| Set | Difference of the absolute error | Gate |", "|---|---:|---|"]
        for label, q in gate:
            met = q["excludes_zero"] and q["mean"] < 0
            lines.append(f"| {label} | {RL._pair(q, 4)} | {'met' if met else 'not met'} |")
        lines.append("")
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", action="append", default=[], choices=["pack", "train", "reading"])
    ap.add_argument("--sets", default=None, help="comma-separated; default: all")
    ap.add_argument("--seeds", default=None, help="comma-separated; default: all of a set")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if there is one)")
    ap.add_argument("--models", default=None, help="labels separated by ;")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print every evaluation")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    sets = tuple(args.sets.split(",")) if args.sets else SETS
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else None
    for stage in args.stage:
        if stage == "pack":
            pack_stage(sets)
        elif stage == "train":
            train_stage(sets, seeds, args.device, args.force,
                        args.models.split(";") if args.models else None, args.verbose)
        elif stage == "reading":
            reading_stage(sets, seeds)
    if args.report:
        print(write_table())


if __name__ == "__main__":
    main()
