#!/usr/bin/env python
"""
Reciprocal-space channels of a crystal against the real-space autocorrelation (plan item M6).

The periodic field autocorrelation of `cmfqsar/periodic.py` is a sum over a neighbour list, so
it sees nothing beyond its cutoff radius. The typed structure factors of `cmfqsar/reciprocal.py`
are sums over the atoms of one cell with no neighbour list at all, so they carry order of any
range, and the width of the Gaussian field sets how much fine detail reaches them. This script
measures what that is worth.

The question, and the test that answers it
------------------------------------------
"Do the reciprocal channels carry information a real-space cutoff cannot reach?" is not answered
by comparing the two descriptors at their best settings, because a 10 angstrom cutoff on a cell
of five atoms already covers the structure several times over. It is answered by **shortening the
cutoff**: the real-space descriptor is measured at 4, 6 and 10 angstrom, alone and with the
reciprocal channels added. If the channels carry long-range order, adding them to the 4 angstrom
descriptor recovers what the 10 angstrom descriptor has, and the gain falls as the cutoff grows.
If they carry nothing the cutoff does not already hold, nothing changes anywhere.

The two descriptions are combined as a sum of their kernels, each normalised on its own,
`K = (1 - t) K_real + t K_reciprocal`, with the mixing weight `t` chosen inside every training
fold. The weight is itself a measurement: it says how much of the signal the training set puts
in reciprocal space.

The data
--------
The three tasks of Matbench v0.1 that the plan item names, with the published five folds:

* `matbench_perovskites` -- 18,928 ABO3 cells of five atoms, formation energy (eV per atom).
  Small cells: a real-space cutoff of 10 angstrom covers 2.5 lattice constants in every
  direction, so this is the task where the reciprocal channels are expected to add least.
* `matbench_log_kvrh` -- 10,987 crystals, log10 of the bulk modulus (GPa).
* `matbench_log_gvrh` -- 10,987 crystals, log10 of the shear modulus (GPa).
  The two moduli sets hold cells of many sizes and compositions, so long-range order has room
  to matter; an elastic constant is also a property of the whole cell and not of one bond.

A random subset of `--limit` structures of each task is used, so that the twenty-odd kernels of
the comparison fit in memory and the run takes hours rather than days. The folds are the
published ones over that subset, and every descriptor sees the same folds and the same ridge
grid.

What else is measured
---------------------
* **The identity between the two descriptions** (`--stage parseval`). The real-space lattice sum
  of a Gaussian and the sum over the reciprocal lattice are the same number (the module
  docstring derives it). The residual over a sample of structures is reported, because it is the
  evidence that the two pieces of code compute one object and not two.
* **The width as a resolution** (`--stage resolution`). The form factor exp(-a^2 q^2) damps the
  high-q channels, so the width `a` of the Gaussian field is a resolution limit. The accuracy is
  measured over a grid of widths and cutoffs, which says what resolution the property needs.
* **The pattern in q** (`--stage pattern`). The weights of a fitted linear model, read per pair
  of field types and per reciprocal length: which spacing of planes in the structure raises the
  property. It is the reciprocal-space counterpart of the co-pattern.

Usage
-----
    python benchmarks/run_reciprocal.py                      # every stage, then the table
    python benchmarks/run_reciprocal.py --limit 600           # a quicker run
    python benchmarks/run_reciprocal.py --stage fit --tasks perovskites
    python benchmarks/run_reciprocal.py --stage concept      # the figure that explains it
    python benchmarks/run_reciprocal.py --stage parseval
    python benchmarks/run_reciprocal.py --stage resolution
    python benchmarks/run_reciprocal.py --stage pattern
    python benchmarks/run_reciprocal.py --report
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar import periodic, reciprocal  # noqa: E402
from cmfqsar.invariant import autocorrelation_kernel, channel_scale, field_pairs  # noqa: E402
from cmfqsar.model import CMFModel, _Learner, cv_predict, r2_score, rmse  # noqa: E402

CACHE = os.path.join(HERE, "cache", "matbench")
FEATURES = os.path.join(HERE, "cache", "reciprocal")
RESULTS = os.path.join(HERE, "results", "reciprocal")

#: the three tasks: the Matbench file, the column to predict, what it is and its unit
TASKS: Dict[str, dict] = {
    "perovskites": {"file": "matbench_perovskites", "target": "e_form",
                    "label": "formation energy", "unit": "eV/atom"},
    "log_kvrh": {"file": "matbench_log_kvrh", "target": "log10(K_VRH)",
                 "label": "bulk modulus", "unit": "log10 GPa"},
    "log_gvrh": {"file": "matbench_log_gvrh", "target": "log10(G_VRH)",
                 "label": "shear modulus", "unit": "log10 GPa"},
}
#: the Matbench v0.1 fold seed, so that the folds are the published ones
SPLIT_SEED = 18012019
OUTER, INNER = 5, 10
LAMS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)
LIMIT = 1200
RUNS, SEED = 2000, 0

#: the real-space descriptor at three cutoffs, with the widths each cutoff allows
REAL: Dict[str, Tuple[float, Tuple[float, ...]]] = {
    "4 A": (4.0, (0.35, 0.5)),
    "6 A": (6.0, (0.35, 0.5)),
    "10 A": (10.0, (0.35, 0.5, 0.75)),
}
AC_TAPER = 2.0
#: the reciprocal channels: the resolution limit q_max and the width of the Gaussian field
REC: Tuple[Tuple[float, float], ...] = ((4.0, 0.3), (8.0, 0.2), (8.0, 0.3), (8.0, 0.5),
                                        (12.0, 0.2))
REC_STEP = 0.25
#: the settings of the reciprocal channels the mixed kernels draw on, and the mixing weights.
#: The mixed family searches the same real-space settings as the real-space family does, so that
#: a difference between the two cannot come from one of them having been given a wider choice.
REC_MIX = ((8.0, 0.3), (8.0, 0.5))
MIX_WEIGHTS = (0.25, 0.5, 0.75)
#: the setting whose pattern is read and whose points are counted in the table
REC_READ = (8.0, 0.3)
#: the fields, as `run_periodic_overlap.py` chooses them: the three physico-chemical fields of
#: the module and the inverse square root of the mass.  Indicator fields per element would
#: predict better and would multiply the number of channel blocks by twenty; the comparison here
#: is between two descriptions of one field set, not between field sets.
FIELDS = periodic.INORGANIC_FIELDS + ("prop:inverse_sqrt_mass",)
#: electronegativity differences rather than pymatgen's oxidation-state guess, which is slow on
#: thousands of structures and fails on the mixed-valence members
CHARGES = "electronegativity"
#: structures whose Parseval residual is computed, and the widths it is computed at
PARSEVAL_SAMPLE = 40
PARSEVAL_WIDTHS = (0.3, 0.5, 0.8)
#: the resolution grid: widths of the Gaussian field, at the two cutoffs
RES_WIDTHS = (0.15, 0.2, 0.3, 0.4, 0.5, 0.7)
RES_Q_MAX = (8.0, 12.0)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def result(name: str) -> str:
    return os.path.join(RESULTS, name)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_default)
        fh.write("\n")


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# The data
# --------------------------------------------------------------------------- #
def fetch(task: str) -> dict:
    """The Matbench table of one task, downloaded once into benchmarks/cache/matbench."""
    name = TASKS[task]["file"]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.json.gz")
    if not os.path.exists(path):
        url = f"https://ml.materialsproject.org/projects/{name}.json.gz"
        log(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (cmfqsar)"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as fh:
            fh.write(r.read())
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def load_set(task: str, limit: int = LIMIT):
    """(the field table, the target, the field set, the chosen rows) of one task's subset."""
    atoms, y, ids = periodic.structures_from_matbench(fetch(task), target=TASKS[task]["target"])
    rng = np.random.RandomState(0)
    rows = np.sort(rng.choice(len(atoms), min(int(limit), len(atoms)), replace=False))
    atoms = [atoms[i] for i in rows]
    y = np.asarray(y, float)[rows]
    names = [f"{a.get_chemical_formula()}_{ids[i]}" for a, i in zip(atoms, rows)]
    fs = periodic.inorganic_fieldset(elements=[], physchem=FIELDS,
                                     charge_model=CHARGES, alpha=1.0)
    table = periodic.crystal_table(atoms, fs, names)
    return table, y, fs, rows


def folds(n: int):
    from sklearn.model_selection import KFold

    return list(KFold(n_splits=OUTER, shuffle=True, random_state=SPLIT_SEED).split(np.arange(n)))


# --------------------------------------------------------------------------- #
# The two feature matrices
# --------------------------------------------------------------------------- #
def _cached(path: str, build):
    """A feature matrix, computed once and kept in benchmarks/cache/reciprocal."""
    if os.path.isfile(path):
        with np.load(path) as z:
            return z["F"], [int(b) for b in z["blocks"]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    F, blocks = build()
    np.savez_compressed(path, F=F, blocks=np.asarray(blocks, int))
    log(f"  {os.path.basename(path)}: {F.shape[0]} x {F.shape[1]} in {time.time() - t0:.0f} s")
    return F, blocks


def real_features(task: str, table, limit: int, r_max: float, sigma: float):
    """The periodic autocorrelation of the set, and the widths of its channel blocks."""
    n_fields = table.weights.shape[1]

    def build():
        spec = periodic.PeriodicSpec(r_max=r_max, step=sigma, sigma=sigma, taper=AC_TAPER,
                                     background=True)
        F = periodic.periodic_feature_matrix(table.molecules, spec, n_fields=n_fields)
        return F, periodic.periodic_block_sizes(spec, n_fields)

    name = f"{task}_{limit}_real_r{r_max:g}_s{sigma:g}.npz"
    return _cached(os.path.join(FEATURES, name), build)


def reciprocal_features(task: str, table, limit: int, q_max: float, form_sigma: float,
                        step: float = REC_STEP):
    """The reciprocal-space channels of the set, and the widths of its channel blocks."""
    n_fields = table.weights.shape[1]

    def build():
        spec = reciprocal.ReciprocalSpec(q_max=q_max, step=step, sigma=step,
                                         form_sigma=form_sigma)
        F = reciprocal.reciprocal_feature_matrix(table.molecules, spec, n_fields=n_fields)
        return F, reciprocal.reciprocal_block_sizes(spec, n_fields)

    name = f"{task}_{limit}_rec_q{q_max:g}_a{form_sigma:g}_s{step:g}.npz"
    return _cached(os.path.join(FEATURES, name), build)


def kernel_of(F: np.ndarray, blocks: Sequence[int]) -> np.ndarray:
    """The cosine-normalised linear kernel of a feature matrix, each block on its own scale."""
    sc = channel_scale(F, int(blocks[0]), blocks=list(blocks))
    return autocorrelation_kernel(F, scale=sc, n_bins=int(blocks[0]), blocks=list(blocks))


# --------------------------------------------------------------------------- #
# The kernels of one task
# --------------------------------------------------------------------------- #
def base_kernels(task: str, table, limit: int) -> Dict[str, np.ndarray]:
    """One kernel per setting of the two descriptors, which the families then draw on."""
    out: Dict[str, np.ndarray] = {}
    for name, (r_max, sigmas) in REAL.items():
        for sigma in sigmas:
            F, blocks = real_features(task, table, limit, r_max, sigma)
            out[f"real {name} sigma {sigma:g}"] = kernel_of(F, blocks)
    for q_max, a in REC:
        F, blocks = reciprocal_features(task, table, limit, q_max, a)
        out[f"reciprocal q {q_max:g} a {a:g}"] = kernel_of(F, blocks)
    return out


def families() -> Dict[str, List[dict]]:
    """The descriptors compared, each a list of settings the inner loop chooses between.

    A setting is either one base kernel, or the weighted sum of a real-space and a
    reciprocal-space kernel.
    """
    out: Dict[str, List[dict]] = {}
    for name, (_, sigmas) in REAL.items():
        out[f"real space, {name}"] = [{"keys": [f"real {name} sigma {s:g}"], "t": 0.0}
                                      for s in sigmas]
    out["reciprocal"] = [{"keys": [f"reciprocal q {q:g} a {a:g}"], "t": 0.0} for q, a in REC]
    for name, (_, sigmas) in REAL.items():
        out[f"real space, {name} + reciprocal"] = [
            {"keys": [f"real {name} sigma {s:g}", f"reciprocal q {q:g} a {a:g}"], "t": t}
            for s in sigmas for q, a in REC_MIX for t in MIX_WEIGHTS]
    return out


def joint(Ks: Dict[str, np.ndarray], setting: dict) -> np.ndarray:
    """The kernel of one setting: a base kernel, or a mixture of two."""
    keys, t = setting["keys"], float(setting["t"])
    if len(keys) == 1:
        return Ks[keys[0]]
    return (1.0 - t) * Ks[keys[0]] + t * Ks[keys[1]]


def label_of(setting: dict) -> str:
    if len(setting["keys"]) == 1:
        return setting["keys"][0]
    return f"{setting['keys'][0]} + {setting['keys'][1]} (t {setting['t']:g})"


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #
def evaluate(Ks: Dict[str, np.ndarray], settings: List[dict], y: np.ndarray) -> dict:
    """Nested cross-validation: the setting and the ridge chosen inside every training fold."""
    pred = np.full(len(y), np.nan)
    per_fold = []
    for k, (tr, te) in enumerate(folds(len(y))):
        best = None
        for setting in settings:
            K = joint(Ks, setting)
            K_tr = K[np.ix_(tr, tr)]
            for lam in LAMS:
                p = cv_predict(K_tr, y[tr], "krr", {"lam": lam}, folds=INNER, seed=0)
                err = rmse(y[tr], p)
                if best is None or err < best[0]:
                    best = (err, setting, lam)
        err, setting, lam = best
        K = joint(Ks, setting)
        learner = _Learner("krr", {"lam": lam}, True).fit(K[np.ix_(tr, tr)], y[tr])
        pred[te] = learner.predict(K[np.ix_(te, tr)])
        per_fold.append({"fold": k, "setting": label_of(setting), "t": float(setting["t"]),
                         "lam": lam, "inner_rmse": float(err),
                         "mae": float(np.mean(np.abs(pred[te] - y[te]))),
                         "rmse": float(rmse(y[te], pred[te])),
                         "r2": float(r2_score(y[te], pred[te]))})
        log(f"    fold {k}: {per_fold[-1]['setting']}, lam {lam:g}, "
            f"R2 {per_fold[-1]['r2']:+.3f}, MAE {per_fold[-1]['mae']:.4f}")
    return {"r2": float(r2_score(y, pred)), "rmse": float(rmse(y, pred)),
            "mae": float(np.mean(np.abs(pred - y))),
            "mae_sd": float(np.std([f["mae"] for f in per_fold])),
            "folds": per_fold, "predictions": pred.tolist()}


def bootstrap_difference(y: np.ndarray, a: np.ndarray, b: np.ndarray,
                         runs: int = RUNS, seed: int = SEED) -> Tuple[float, float, float]:
    """(difference in R2, low, high) for prediction `a` against `b`, resampling the structures."""
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(runs):
        idx = rng.randint(0, len(y), len(y))
        if np.var(y[idx]) <= 0:
            continue
        diffs.append(r2_score(y[idx], a[idx]) - r2_score(y[idx], b[idx]))
    return (float(r2_score(y, a) - r2_score(y, b)),
            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def fit(tasks: Sequence[str] = tuple(TASKS), limit: int = LIMIT, force: bool = False) -> dict:
    path = result("fit.json")
    out = jload(path) if os.path.isfile(path) and not force else {}
    if out.get("limit") != int(limit):
        out = {"limit": int(limit)}
    for task in tasks:
        log(f"{task}: {TASKS[task]['label']} ({TASKS[task]['unit']})")
        table, y, fs, rows = load_set(task, limit)
        log(f"  {len(y)} structures, {np.mean([m.n_atoms for m in table.molecules]):.1f} atoms "
            f"per cell, {np.mean([m.volume for m in table.molecules]):.0f} A^3 per cell")
        Ks = base_kernels(task, table, limit)
        rec = out.setdefault(task, {})
        rec["n"] = int(len(y))
        rec["y_sd"] = float(np.std(y))
        rec["atoms_per_cell"] = float(np.mean([m.n_atoms for m in table.molecules]))
        rec["volume_per_cell"] = float(np.mean([m.volume for m in table.molecules]))
        rec["reciprocal_points"] = float(np.mean(
            [len(reciprocal.reciprocal_vectors(m.cell, REC_READ[0])[0]) for m in table.molecules]))
        rec.setdefault("descriptors", {})
        for name, settings in families().items():
            if name in rec["descriptors"] and not force:
                log(f"  {name}: there")
                continue
            log(f"  {name}")
            rec["descriptors"][name] = evaluate(Ks, settings, y)
            jdump(out, path)
        # what the reciprocal channels add to each cutoff, with an interval
        rec["gain"] = {}
        for name in REAL:
            a = np.asarray(rec["descriptors"][f"real space, {name} + reciprocal"]["predictions"])
            b = np.asarray(rec["descriptors"][f"real space, {name}"]["predictions"])
            d, lo, hi = bootstrap_difference(y, a, b)
            rec["gain"][name] = {"d_r2": d, "low": lo, "high": hi}
            log(f"  {name}: adding the reciprocal channels changes R2 by "
                f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        jdump(out, path)
        del Ks
    return out


# --------------------------------------------------------------------------- #
# The identity between the two descriptions
# --------------------------------------------------------------------------- #
def parseval(tasks: Sequence[str] = tuple(TASKS), limit: int = LIMIT,
             force: bool = False) -> dict:
    """The residual of the real-space / reciprocal-space identity over a sample of structures."""
    path = result("parseval.json")
    if os.path.isfile(path) and not force:
        log(f"parseval: {path} is there")
        return jload(path)
    out = {"sample": int(PARSEVAL_SAMPLE), "widths": list(PARSEVAL_WIDTHS), "tasks": {}}
    for task in tasks:
        table, _, _, _ = load_set(task, limit)
        rng = np.random.RandomState(0)
        pick = rng.choice(len(table.molecules), min(PARSEVAL_SAMPLE, len(table.molecules)),
                          replace=False)
        rec = {}
        for width in PARSEVAL_WIDTHS:
            worst = [reciprocal.parseval_residual(table.molecules[int(i)], width)["worst"]
                     for i in pick]
            rec[f"{width:g}"] = {"median": float(np.median(worst)), "worst": float(np.max(worst))}
            log(f"{task}, u = {width:g} A: median residual {np.median(worst):.2e}, "
                f"worst {np.max(worst):.2e}")
        out["tasks"][task] = rec
    jdump(out, path)
    return out


# --------------------------------------------------------------------------- #
# The width of the field as a resolution
# --------------------------------------------------------------------------- #
def resolution(task: str = "log_kvrh", limit: int = LIMIT, force: bool = False) -> dict:
    """The accuracy of the reciprocal channels over a grid of widths and resolution limits."""
    path = result("resolution.json")
    if os.path.isfile(path) and not force:
        log(f"resolution: {path} is there")
        return jload(path)
    table, y, _, _ = load_set(task, limit)
    out = {"task": task, "limit": int(limit), "widths": list(RES_WIDTHS),
           "q_max": list(RES_Q_MAX), "grid": {}}
    for q_max in RES_Q_MAX:
        for a in RES_WIDTHS:
            F, blocks = reciprocal_features(task, table, limit, q_max, a)
            K = kernel_of(F, blocks)
            rec = evaluate({"one": K}, [{"keys": ["one"], "t": 0.0}], y)
            out["grid"][f"q {q_max:g} a {a:g}"] = {
                "q_max": q_max, "form_sigma": a, "r2": rec["r2"], "mae": rec["mae"],
                "spacing": float(2.0 * np.pi / q_max)}
            log(f"  q_max {q_max:g}, a {a:g} A: R2 {rec['r2']:+.3f}, MAE {rec['mae']:.4f}")
            jdump(out, path)
    jdump(out, path)
    return out


# --------------------------------------------------------------------------- #
# The pattern in q
# --------------------------------------------------------------------------- #
def pattern(task: str = "log_kvrh", limit: int = LIMIT, force: bool = False) -> dict:
    """The channel weights of a fitted reciprocal model, per pair of field types and per q."""
    path = result("pattern.json")
    if os.path.isfile(path) and not force:
        log(f"pattern: {path} is there")
        return jload(path)
    table, y, fs, _ = load_set(task, limit)
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 1e-4},
                     kernel_source="reciprocal", normalize=True)
    model.rec_q_max, model.rec_step, model.rec_form_sigma = REC_READ[0], REC_STEP, REC_READ[1]
    model.fit(table, y)
    spec = model._ac_spec()
    P = reciprocal.reciprocal_pattern(model, len(fs.specs), spec, effect=True)
    names = [s.label for s in fs.specs]
    pairs = [names[f] if f == g else f"{names[f]} x {names[g]}"
             for f, g in field_pairs(len(names))]
    top = reciprocal.pattern_table(model, fs, spec, n=12)
    out = {"task": task, "limit": int(limit), "q_max": spec.q_max, "step": spec.step,
           "form_sigma": spec.form_sigma, "centres": spec.centres.tolist(),
           "spacings": [None if not np.isfinite(d) else float(d)
                        for d in reciprocal.spacings(spec)],
           "pairs": pairs, "weights": P.tolist(),
           "raw_weights": reciprocal.reciprocal_pattern(model, len(fs.specs), spec).tolist(),
           "top": [{"label": lab, "weight": w, "q": q} for lab, w, q in top],
           "r2_fit": float(r2_score(y, model.predict(table)))}
    jdump(out, path)
    figure(out)
    return out


def figure(rec: dict) -> None:
    """The pattern in q: one curve per pair of field types, and the strongest channels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = np.asarray(rec["centres"], float)
    P = np.asarray(rec["weights"], float)
    order = np.argsort(-np.abs(P).max(axis=1))[:5]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for k in order:
        ax.plot(q, P[k], lw=1.4, label=rec["pairs"][k])
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("reciprocal length q (1/A)")
    lengths = [d for d in (10.0, 5.0, 3.0, 2.0, 1.5, 1.0)
               if 0.0 < 2.0 * np.pi / d <= float(q[-1])]
    top_axis = ax.twiny()
    top_axis.set_xlim(ax.get_xlim())
    top_axis.set_xticks([2.0 * np.pi / d for d in lengths])
    top_axis.set_xticklabels([f"{d:g}" for d in lengths], fontsize=8)
    top_axis.set_xlabel("the length it describes, 2 pi / q (A)", fontsize=9)
    ax.set_ylabel("weight times the spread of the channel")
    ax.set_title(f"What the model reads at each reciprocal length: {rec['task']}")
    ax.legend(fontsize=7, frameon=False)
    top = rec["top"][:10][::-1]
    bx.barh(range(len(top)), [t["weight"] for t in top], color="#3b6ea5")
    bx.set_yticks(range(len(top)))
    bx.set_yticklabels([t["label"] for t in top], fontsize=7)
    bx.axvline(0.0, color="0.6", lw=0.8)
    bx.set_xlabel("weight times the spread of the channel")
    bx.set_title("The channels that move the prediction most")
    fig.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    fig.savefig(result("pattern.png"), dpi=150)
    plt.close(fig)
    log(f"figure: {result('pattern.png')}")


# --------------------------------------------------------------------------- #
# The concept, drawn on a pair of structures built for the purpose
# --------------------------------------------------------------------------- #
#: the host cell of the concept figure, and the two separations of the substituted sites
CONCEPT_REPEAT = (4, 1, 1)
CONCEPT_SITES = (8, 16)
CONCEPT_CUTOFF = 4.0


def concept_structures():
    """Two cells that differ only in the distance between two substituted sites.

    One host lattice (four cubic cells of silicon along x, 21.7 angstrom), two silicon atoms
    replaced by magnesium, 5.4 angstrom apart in one cell and 10.9 angstrom apart in the other.
    The two cells have the same composition and the same local environment around every atom, so
    everything that separates them lies beyond a few angstrom.
    """
    from ase.build import bulk

    host = bulk("Si", "diamond", a=5.43, cubic=True).repeat(CONCEPT_REPEAT)
    fs = periodic.inorganic_fieldset(["Si", "Mg"], physchem=["steric"], charge_model=CHARGES,
                                     alpha=1.0)
    out = []
    for other in CONCEPT_SITES:
        s = host.copy()
        s.symbols[0] = "Mg"
        s.symbols[other] = "Mg"
        d = float(np.linalg.norm(s.positions[other] - s.positions[0]))
        out.append((periodic.crystal_fields(s, fs, f"Mg-Mg {d:.1f} A"), d))
    return fs, out


def _cosine(F: np.ndarray, blocks: Sequence[int]) -> float:
    """The kernel the program uses, for the two rows of a feature matrix."""
    K = kernel_of(F, blocks)
    return float(K[0, 1])


def _pair_block(F: np.ndarray, blocks: Sequence[int], labels: Sequence[str],
                wanted: str) -> np.ndarray:
    """The columns of one pair-of-field-types block, named by its label."""
    start = 0
    for width in blocks:
        if wanted in labels[start]:
            return F[:, start:start + width]
        start += width
    raise KeyError(f"no channel block whose label holds {wanted!r}")


def concept(force: bool = False) -> dict:
    """The figure that explains what a reciprocal-space channel is and why it is not a cutoff."""
    path = result("concept.json")
    if os.path.isfile(path) and not force:
        log(f"concept: {path} is there")
        rec = jload(path)
        concept_figure(rec)
        return rec
    fs, pair = concept_structures()
    n_fields = len(fs.specs)
    cells = [c for c, _ in pair]
    gaps = [d for _, d in pair]
    # the real-space descriptor, at the short cutoff and at a long one
    rows = {}
    for r_max in (CONCEPT_CUTOFF, 12.0):
        spec = periodic.PeriodicSpec(r_max=r_max, step=0.25, sigma=0.25, taper=AC_TAPER)
        F = periodic.periodic_feature_matrix(cells, spec, n_fields=n_fields)
        blocks = periodic.periodic_block_sizes(spec, n_fields)
        labels = periodic.periodic_channel_labels(fs, spec)
        rows[f"real {r_max:g}"] = {
            "centres": spec.centres.tolist(),
            "mg": _pair_block(F, blocks, labels, "CIF Mg @").tolist(),
            "kernel": _cosine(F, blocks)}
    # the reciprocal channels
    spec = reciprocal.ReciprocalSpec(q_max=REC_READ[0], step=REC_STEP, sigma=REC_STEP,
                                     form_sigma=REC_READ[1])
    F = reciprocal.reciprocal_feature_matrix(cells, spec, n_fields=n_fields)
    blocks = reciprocal.reciprocal_block_sizes(spec, n_fields)
    labels = reciprocal.reciprocal_channel_labels(fs, spec)
    recip = {"centres": spec.centres.tolist(),
             "mg": _pair_block(F, blocks, labels, "CIF Mg @").tolist(),
             "kernel": _cosine(F, blocks), "q_max": spec.q_max,
             "form_sigma": spec.form_sigma,
             "envelope": {f"{a:g}": reciprocal.ReciprocalSpec(form_sigma=a).form_factor(
                 spec.centres).tolist() for a in (0.2, 0.3, 0.5)}}
    # the identity between the two descriptions, over a range of widths
    widths = [0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    pairs_of = [(0, 0)]                       # the steric field with itself: one curve, not many
    identity = {"widths": widths,
                "real": [float(reciprocal.gaussian_lattice_sum(cells[0], u, pairs_of)[0])
                         for u in widths],
                "reciprocal": [float(reciprocal.gaussian_reciprocal_sum(cells[0], u, pairs_of)[0])
                               for u in widths]}
    identity["worst"] = float(np.max(np.abs(
        (np.array(identity["real"]) - np.array(identity["reciprocal"]))
        / np.maximum(np.abs(identity["real"]), 1e-300))))
    out = {"gaps": gaps, "names": [c.name for c in cells], "cutoff": CONCEPT_CUTOFF,
           "cell_x": float(cells[0].cell[0, 0]), "n_atoms": int(cells[0].n_atoms),
           "positions": [np.asarray(c.coords, float)[:, :2].tolist() for c in cells],
           "symbols": [list(c.symbols) for c in cells],
           "real": rows, "reciprocal": recip, "identity": identity}
    jdump(out, path)
    log(f"concept: real-space kernel at {CONCEPT_CUTOFF:g} A "
        f"{rows[f'real {CONCEPT_CUTOFF:g}']['kernel']:.6f}, at 12 A "
        f"{rows['real 12']['kernel']:.6f}, reciprocal {recip['kernel']:.6f}; "
        f"identity worst {identity['worst']:.1e}")
    concept_figure(out)
    return out


def concept_figure(rec: dict) -> None:
    """Four panels: the two structures, the two descriptions of them, and the identity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    (ax, bx), (cx, dx) = axes
    colours = ("#2f6fad", "#c2562a")

    # (a) the two cells
    for k, (pos, syms, gap) in enumerate(zip(rec["positions"], rec["symbols"], rec["gaps"])):
        P = np.asarray(pos, float)
        y = -k * 8.0
        si = [i for i, s in enumerate(syms) if s != "Mg"]
        mg = [i for i, s in enumerate(syms) if s == "Mg"]
        ax.scatter(P[si, 0], P[si, 1] * 0.45 + y, s=9, c="0.75", edgecolors="none")
        ax.scatter(P[mg, 0], P[mg, 1] * 0.45 + y, s=70, c=colours[k], edgecolors="k", lw=0.5,
                   zorder=3)
        x0, x1 = sorted(P[mg, 0])
        ax.annotate("", xy=(x1, y + 4.0), xytext=(x0, y + 4.0),
                    arrowprops=dict(arrowstyle="<->", color=colours[k], lw=1.2))
        ax.text((x0 + x1) / 2, y + 4.6, f"{gap:.1f} A", ha="center", fontsize=9,
                color=colours[k])
    ax.set_xlim(-1.5, rec["cell_x"] + 1.5)
    ax.set_ylim(-11.5, 6.5)
    ax.set_yticks([])
    ax.set_xlabel("x (A), one cell of " + f"{rec['cell_x']:.1f} A")
    ax.set_title("(a) Two cells: the same composition, the same local environments")
    ax.text(0.5, -10.8, "silicon host; the two substituted sites are the coloured atoms",
            fontsize=8, color="0.35")

    # (b) the real-space channels of the Mg-Mg pair
    key = f"real {rec['cutoff']:g}"
    long_ = rec["real"]["real 12"]
    q = np.asarray(long_["centres"], float)
    for k in range(2):
        bx.plot(q, np.asarray(long_["mg"], float)[k], lw=1.5, color=colours[k],
                label=f"{rec['gaps'][k]:.1f} A apart")
    bx.axvspan(0.0, rec["cutoff"], color="0.88", zorder=0)
    bx.axvline(rec["cutoff"], color="0.4", lw=1.0, ls="--")
    bx.text(rec["cutoff"] * 0.5, bx.get_ylim()[1] * 0.92, "inside a 4 A cutoff:\nthe two are the "
            "same", ha="center", va="top", fontsize=8.5, color="0.25")
    bx.set_xlabel("distance r (A)")
    bx.set_ylabel("channel of the Mg-Mg pair")
    bx.set_title("(b) Real space: the difference is outside the cutoff")
    bx.legend(fontsize=8, frameon=False, loc="upper right")
    lo, hi = bx.get_ylim()
    bx.set_ylim(lo - 0.28 * (hi - lo), hi)        # room under the curves for the box
    bx.text(0.98, 0.02, f"kernel at 4 A: {rec['real'][key]['kernel']:.6f}\n"
                        f"kernel at 12 A: {long_['kernel']:.4f}",
            transform=bx.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(fc="white", ec="0.8", lw=0.6))

    # (c) the reciprocal channels of the same pair
    r = rec["reciprocal"]
    qq = np.asarray(r["centres"], float)
    for k in range(2):
        cx.plot(qq, np.asarray(r["mg"], float)[k], lw=1.5, color=colours[k],
                label=f"{rec['gaps'][k]:.1f} A apart")
    peak = float(np.abs(np.asarray(r["mg"], float)).max())
    for a, env in sorted(r["envelope"].items(), key=lambda kv: float(kv[0])):
        e = np.asarray(env, float)
        cx.plot(qq, e * peak, lw=0.9, ls=":", color="0.45")
        # each envelope is named where it has fallen to half, so that the three names do not
        # land on top of each other at the right edge
        k = int(np.argmin(np.abs(e - 0.5)))
        cx.text(qq[k], e[k] * peak, f" a = {a} A", fontsize=7.5, color="0.45", va="bottom")
    cx.set_xlabel("reciprocal length q (1/A)")
    cx.set_ylabel("channel of the Mg-Mg pair")
    cx.set_title("(c) Reciprocal space: no cutoff, and the two differ")
    cx.legend(fontsize=8, frameon=False, loc="upper right")
    cx.text(0.64, 0.56, f"kernel: {r['kernel']:.4f}", transform=cx.transAxes, ha="left",
            va="bottom", fontsize=8.5, bbox=dict(fc="white", ec="0.8", lw=0.6))
    # The length a channel describes is 2 pi / q, which has no value at q = 0, so the second
    # axis carries chosen lengths rather than a transformed copy of the first.
    lengths = [d for d in (10.0, 5.0, 3.0, 2.0, 1.5, 1.0)
               if 0.0 < 2.0 * np.pi / d <= float(r["q_max"])]
    top = cx.twiny()
    top.set_xlim(cx.get_xlim())
    top.set_xticks([2.0 * np.pi / d for d in lengths])
    top.set_xticklabels([f"{d:g}" for d in lengths], fontsize=8)
    top.set_xlabel("the length it describes, 2 pi / q (A)", fontsize=9)

    # (d) the identity
    idn = rec["identity"]
    u = np.asarray(idn["widths"], float)
    dx.plot(u, idn["real"], lw=3.0, color="#2f6fad", alpha=0.45,
            label="real space: the lattice sum over the images")
    dx.plot(u, idn["reciprocal"], lw=1.3, color="#c2562a", ls="--",
            label="reciprocal space: the sum over G")
    dx.set_xlabel("width u of the Gaussian (A)")
    dx.set_ylabel("the two sides of the identity")
    dx.set_title("(d) The two descriptions are one object")
    dx.legend(fontsize=8, frameon=False, loc="upper left")
    dx.text(0.98, 0.02, f"largest relative difference: {idn['worst']:.0e}",
            transform=dx.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(fc="white", ec="0.8", lw=0.6))
    fig.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    fig.savefig(result("concept.png"), dpi=150)
    plt.close(fig)
    log(f"figure: {result('concept.png')}")


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def report() -> str:
    """benchmarks/results/reciprocal/table.md from whatever stages have run."""
    lines = ["# Reciprocal-space channels of a crystal (plan item M6)", ""]
    fit_path, par_path = result("fit.json"), result("parseval.json")
    res_path, pat_path = result("resolution.json"), result("pattern.json")
    if os.path.isfile(fit_path):
        rec = jload(fit_path)
        tasks = [t for t in TASKS if t in rec]
        lines += [f"Nested cross-validation over the five published Matbench folds of a random "
                  f"subset of {rec.get('limit', LIMIT)} structures of each task. The setting of "
                  f"the descriptor, the mixing weight and the ridge are chosen inside every "
                  f"training fold. R2 over the held-out predictions; MAE in the unit of the "
                  f"task.", ""]
        lines += ["| The set | Structures | Atoms per cell | Reciprocal-lattice points below "
                  "q = 8 1/A |", "|---|---:|---:|---:|"]
        for t in tasks:
            r = rec[t]
            lines.append(f"| {TASKS[t]['label']} ({TASKS[t]['unit']}) | {r['n']} | "
                         f"{r['atoms_per_cell']:.1f} | {r['reciprocal_points']:.0f} |")
        lines.append("")
        head = "| Descriptor | " + " | ".join(
            f"{TASKS[t]['label']}: R2 | MAE" for t in tasks) + " |"
        lines += [head, "|---" + "|---:" * (2 * len(tasks)) + "|"]
        names = ([f"real space, {n}" for n in REAL] + ["reciprocal"]
                 + [f"real space, {n} + reciprocal" for n in REAL])
        for name in names:
            cells = []
            for t in tasks:
                d = rec[t]["descriptors"].get(name)
                cells += ["n/a", "n/a"] if d is None else [f"{d['r2']:+.3f}", f"{d['mae']:.4f}"]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
        lines += ["**What the reciprocal channels add to each cutoff.** The change in R2 when "
                  "they are added to the real-space descriptor, with a 95 % interval from "
                  f"{RUNS} bootstrap resamples of the structures.", ""]
        lines += ["| Real-space cutoff | " + " | ".join(TASKS[t]["label"] for t in tasks) + " |",
                  "|---" + "|---:" * len(tasks) + "|"]
        for cut in REAL:
            cells = []
            for t in tasks:
                g = rec[t].get("gain", {}).get(cut)
                cells.append("n/a" if g is None
                             else f"{g['d_r2']:+.3f} [{g['low']:+.3f}, {g['high']:+.3f}]")
            lines.append(f"| {cut} | " + " | ".join(cells) + " |")
        lines.append("")
        lines += ["**The mixing weight chosen.** How much of the kernel the training folds gave "
                  "to the reciprocal channels, averaged over the five folds.", ""]
        lines += ["| Real-space cutoff | " + " | ".join(TASKS[t]["label"] for t in tasks) + " |",
                  "|---" + "|---:" * len(tasks) + "|"]
        for cut in REAL:
            cells = []
            for t in tasks:
                d = rec[t]["descriptors"].get(f"real space, {cut} + reciprocal")
                cells.append("n/a" if d is None
                             else f"{np.mean([f['t'] for f in d['folds']]):.2f}")
            lines.append(f"| {cut} | " + " | ".join(cells) + " |")
        lines.append("")
    if os.path.isfile(par_path):
        rec = jload(par_path)
        lines += ["## The identity between the two descriptions", "",
                  "The real-space lattice sum of a Gaussian against the sum over the reciprocal "
                  "lattice, over " + str(rec["sample"]) + " structures of each task: the largest "
                  "relative difference over the pairs of field types, median and worst over the "
                  "structures. The two sides are computed by the neighbour list of `periodic.py` "
                  "and by the structure factors of `reciprocal.py`, so the agreement is evidence "
                  "for both.", ""]
        lines += ["| The set | " + " | ".join(f"u = {w:g} A" for w in rec["widths"]) + " |",
                  "|---" + "|---:" * len(rec["widths"]) + "|"]
        for t, r in rec["tasks"].items():
            cells = [f"{r[f'{w:g}']['median']:.1e} (worst {r[f'{w:g}']['worst']:.1e})"
                     for w in rec["widths"]]
            lines.append(f"| {TASKS[t]['label']} | " + " | ".join(cells) + " |")
        lines.append("")
    if os.path.isfile(res_path):
        rec = jload(res_path)
        lines += ["## The width of the field as a resolution", "",
                  f"The reciprocal channels alone on the {TASKS[rec['task']]['label']} set, over "
                  f"a grid of widths `a` of the Gaussian field and resolution limits `q_max`. "
                  f"The form factor is exp(-a^2 q^2), so a wide field keeps only the first few "
                  f"reflections.", ""]
        lines += ["| Width a (A) | " + " | ".join(f"q_max {q:g} 1/A: R2" for q in rec["q_max"])
                  + " |", "|---" + "|---:" * len(rec["q_max"]) + "|"]
        for a in rec["widths"]:
            cells = []
            for q in rec["q_max"]:
                g = rec["grid"].get(f"q {q:g} a {a:g}")
                cells.append("n/a" if g is None else f"{g['r2']:+.3f}")
            lines.append(f"| {a:g} | " + " | ".join(cells) + " |")
        lines.append("")
    if os.path.isfile(pat_path):
        rec = jload(pat_path)
        lines += ["## The pattern in q", "",
                  f"A linear model on the reciprocal channels of the "
                  f"{TASKS[rec['task']]['label']} set (q_max {rec['q_max']:g} 1/A, field width "
                  f"{rec['form_sigma']:g} A), read back. Each channel is reported as its weight "
                  f"times the spread of that channel over the training structures, which is how "
                  f"much it moves the prediction and is the ranking `top_channels` uses; a "
                  f"positive value means that more intensity at that reciprocal length raises "
                  f"the property, and the plane spacing is 2 pi / q. Neighbouring channels are "
                  f"strongly correlated, so single weights alternate in sign and it is the "
                  f"envelope over q that carries the reading.", ""]
        lines += ["| Channel | Weight x spread | q (1/A) | Spacing (A) |",
                  "|---|---:|---:|---:|"]
        for t in rec["top"][:10]:
            d = "-" if t["q"] <= 0 else f"{2.0 * np.pi / t['q']:.2f}"
            lines.append(f"| {t['label']} | {t['weight']:+.4f} | {t['q']:.2f} | {d} |")
        lines += ["", "![The pattern in q](pattern.png)", ""]
    if os.path.isfile(result("concept.json")):
        rec = jload(result("concept.json"))
        key = f"real {rec['cutoff']:g}"
        lines += ["## What a reciprocal channel adds, on two cells built for the purpose", "",
                  f"Two cells of a silicon host with two substituted sites, "
                  f"{rec['gaps'][0]:.1f} and {rec['gaps'][1]:.1f} angstrom apart: the same "
                  f"composition and the same local environment around every atom. The kernel of "
                  f"the pair is {rec['real'][key]['kernel']:.6f} for the real-space descriptor "
                  f"with a {rec['cutoff']:g} angstrom cutoff, which cannot tell them apart at "
                  f"all; {rec['real']['real 12']['kernel']:.4f} with a 12 angstrom cutoff; and "
                  f"{rec['reciprocal']['kernel']:.4f} for the reciprocal channels, which have no "
                  f"cutoff. The two sides of the identity agree to "
                  f"{rec['identity']['worst']:.0e}.", "",
                  "![What a reciprocal-space channel adds](concept.png)", ""]
    text = "\n".join(lines) + "\n"
    os.makedirs(RESULTS, exist_ok=True)
    with open(result("table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    log(f"table: {result('table.md')}")
    return text


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["fit", "parseval", "resolution", "pattern", "concept"],
                   default=None)
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    p.add_argument("--limit", type=int, default=LIMIT)
    p.add_argument("--resolution-task", default="log_kvrh", choices=list(TASKS))
    p.add_argument("--force", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args(argv)
    if a.report:
        print(report())
        return 0
    if a.stage in (None, "concept"):
        concept(a.force)
    if a.stage in (None, "parseval"):
        parseval(a.tasks, a.limit, a.force)
    if a.stage in (None, "fit"):
        fit(a.tasks, a.limit, a.force)
    if a.stage in (None, "resolution"):
        resolution(a.resolution_task, a.limit, a.force)
    if a.stage in (None, "pattern"):
        pattern(a.resolution_task, a.limit, a.force)
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
