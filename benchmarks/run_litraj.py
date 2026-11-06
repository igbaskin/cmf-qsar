#!/usr/bin/env python
"""
The hop models of plan item BT7 on the published nebBVSE122k benchmark: the external check of BT7.

``benchmarks/prepare_litraj.py`` draws a sample of the nebBVSE122k set of Dembitskiy et al.,
npj Computational Materials 11, 131 (2025), which holds 122,421 bond-valence barriers of single
Li hops in 11,290 structures of the Materials Project.  This script fits on that sample the same
models ``benchmarks/run_hops.py`` fits on the stored public set of item BT7, with the same
protocol: 10 folds that hold out whole structures, the same ridge grid, the environment within
6 A of the hop midpoint, and a paired bootstrap that resamples structures, because the hops of
one cell are not independent of each other.

What the comparison can and cannot say:

* **The protocol matches.**  The published split of the collection holds out whole structures as
  well -- no structure of its test part appears in its training part -- so both sides measure
  prediction for a cell that was never seen, and the numbers are comparable in kind.
* **The label range differs, and q2 follows it.**  The barriers of the collection spread 2.4
  times as widely as those of the public set of item BT7 (standard deviation 1.97 eV against
  0.81 eV), because it covers every chemistry that holds lithium rather than ternary lithium
  oxides alone.  The same absolute error therefore gives a much higher q2 on it, so the table
  reports the error in eV beside q2 and prints the standard deviation of the sample.
* **The training set differs in size.**  The published models are fitted on 96,849 hops of 8,945
  structures; a kernel model holds one number per pair of samples, so this one is fitted on a
  sample about forty times smaller.  The learning curve at the end of the table is what says how
  much of the difference that explains.

Usage
-----
    python benchmarks/run_litraj.py                  # the run that writes the stored result
    python benchmarks/run_litraj.py --report         # rebuild the table from the stored result
    python benchmarks/run_litraj.py --max-structures 40    # a quick look
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar.hops import (Hop, HopModel, HopSet, bottleneck,  # noqa: E402
                          cross_validate_hop_model, framed_environments, hop_count_features,
                          hop_types)
from cmfqsar.localenv import (LAMS, LOCAL_SPEC, choose_lam, count_kernel,  # noqa: E402
                              local_features, local_kernel, paired_interval)
from cmfqsar.sites import site_kernel  # noqa: E402

from run_hops import _entry, fieldset_for, site_pair_features  # noqa: E402

RESULTS = os.path.join(HERE, "results", "litraj")
PREPARED = os.path.join(RESULTS, "prepared.json")
#: the envelope width the public set of item BT7 does best at; a list measures a width series
RHOS = (0.25,)
R_MAX = 6.0
FOLDS = 10
BOOTSTRAP = 1000
#: the cutoff of the site profile the site-resolved reference is taken at, in angstrom; item BT7
#: compares at the radius the hop environments themselves use
SITE_R = 6.0
#: how many structures the learning curve is drawn at
CURVE = (25, 50, 100, 200)
#: Table 2 of the paper: what its models reach on the test part of the collection, as
#: (model, mean absolute error in eV, root mean squared error in eV, R2)
PUBLISHED = (("M3GNet (graph network, dummy atom at the hop midpoint)", 0.21, 0.44, 0.95),
             ("XGBRF (gradient-boosted forest, Voronoi features of the hop)", 0.26, 0.49, 0.94),
             ("RF (random forest, the same features)", 0.26, 0.50, 0.94),
             ("Nequip (graph network)", 0.25, 0.52, 0.93),
             ("KNN (nearest neighbours, the same features)", 0.27, 0.58, 0.92),
             ("Allegro (graph network)", 0.38, 0.65, 0.89))
#: hops and structures of the training part of the collection, and of the test part those
#: numbers are measured on
PUBLISHED_TRAIN = (96849, 8945)
PUBLISHED_TEST = (13167, 1172)
#: trees in the random forest of the forest arms, which ask whether the way a description is
#: fitted explains the distance to the published models
FOREST_TREES = 300
#: the two fitting methods those arms use, as (name in the table, key in the result file)
FORESTS = (("random forest", "rf"), ("boosted forest", "gb"))
#: widths of the Gaussian kernel the nested arms search over, as gamma in
#: exp(-gamma * squared distance between two rows).  The rows are unit vectors, so a squared
#: distance lies in [0, 4]; this grid covers the useful range around the median distance of both
#: descriptions, and its ends are well past the width that does best.
GAMMAS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
#: folds of the inner split that chooses the width and the ridge inside a training part
INNER_FOLDS = 5
#: the widths the free-volume channel is measured at, as beta in exp(-beta * squared distance).
#: ``None`` is the radical Voronoi cell itself, which is the published feature, and a number is
#: the field version at that width; comparing them says what a field loses here.
FREE_BETAS = (None, 32.0, 8.0, 2.0, 0.5)
#: radius of the ball the free volume is integrated over, in angstrom, and the spacing of the
#: cubic grid inside it.  A site's cell in a framework is a few angstrom across, so the ball
#: holds it; the spacing sets how exactly a volume is resolved.
FREE_RADIUS = 3.5
FREE_STEP = 0.25
#: how the three sites of a hop are reduced to features: the published models use the smallest
#: and the largest of the three, and the midpoint alone is the place the ion has to pass
FREE_AGGREGATES = ("min", "max", "mean", "midpoint")
#: weights of the mixture of the three-distance kernel with the free-volume kernel.  The mixture
#: uses the Voronoi cell rather than a field version of it, because the cell is the published
#: feature and the question is what adding that feature to this program's kernel is worth.
FREE_MIX_WEIGHTS = (0.0, 0.1, 0.25, 0.5)
#: the three-distance descriptions the nested arm searches over, as (radius from the centre,
#: spacing of both grids) in angstrom.  The two-point description of an environment carries the
#: distance between two neighbours but not the two sides that meet at the centre, so it does not
#: resolve the angle there; a barrier is set by the window the ion passes through, which is a
#: relation among three atoms.  :mod:`cmfqsar.triangles` resolves it for an environment whose
#: first atom is the centre, which is what the midpoint of a hop is.
TRIANGLE_SPECS = ((4.0, 0.75), (3.0, 0.5), (5.0, 0.75))
#: the exponents of the three-distance kernel the same arm searches over; this is SOAP's zeta
TRIANGLE_POWERS = (1, 2)
#: weights of the convex mixture of the two-point and three-distance kernels, 0 being the
#: two-point kernel alone and 1 the three-distance kernel alone
MIX_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
#: how the rows a forest arm is given are scaled.  Both kernels here divide every row by its own
#: length, so the kernel never sees how large an environment's features are, only their
#: direction; a forest given the raw rows sees both.  "as the kernel" is therefore the arm that
#: compares the fitting methods on one description, and "raw" asks what the row length is worth.
FOREST_ROWS = ("as the kernel", "raw")
#: which of those row forms the forest arms on the three-distance description are given.  The
#: nested three-distance arm chose the setting that keeps the row length in every fold, so the
#: raw rows are the rows of the model those forests are compared with; the normalised form would
#: pair with a setting that was never chosen and would only repeat the row-length question the
#: two-point arms already answered.  It is left out for that reason and for its cost: the
#: three-distance feature matrix is 24 times wider than the two-point one, and one forest arm on
#: it takes hours.
TRIANGLE_FOREST_ROWS = "raw"


# ------------------------------------------------------------------- one run and what it wrote


def run_tag(a) -> str:
    """The name of one run of this script, used for every file that run writes.

    Every option that changes a stored number enters the tag -- which prepared sample is read,
    how much of it, the envelope widths, the radius of the environment, the cutoff of the
    site-resolved reference, the folds, the seed and the number of bootstrap draws -- and the
    options that only choose which parts of the run are done leave it alone.  Pointing
    ``--prepared`` at the stored sample gives the empty tag, which is why ``public.json`` has no
    suffix.
    """
    tag = ""
    if os.path.abspath(a.prepared) != os.path.abspath(PREPARED):
        stem = os.path.splitext(os.path.basename(a.prepared))[0]
        tag += "_" + stem
    if a.max_structures:
        tag += f"_n{a.max_structures}"
    if list(a.rho) != list(RHOS):
        tag += "_rho" + "-".join(f"{r:g}" for r in a.rho)
    if a.r_max != R_MAX:
        tag += f"_r{a.r_max:g}"
    if a.site_r != SITE_R:
        tag += f"_siter{a.site_r:g}"
    if a.folds != FOLDS:
        tag += f"_folds{a.folds}"
    if a.seed != 0:
        tag += f"_seed{a.seed}"
    if a.bootstrap != BOOTSTRAP:
        tag += f"_boot{a.bootstrap}"
    return tag


def run_provenance(results_dir: str = RESULTS, tag: str = "") -> Optional[dict]:
    """What the run with this tag was asked to do, or ``None`` if the run did not record it."""
    path = os.path.join(results_dir, f"run{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_provenance(args, tag: str, argv: Optional[List[str]] = None, **extra) -> str:
    """The command of this run and every option that changes a number, next to its results."""
    out = {"argv": list(argv) if argv is not None else sys.argv[1:], "tag": tag,
           "prepared": os.path.relpath(args.prepared, ROOT).replace(os.sep, "/"),
           "max_structures": args.max_structures, "rho": [float(r) for r in args.rho],
           "r_max": args.r_max, "site_r": args.site_r, "folds": args.folds,
           "seed": args.seed, "bootstrap": args.bootstrap,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.update(extra)
    path = os.path.join(RESULTS, f"run{tag}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    return path


# ----------------------------------------------------------------------------- the data


def load_prepared(path: str = PREPARED, max_structures: int = 0) -> tuple:
    """The prepared sample as a HopSet, with its oxidation states as a per-atom array.

    Returns the set and the document it was read from, whose header carries what the whole
    collection looks like.
    """
    from ase import Atoms
    doc = json.load(open(path))
    rows = doc["structures"]
    if max_structures:
        rows = rows[:max_structures]
    structures, names, hops = [], [], []
    for k, row in enumerate(rows):
        a = Atoms(symbols=row["symbols"], positions=np.asarray(row["positions"], float),
                  cell=np.asarray(row["cell"], float), pbc=True)
        a.set_array("oxi_states", np.array([float(row["oxidation"][s]) for s in row["symbols"]]))
        structures.append(a)
        names.append(f"{row['id']} {row['formula']}")
        cell = np.asarray(row["cell"], float)
        pos = np.asarray(row["positions"], float)
        for h in row["hops"]:
            off = np.asarray(h["offset"], int)
            shift = off @ cell
            hop = Hop(k, int(h["a"]), int(h["b"]), tuple(int(x) for x in off),
                      start=pos[h["a"]].copy(), vector=pos[h["b"]] + shift - pos[h["a"]],
                      shift=shift, y=float(h["y"]))
            hop.name = f"{row['id']}: {h['edge_id']}"
            hops.append(hop)
    return HopSet(structures, names, hops, "Li"), doc


# --------------------------------------------------------------- the fitting method on its own


def gaussian_kernel(K: np.ndarray, gamma: float) -> np.ndarray:
    """A Gaussian kernel on the rows whose cosine-normalised linear kernel is `K`.

    The kernels of this program are linear: :func:`cmfqsar.invariant.autocorrelation_kernel`
    rescales the channels, divides every row by its own length and takes the inner product, so
    ridge regression on that kernel is a **linear** model in the feature space.  The forest arms
    above beat it on identical rows, which says the barrier is not a linear function of those
    features.  This is the kernel that asks whether a kernel method can hold that nonlinearity:
    the rows are unit vectors, so the squared distance between rows i and j is ``2 - 2 K[i, j]``
    and the whole Gaussian family is a function of the matrix already computed -- no feature
    matrix and no field is touched again.

    The identity needs unit rows, so the caller passes a normalised kernel; the clamp at zero
    only removes rounding, since a squared distance cannot be negative.
    """
    d2 = np.maximum(2.0 - 2.0 * np.asarray(K, float), 0.0)
    return np.exp(-float(gamma) * d2)


def free_volume(positions: np.ndarray, radii: np.ndarray, centre_radius: float,
                radius: float = FREE_RADIUS, step: float = FREE_STEP,
                betas: Sequence = FREE_BETAS) -> List[float]:
    """The free volume at the origin, as a Voronoi cell and as a field, on one grid.

    The strongest feature of the published models on this benchmark is the smallest volume of the
    Voronoi polyhedra around the initial, final and midpoint sites of the hop, so the question is
    whether this program can carry that quantity.  It can, in a precise sense.  With ``d_i`` the
    squared distance from a point to atom *i* less that atom's squared radius, the radical (power)
    Voronoi cell of the centre is where ``d_centre`` is the smallest, and

        w(r) = exp(-beta d_centre) / sum over i of exp(-beta d_i)

    tends to the indicator function of that cell as `beta` grows, everywhere except its boundary.
    The denominator is a steric field of this program with unit weights and a Gaussian of width
    ``1 / sqrt(2 beta)``, so the volume is the integral of a ratio of fields and the Voronoi cell
    is its narrow-width limit.  A `beta` of ``None`` asks for the cell itself, which is what the
    published feature is, and a number asks for the field version at that width; measuring both
    says what the smoothing costs.

    `positions` are measured from the centre and `radii` are the atoms' radii, as the first atom
    of a :func:`cmfqsar.hops.point_environment` already arranges.  The integral is taken over a
    ball of `radius` on a cubic grid of spacing `step`, which is enough because the cell of a
    site in a framework is small; a cell that reaches the edge of the ball is cut off there, and
    the ball volume is the largest number this can return.  This lives in the benchmark rather
    than in the library because it is one arm of one question; it belongs in a module only if it
    earns a place.
    """
    pos = np.asarray(positions, float).reshape(-1, 3)
    rad = np.asarray(radii, float).reshape(-1)
    g = np.arange(-radius, radius + 1e-9, step)
    P = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    P = P[np.sum(P * P, axis=1) <= radius * radius]
    cell = step ** 3
    d_centre = np.sum(P * P, axis=1) - float(centre_radius) ** 2
    if len(pos):
        d_atoms = (np.sum((P[:, None, :] - pos[None, :, :]) ** 2, axis=2) - rad[None, :] ** 2)
        nearest = np.min(d_atoms, axis=1)
    else:
        d_atoms, nearest = None, np.full(len(P), np.inf)
    out = []
    for beta in betas:
        if beta is None:
            out.append(float(np.count_nonzero(d_centre < nearest)) * cell)
            continue
        floor = np.minimum(d_centre, nearest)          # subtracted for numerical safety
        num = np.exp(-float(beta) * (d_centre - floor))
        den = num.copy()
        if d_atoms is not None:
            den = den + np.sum(np.exp(-float(beta) * (d_atoms - floor[:, None])), axis=1)
        out.append(float(np.sum(num / den)) * cell)
    return out


def free_volume_name(beta) -> str:
    """The name of one free-volume setting: the Voronoi cell, or the field at a given width."""
    return "Voronoi cell" if beta is None else f"field, beta {beta:g}"


def free_volume_features(volumes: np.ndarray) -> np.ndarray:
    """The three sites of every hop reduced to the columns a model is given.

    `volumes` holds one free volume per hop and site, in the order start, midpoint, end.  The
    published models use the smallest and the largest of the three; the mean and the midpoint
    alone are added because the midpoint is the place the ion has to pass.  The columns are
    standardised, as the length-and-bottleneck reference of the first table is.
    """
    v = np.asarray(volumes, float)
    A = np.column_stack([v.min(axis=1), v.max(axis=1), v.mean(axis=1), v[:, 1]])
    return (A - A.mean(axis=0)) / np.maximum(A.std(axis=0), 1e-12)


def declared_triangle_settings() -> set:
    """The names of every three-distance setting the nested arm is allowed to choose from.

    The run builds these names and the test checks the stored result against them, so both read
    one declaration and cannot drift apart.  The last one keeps the row length that every other
    setting divides out, which is what item M15 of the plan asks about.
    """
    out = {f"reach {r:g}, step {s:g}, power {p}"
           for r, s in TRIANGLE_SPECS for p in TRIANGLE_POWERS}
    out.add(f"reach {TRIANGLE_SPECS[0][0]:g}, step {TRIANGLE_SPECS[0][1]:g}, "
            f"power 1, row length kept")
    return out


def nested_result(kernels: dict, targets, folds: int, seed: int, name: str,
                  inner: int = INNER_FOLDS, lams=LAMS) -> tuple:
    """Out-of-fold predictions with the kernel and the ridge chosen **inside** each training part.

    Every kernel row of the first table takes the ridge that maximises the very q2 it reports,
    which is optimistic by whatever that search is worth.  Searching over kernel widths as well
    would be more optimistic again, and a comparison against a forest that searches for nothing
    would mean nothing.  So these arms are fitted the way item M14's benchmark is: inside each
    training part the settings are chosen on an inner split that holds out whole structures, and
    the test fold is touched once, with the choice already made.

    `kernels` maps the name of a setting to its whole kernel matrix.  Passing a single entry
    gives one kernel the same treatment, which is how the linear kernel earns a row that can be
    set beside the Gaussian one.  Returns the result and what each outer fold chose.
    """
    from cmfqsar.localenv import LocalResult, cross_validate_lams, r2_score, rmse
    from cmfqsar.model import krr_fit
    y = np.asarray(targets.y, float)
    parts = targets.folds(folds, seed)
    if not parts:
        raise ValueError("there is nothing to cross-validate: no hop carries the barrier")
    pred = np.full(len(y), np.nan)
    chosen = []
    for te in parts:
        tr = np.setdiff1d(np.arange(len(y)), te)
        inside = targets.subset(tr)
        best = None
        for key, K in kernels.items():
            for r in cross_validate_lams(K[np.ix_(tr, tr)], inside, lams, folds=inner,
                                         seed=seed):
                if best is None or r.q2 > best[0]:
                    best = (r.q2, key, r.lam)
        _, key, lam = best
        chosen.append({"setting": key, "lam": float(lam)})
        K = kernels[key]
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], float(lam))
        pred[te] = K[np.ix_(te, tr)] @ a + b
    ok = np.isfinite(pred)
    out = LocalResult(name=name, predicted=pred, q2=r2_score(y[ok], pred[ok]),
                      error=rmse(y[ok], pred[ok]),
                      mae=float(np.mean(np.abs(y[ok] - pred[ok]))), lam=float("nan"),
                      n_samples=int(ok.sum()), n_molecules=targets.n_measured_molecules,
                      folds=len(parts), molecule=np.asarray(targets.molecule, int))
    return out, chosen


def kernel_rows(X: np.ndarray, n_bins: int = 1, scale: Optional[np.ndarray] = None) -> np.ndarray:
    """The rows the cosine-normalised kernel of this program is really built from.

    :func:`cmfqsar.invariant.autocorrelation_kernel` rescales the channels and then divides every
    row by its own length, so a kernel entry is the cosine of the angle between two rows and the
    length of a row is divided out.  A forest given the raw rows would therefore be given more
    than the kernel has, and the comparison would not be about the fitting method alone.  This
    returns the rows after both steps, so that a forest can be given exactly what the kernel got.
    Trees do not care about a rescaling of a single column, so only the row lengths matter here.
    """
    from cmfqsar.invariant import channel_scale
    A = np.asarray(X, float)
    s = channel_scale(A, n_bins) if scale is None else np.asarray(scale, float)
    A = A * s
    d = np.sqrt(np.maximum(np.sum(A * A, axis=1), 1e-300))
    return A / d[:, None]


def forest_result(X: np.ndarray, targets, folds: int, seed: int, name: str,
                  kind: str = "random forest"):
    """Out-of-fold predictions of a forest, on the folds the kernel models are given.

    The two published models that come closest to the graph network are a random forest and a
    gradient-boosted forest on hand-made features of the hop, not kernel models.  So the table
    above mixes two questions: whether this program's **description** of a hop is behind the
    published one, or whether fitting a description with a **kernel** is.  Fitting a forest on
    this program's own descriptions, over the same hops and the same folds, separates them.  If
    the forest lands near the kernel, the fitting method is not what the distance is made of.

    The random forest is the same kind of model the paper reports.  The boosted forest is the
    histogram boosting of scikit-learn, which stands in for the paper's XGBoost: it is a
    gradient-boosted forest of the same family, and it needs nothing this program does not
    already install.  A forest has no ridge, so these rows report none; nothing is chosen on a
    test fold.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

    from cmfqsar.localenv import LocalResult, r2_score, rmse
    X = np.asarray(X, float)
    y = np.asarray(targets.y, float)
    parts = targets.folds(folds, seed)
    if not parts:
        raise ValueError("there is nothing to cross-validate: no hop carries the barrier")
    pred = np.full(len(y), np.nan)
    for te in parts:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(tr) < 2:
            continue
        if kind == "random forest":
            m = RandomForestRegressor(n_estimators=FOREST_TREES, n_jobs=-1, random_state=seed)
        else:
            m = HistGradientBoostingRegressor(random_state=seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    ok = np.isfinite(pred)
    return LocalResult(name=name, predicted=pred, q2=r2_score(y[ok], pred[ok]),
                       error=rmse(y[ok], pred[ok]),
                       mae=float(np.mean(np.abs(y[ok] - pred[ok]))), lam=float("nan"),
                       n_samples=int(ok.sum()), n_molecules=targets.n_measured_molecules,
                       folds=len(parts), molecule=np.asarray(targets.molecule, int))


# ----------------------------------------------------------------------------- the run


def run(args, log) -> dict:
    t0 = time.time()
    hs, doc = load_prepared(args.prepared, args.max_structures)
    fs = fieldset_for(hs, "file")
    crystals = hs.crystals(fs)
    t = hs.targets()
    y = np.asarray(t.y, float)
    log(hs.describe())
    log(f"  the whole collection: {doc['n_index_hops']} hops of "
        f"{doc['n_index_structures']} structures and {doc['n_index_chemsys']} chemical systems, "
        f"barriers mean {doc['index_y_mean']:.3f}, standard deviation {doc['index_y_sd']:.3f} eV")
    log(f"  this sample: {len(t)} hops in {t.n_measured_molecules} structures, barriers mean "
        f"{y.mean():.3f}, standard deviation {y.std(ddof=1):.3f} eV; "
        f"fields: {', '.join(fs.names)}")
    out = {"n_structures": len(hs.structures), "n_samples": len(t), "fields": fs.names,
           "r_max": args.r_max, "folds": args.folds, "site_r": args.site_r,
           "y_mean": float(y.mean()), "y_sd": float(y.std(ddof=1)),
           "lengths": [float(np.min([hs.hops[r].length for r in t.atom])),
                       float(np.max([hs.hops[r].length for r in t.atom]))],
           "collection": {k: doc[k] for k in ("source", "dataset", "reference", "licence",
                                              "n_index_hops", "n_index_structures",
                                              "n_index_chemsys", "index_y_mean", "index_y_sd",
                                              "drawn", "max_per_structure", "dropped_hops")},
           "models": [], "pairs": []}
    # the references: counts of atoms in shells about the midpoint, and the two numbers a chemist
    # starts from.  The bottleneck is read off the environments, which are needed anyway.
    envs0 = hs.environments(crystals, rows=t.atom, rho=args.rho[0], r_max=args.r_max)
    for r, e in zip(t.atom, envs0):
        h = hs.hops[int(r)]
        h.bottleneck, h.bottleneck_at, h.bottleneck_atoms = bottleneck(e, h.vector)
    types = hop_types(envs0, min_atoms=20)
    counts_features = hop_count_features(envs0, types)
    counts_kernel = count_kernel(counts_features)
    ref = choose_lam(counts_kernel, t, folds=args.folds,
                     seed=args.seed, name="counts about the midpoint")
    log("  " + ref.summary())
    X = np.array([[hs.hops[r].length, hs.hops[r].bottleneck] for r in t.atom])
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    geo = choose_lam(X @ X.T, t, LAMS, folds=args.folds, seed=args.seed,
                     name="hop length and bottleneck")
    log("  " + geo.summary())
    sp = site_pair_features(hs, crystals, t.atom, args.site_r)
    sites = choose_lam(site_kernel(sp), t, folds=args.folds, seed=args.seed,
                       name=f"the two end sites (M5), cutoff {args.site_r:g} A")
    log("  " + sites.summary())
    out["models"] += [_entry(ref, extra={"key": "counts", "types": types}),
                      _entry(geo, extra={"key": "geometry"}),
                      _entry(sites, extra={"key": "sites", "site_r_max": float(args.site_r),
                                           "channels": int(sp.shape[1])})]
    results = {"counts": ref, "geometry": geo, "sites": sites}
    # the feature matrix behind every kernel above, kept for the forest arms further down, both
    # as it stands and as the kernel sees it.  The geometry kernel is a plain product of
    # standardised columns and divides out no row length, so its two forms are the same matrix.
    features = {"counts": counts_features, "geometry": X, "sites": sp}
    kernel_features = {"counts": kernel_rows(counts_features,
                                             scale=np.ones(counts_features.shape[1])),
                       "geometry": X}
    # the kernel matrices themselves, for the nested arms that put a width on them
    matrices = {"counts": counts_kernel}
    # the two hop kernels of item BT7
    best = None
    for rho in args.rho:
        envs = envs0 if rho == args.rho[0] else hs.environments(crystals, rows=t.atom, rho=rho,
                                                               r_max=args.r_max)
        local = local_features(envs)
        features[f"invariant {rho}"] = local
        kernel_features[f"invariant {rho}"] = kernel_rows(local, n_bins=LOCAL_SPEC.n_bins)
        matrices[f"invariant {rho}"] = local_kernel(local)
        inv = choose_lam(matrices[f"invariant {rho}"], t, folds=args.folds, seed=args.seed,
                         name=f"invariant local, rho {rho:g}")
        log("  " + inv.summary())
        fr = framed_environments(hs, crystals, rows=t.atom, rho=rho, r_max=args.r_max)
        t1 = time.time()
        model = HopModel.build(fr, fs, n_jobs=args.jobs)
        cyl = cross_validate_hop_model(model, t, folds=args.folds, seed=args.seed,
                                       name=f"rotation-averaged overlap, rho {rho:g}")
        atoms = float(np.mean([e.n_atoms for e in envs]))
        log(f"  {cyl.summary()}  (kernel {time.time() - t1:.0f} s, "
            f"{atoms:.0f} atoms per environment)")
        out["models"] += [_entry(inv, rho, {"key": f"invariant {rho}", "atoms": atoms}),
                          _entry(cyl, rho, {"key": f"overlap {rho}", "atoms": atoms})]
        results[f"invariant {rho}"] = inv
        results[f"overlap {rho}"] = cyl
        for name, r in ((f"invariant {rho}", inv), (f"overlap {rho}", cyl)):
            for ref_name in ("counts", "geometry", "sites"):
                d, lo, hi = paired_interval(r, results[ref_name], t, runs=args.bootstrap,
                                            seed=args.seed)
                out["pairs"].append({"a": name, "b": ref_name, "diff": d, "low": lo, "high": hi})
        d, lo, hi = paired_interval(cyl, inv, t, runs=args.bootstrap, seed=args.seed)
        out["pairs"].append({"a": f"overlap {rho}", "b": f"invariant {rho}", "diff": d,
                             "low": lo, "high": hi})
        log(f"    overlap against invariant at the same width {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        if best is None or cyl.q2 > best[0].q2:
            best = (cyl, rho, model)
    cyl, rho, model = best
    out["best_rho"] = rho
    # the fitting method on its own: the same descriptions, fitted the way the published models
    # are fitted.  The overlap kernel has no feature matrix of its own -- it is a kernel between
    # two fields, not a list of numbers per hop -- so it has no forest arm.
    out["forests"] = []
    for key in ("counts", "geometry", f"invariant {rho}"):
        forms = [("as the kernel", kernel_features[key])]
        if features[key] is not kernel_features[key]:
            forms.append(("raw", features[key]))     # the geometry kernel has only one form
        for rows, M in forms:
            for kind, short in FORESTS:
                fr_name = f"{results[key].name}, {rows}, {kind}"
                r = forest_result(M, t, args.folds, args.seed, fr_name, kind)
                log("  " + r.summary())
                fkey = f"{short} {key} {rows}"
                out["forests"].append(_entry(r, extra={
                    "key": fkey, "on": key, "method": kind, "rows": rows,
                    "columns": int(M.shape[1]), "lam": None}))
                results[fkey] = r
                d, lo, hi = paired_interval(r, results[key], t, runs=args.bootstrap,
                                            seed=args.seed)
                out["pairs"].append({"a": fkey, "b": key, "diff": d, "low": lo, "high": hi})
                log(f"    against the same description fitted with a kernel {d:+.3f} "
                    f"[{lo:+.3f}, {hi:+.3f}]")
    best_forest = max(out["forests"], key=lambda e: e["q2"])
    d, lo, hi = paired_interval(results[best_forest["key"]], cyl, t, runs=args.bootstrap,
                                seed=args.seed)
    out["pairs"].append({"a": best_forest["key"], "b": f"overlap {rho}", "diff": d, "low": lo,
                         "high": hi})
    log(f"  the best forest against the best model of item BT7 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out["best_forest"] = best_forest["key"]
    # can a kernel hold what the forests found?  The kernels of this program are linear, so the
    # first guess is that the barrier is not a linear function of the features.  A Gaussian
    # kernel on the same rows tests that guess, with its width and the ridge chosen inside the
    # training part, and the linear kernel is given the same treatment so the two can be
    # compared without the ridge search of the first table flattering either.
    out["nested"] = []
    for key in ("counts", f"invariant {rho}"):
        arms = {"linear": {"linear": matrices[key]},
                "Gaussian": {f"gamma {g:g}": gaussian_kernel(matrices[key], g) for g in GAMMAS}}
        for label, ks in arms.items():
            r, picked = nested_result(ks, t, args.folds, args.seed,
                                      f"{results[key].name}, {label}, nested")
            settings = ", ".join(sorted({c["setting"] for c in picked}))
            log(f"  {r.summary()}  (chose {settings})")
            nkey = f"nested {label} {key}"
            out["nested"].append(_entry(r, extra={"key": nkey, "on": key, "kernel": label,
                                                  "lam": None, "chosen": picked}))
            results[nkey] = r
            d, lo, hi = paired_interval(r, results[key], t, runs=args.bootstrap, seed=args.seed)
            out["pairs"].append({"a": nkey, "b": key, "diff": d, "low": lo, "high": hi})
        # what the width is worth, and whether it reaches the forest on the same rows
        for a_key, b_key in ((f"nested Gaussian {key}", f"nested linear {key}"),
                             (f"nested Gaussian {key}", f"rf {key} as the kernel")):
            if b_key not in results:
                continue
            d, lo, hi = paired_interval(results[a_key], results[b_key], t, runs=args.bootstrap,
                                        seed=args.seed)
            out["pairs"].append({"a": a_key, "b": b_key, "diff": d, "low": lo, "high": hi})
            log(f"    {a_key} against {b_key} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # the window the ion passes through: a three-distance description of the same environments,
    # which resolves the angle at the midpoint that the two-point description leaves open
    from cmfqsar.triangles import TriangleSpec, triangle_kernel_matrix
    envs_best = envs0 if rho == args.rho[0] else hs.environments(crystals, rows=t.atom, rho=rho,
                                                                r_max=args.r_max)
    tri, channels = {}, {}
    for reach, step in TRIANGLE_SPECS:
        spec = TriangleSpec(r_max=reach, step=step)
        t1 = time.time()
        K3 = triangle_kernel_matrix(envs_best, spec, n_fields=len(fs.names))
        for power in TRIANGLE_POWERS:
            key = f"reach {reach:g}, step {step:g}, power {power}"
            tri[key] = K3 if power == 1 else np.clip(K3, -1.0, 1.0) ** power
            channels[key] = int(spec.n_channels(len(fs.names)))
        log(f"  three-distance reach {reach:g} step {step:g}: "
            f"{spec.n_channels(len(fs.names))} channels, {time.time() - t1:.0f} s")
    # the same description with the row length left in, which is what item M15 asks about: the
    # forest on the raw two-point rows is ahead of every normalised kernel here, and the length
    # is what it has that they do not.  The overall scale is divided out by the mean diagonal,
    # which is one constant for the whole matrix and takes no row's length away.
    from cmfqsar.triangles import triangle_feature_matrix, triangle_kernel
    reach, step = TRIANGLE_SPECS[0]
    spec = TriangleSpec(r_max=reach, step=step)
    t1 = time.time()
    F3 = triangle_feature_matrix(envs_best, spec, len(fs.names))
    Kl = triangle_kernel(F3, spec=spec, n_fields=len(fs.names), normalize=False)
    Kl = Kl / max(float(np.mean(np.diag(Kl))), 1e-300)
    key = f"reach {reach:g}, step {step:g}, power 1, row length kept"
    tri[key] = Kl
    channels[key] = int(spec.n_channels(len(fs.names)))
    log(f"  three-distance reach {reach:g} step {step:g} with the row length kept: "
        f"{time.time() - t1:.0f} s")
    del F3
    # what each setting scores on its own, so that the table can show whether the gain depends
    # on the grid.  These are the optimistic kind -- the ridge is chosen on the folds they
    # report -- and they are here as evidence of flatness, not as the result.
    out["triangle_settings"] = []
    for setting, K3 in tri.items():
        s = choose_lam(K3, t, LAMS, folds=args.folds, seed=args.seed, name=setting)
        out["triangle_settings"].append({"setting": setting, "q2": s.q2, "rmse": s.error,
                                         "mae": s.mae, "lam": s.lam,
                                         "channels": channels[setting]})
        log(f"    {setting}: q2 {s.q2:+.3f}, error {s.error:.3f}, mae {s.mae:.3f} "
            f"(ridge on the folds it reports)")
    mix_key = (f"reach {TRIANGLE_SPECS[0][0]:g}, step {TRIANGLE_SPECS[0][1]:g}, power 1")
    mixes = {f"weight {w:g}": (1.0 - w) * matrices[f"invariant {rho}"] + w * tri[mix_key]
             for w in MIX_WEIGHTS}
    out["triangles"] = []
    for label, ks, extra in (("three-distance", tri, {"channels": channels}),
                             ("two-point and three-distance", mixes, {"mixes": mix_key})):
        r, picked = nested_result(ks, t, args.folds, args.seed, f"{label}, nested")
        settings = ", ".join(sorted({c["setting"] for c in picked}))
        log(f"  {r.summary()}  (chose {settings})")
        tkey = f"nested {label}"
        entry = {"key": tkey, "kernel": label, "lam": None, "chosen": picked}
        entry.update(extra)
        out["triangles"].append(_entry(r, extra=entry))
        results[tkey] = r
    key = f"invariant {rho}"
    for a_key, b_key in (("nested three-distance", f"nested linear {key}"),
                         ("nested three-distance", f"rf {key} as the kernel"),
                         ("nested three-distance", f"gb {key} as the kernel"),
                         ("nested three-distance", f"rf {key} raw"),
                         ("nested three-distance", f"overlap {rho}"),
                         ("nested two-point and three-distance", "nested three-distance")):
        if a_key not in results or b_key not in results:
            continue
        d, lo, hi = paired_interval(results[a_key], results[b_key], t, runs=args.bootstrap,
                                    seed=args.seed)
        out["pairs"].append({"a": a_key, "b": b_key, "diff": d, "low": lo, "high": hi})
        log(f"    {a_key} against {b_key} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # the free volume at the three sites of the hop.  The strongest feature of the published
    # models is the smallest volume of the Voronoi polyhedra around the start, the midpoint and
    # the end, so this asks two things: what that feature is worth here, and what it costs to
    # carry it as a field instead of as a cell.
    from cmfqsar.hops import point_environment
    t1 = time.time()
    vols = np.zeros((len(t), len(FREE_BETAS), 3))
    for n, r in enumerate(t.atom):
        h = hs.hops[int(r)]
        cr = crystals[h.structure]
        leave_out = [(h.a, (0, 0, 0)), (h.b, h.offset)]
        centre_radius = float(cr.radii[h.a])
        for j, c in enumerate((h.start, h.midpoint, h.start + h.vector)):
            e = point_environment(cr, c, rho, exclude=leave_out, r_max=args.r_max)
            vols[n, :, j] = free_volume(e.coords[1:], e.radii[1:], centre_radius)
    log(f"  free volume at the three sites of every hop: {time.time() - t1:.0f} s")
    free, free_long = {}, {}
    geo_cols = np.array([[hs.hops[r].length, hs.hops[r].bottleneck] for r in t.atom])
    geo_cols = (geo_cols - geo_cols.mean(axis=0)) / np.maximum(geo_cols.std(axis=0), 1e-12)
    for i, beta in enumerate(FREE_BETAS):
        A = free_volume_features(vols[:, i, :])
        free[free_volume_name(beta)] = A @ A.T
        B = np.hstack([A, geo_cols])
        free_long[free_volume_name(beta)] = B @ B.T
    out["free_volume"] = []
    # what each setting scores on its own: the cell against the field versions of it, which is
    # what says whether this program's own terms can carry the published feature
    out["free_settings"] = []
    for i, beta in enumerate(FREE_BETAS):
        name = free_volume_name(beta)
        s = choose_lam(free[name], t, LAMS, folds=args.folds, seed=args.seed, name=name)
        out["free_settings"].append({
            "setting": name, "q2": s.q2, "rmse": s.error, "mae": s.mae, "lam": s.lam,
            "midpoint_volume": float(np.mean(vols[:, i, 1])),
            "smallest_volume": float(np.mean(np.min(vols[:, i, :], axis=1)))})
        log(f"    {name}: q2 {s.q2:+.3f}, error {s.error:.3f}, mae {s.mae:.3f}, mean midpoint "
            f"volume {np.mean(vols[:, i, 1]):.2f} A^3 (ridge on the folds it reports)")
    tri_key = f"reach {TRIANGLE_SPECS[0][0]:g}, step {TRIANGLE_SPECS[0][1]:g}, power 1, " \
              f"row length kept"
    with_tri = {f"weight {w:g}": (1.0 - w) * tri[tri_key] + w * free[free_volume_name(None)]
                for w in FREE_MIX_WEIGHTS}
    for label, ks in (("free volume", free),
                      ("free volume with hop length and bottleneck", free_long),
                      ("three-distance and free volume", with_tri)):
        r, picked = nested_result(ks, t, args.folds, args.seed, f"{label}, nested")
        settings = ", ".join(sorted({c["setting"] for c in picked}))
        log(f"  {r.summary()}  (chose {settings})")
        fkey = f"nested {label}"
        out["free_volume"].append(_entry(r, extra={"key": fkey, "kernel": label, "lam": None,
                                                   "chosen": picked}))
        results[fkey] = r
    for a_key, b_key in (("nested free volume", "counts"),
                         ("nested free volume", "geometry"),
                         ("nested free volume with hop length and bottleneck", "geometry"),
                         ("nested free volume", f"nested linear {key}"),
                         ("nested three-distance and free volume", "nested three-distance"),
                         ("nested three-distance and free volume", f"rf {key} raw")):
        if a_key not in results or b_key not in results:
            continue
        d, lo, hi = paired_interval(results[a_key], results[b_key], t, runs=args.bootstrap,
                                    seed=args.seed)
        out["pairs"].append({"a": a_key, "b": b_key, "diff": d, "low": lo, "high": hi})
        log(f"    {a_key} against {b_key} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # the fitting method on the window the ion passes through.  The best model so far is a
    # forest given the raw two-point rows, which mixes a better description (the row length the
    # kernel divides out) with a stronger fitting method, so it is not clear which of them it
    # owes its lead to.  This arm separates them: a forest on the three-distance rows, over the
    # same hops and the same folds.  If the better description is worth to a forest what it was
    # worth to the kernel, the description side is still open; if it is worth nothing, the
    # remainder belongs to the fitting method.
    from cmfqsar.triangles import TriangleSpec, triangle_feature_matrix, triangle_scale
    spec3 = TriangleSpec(r_max=TRIANGLE_SPECS[0][0], step=TRIANGLE_SPECS[0][1])
    t1 = time.time()
    F3 = triangle_feature_matrix(envs_best, spec3, len(fs.names))
    F3 *= triangle_scale(F3, spec3, len(fs.names))      # the channel scale the kernel applies
    row_form = TRIANGLE_FOREST_ROWS
    M = F3 if row_form == "raw" else kernel_rows(F3, scale=np.ones(F3.shape[1]))
    log(f"  the three-distance rows for the forest arms: {M.shape[1]} columns, rows "
        f"{row_form}, {time.time() - t1:.0f} s")
    out["triangle_forests"] = []
    for kind, short in FORESTS:
        t1 = time.time()
        r = forest_result(M, t, args.folds, args.seed,
                          f"three-distance, {row_form}, {kind}", kind)
        log(f"  {r.summary()}  ({time.time() - t1:.0f} s)")
        fkey = f"{short} three-distance {row_form}"
        out["triangle_forests"].append(_entry(r, extra={
            "key": fkey, "on": "nested three-distance", "method": kind, "rows": row_form,
            "columns": int(M.shape[1]), "lam": None}))
        results[fkey] = r
    del M, F3
    best_tri_forest = max(out["triangle_forests"], key=lambda e: e["q2"])
    out["best_triangle_forest"] = best_tri_forest["key"]
    for a_key, b_key in [(f["key"], "nested three-distance") for f in out["triangle_forests"]] \
            + [(best_tri_forest["key"], f"rf {key} raw"),
               (best_tri_forest["key"], "nested three-distance and free volume")]:
        if a_key not in results or b_key not in results:
            continue
        d, lo, hi = paired_interval(results[a_key], results[b_key], t, runs=args.bootstrap,
                                    seed=args.seed)
        out["pairs"].append({"a": a_key, "b": b_key, "diff": d, "low": lo, "high": hi})
        log(f"    {a_key} against {b_key} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # the learning curve of the best model, on the kernel already in hand: how much of the
    # distance to the published models is the size of the training set
    curve = []
    which = np.unique(t.molecule)
    rng = np.random.RandomState(args.seed)
    for n_struct in CURVE:
        if n_struct >= len(which):
            break
        q2s, errs = [], []
        for rep in range(3):
            drawn = set(rng.choice(which, n_struct, replace=False).tolist())
            rows = np.array([i for i in range(len(t)) if int(t.molecule[i]) in drawn])
            sub = t.subset(rows)
            r = choose_lam(model.K[np.ix_(rows, rows)], sub, LAMS,
                           folds=min(args.folds, n_struct), seed=args.seed + rep, name="subset")
            q2s.append(r.q2)
            errs.append(r.error)
        curve.append({"structures": n_struct, "q2": float(np.mean(q2s)),
                      "spread": float(np.std(q2s)), "rmse": float(np.mean(errs))})
        log(f"    learning curve: {n_struct} structures, q2 {np.mean(q2s):+.3f} "
            f"+- {np.std(q2s):.3f}, error {np.mean(errs):.3f} eV")
    curve.append({"structures": int(len(which)), "q2": cyl.q2, "spread": 0.0,
                  "rmse": cyl.error})
    out["learning_curve"] = curve
    # the same curve for the best forest arm.  The draws are started from the same seed as the
    # curve above, so the two curves are measured on the same subsets of structures and their
    # rates can be set beside each other.
    fkey = best_forest["key"]
    fmat = (kernel_features if best_forest["rows"] == "as the kernel" else
            features)[best_forest["on"]]
    fcurve, rng = [], np.random.RandomState(args.seed)
    for n_struct in CURVE:
        if n_struct >= len(which):
            break
        q2s, errs = [], []
        for rep in range(3):
            drawn = set(rng.choice(which, n_struct, replace=False).tolist())
            rows = np.array([i for i in range(len(t)) if int(t.molecule[i]) in drawn])
            r = forest_result(fmat[rows], t.subset(rows), min(args.folds, n_struct),
                              args.seed + rep, "subset", best_forest["method"])
            q2s.append(r.q2)
            errs.append(r.error)
        fcurve.append({"structures": n_struct, "q2": float(np.mean(q2s)),
                       "spread": float(np.std(q2s)), "rmse": float(np.mean(errs))})
        log(f"    forest learning curve: {n_struct} structures, q2 {np.mean(q2s):+.3f} "
            f"+- {np.std(q2s):.3f}, error {np.mean(errs):.3f} eV")
    fcurve.append({"structures": int(len(which)), "q2": results[fkey].q2, "spread": 0.0,
                   "rmse": results[fkey].error})
    out["forest_curve"] = fcurve
    out["seconds"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------- the table


def projection(curve: List[dict]) -> Optional[tuple]:
    """(gain per doubling, doublings left, where that rate would arrive) for a learning curve.

    The rate is read off the last step of the curve, and the doublings are counted from the end
    of the curve up to the number of structures the published models are fitted on.  q2 cannot
    pass 1, so a rate that would carry the model past it says only that the rate cannot hold that
    far; the caller reports that case rather than printing an impossible number.
    """
    if len(curve) < 2:
        return None
    a, b = curve[-2], curve[-1]
    if b["structures"] <= a["structures"] or b["structures"] >= PUBLISHED_TRAIN[1]:
        return None
    per = (b["q2"] - a["q2"]) / float(np.log2(b["structures"] / a["structures"]))
    steps = float(np.log2(PUBLISHED_TRAIN[1] / b["structures"]))
    return per, steps, b["q2"] + per * steps


def write_table(results_dir: str = RESULTS, tag: str = "") -> str:
    """The table of one run, from the result files that carry its tag; returns the text.

    ``run<tag>.json`` holds the command that produced those files; the table prints it, so that a
    reader can repeat the run and so that a stored result with no command is visible as such.
    """
    pub = json.load(open(os.path.join(results_dir, f"public{tag}.json")))
    prov = run_provenance(results_dir, tag)
    col = pub["collection"]
    pairs = {(p["a"], p["b"]): p for p in pub["pairs"]}
    lines = []
    w = lines.append
    w("# The hop models on the published nebBVSE122k benchmark (the external check of item BT7)")
    w("")
    if prov:
        w("`" + " ".join(["python benchmarks/run_litraj.py"] + list(prov["argv"])) + "`, reading "
          f"`{prov['prepared']}` (written by `python benchmarks/prepare_litraj.py --fetch`), "
          f"seed {prov['seed']}, {prov['bootstrap']} bootstrap draws.")
    else:
        w("The command that produced these result files was not recorded.")
    w("")
    w(f"The collection is {col['dataset']} of {col['reference']}, {col['licence']} licence, "
      f"{col['source']}: {col['n_index_hops']} bond-valence barriers of single Li hops in "
      f"{col['n_index_structures']} structures of the Materials Project over "
      f"{col['n_index_chemsys']} chemical systems. Its label is the barrier of one hop on a rigid "
      f"framework, as in item BT7, but relaxed with the nudged elastic band method instead of "
      f"taken as the minimax path on a grid.")
    w("")
    w(f"A kernel model holds one number per pair of samples, so this run reads a sample: "
      f"{col['drawn']} structures drawn at random with at most {col['max_per_structure']} of "
      f"their hops each, which leaves {pub['n_samples']} hops in {pub['n_structures']} "
      f"structures. Hop lengths {pub['lengths'][0]:.2f} to {pub['lengths'][1]:.2f} A. "
      f"{pub['folds']} folds holding out whole structures; environments within "
      f"{pub['r_max']:g} A of the midpoint. Fields: {', '.join(pub['fields'])}.")
    w("")
    w(f"**The barriers of this sample spread more widely than those of the public set of item "
      f"BT7**: standard deviation {pub['y_sd']:.3f} eV against 0.807 eV, mean "
      f"{pub['y_mean']:.3f} eV, because the collection covers every chemistry that holds lithium "
      f"rather than ternary lithium oxides alone. q2 is measured against the spread of the "
      f"sample, so the same error in eV gives a higher q2 here: **the error in eV is the column "
      f"to compare with item BT7**. Against the published models of the next section both columns "
      f"hold, because this sample and the whole collection spread almost alike "
      f"({pub['y_sd']:.2f} eV against {col['index_y_sd']:.2f} eV).")
    w("")
    w("| descriptor | envelope width rho | q2 | error (eV) | mean absolute error (eV) | ridge "
      "| against the counts | against length and bottleneck | against the two end sites |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in pub["models"]:
        key = m.get("key", "")
        cells = []
        for ref in ("counts", "geometry", "sites"):
            p = pairs.get((key, ref))
            cells.append("--" if p is None else
                         f"{p['diff']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]")
        rho = f"{m['rho']:g}" if m.get("rho") is not None else "--"
        w(f"| {m['name']} | {rho} | {m['q2']:+.3f} | {m['rmse']:.3f} | {m['mae']:.3f} | "
          f"{m['lam']:g} | " + " | ".join(cells) + " |")
    w("")
    best = f"overlap {pub['best_rho']}"
    p = pairs.get((best, f"invariant {pub['best_rho']}"))
    if p:
        w(f"Overlap in the hop frame against the invariant kernel at the same width: "
          f"{p['diff']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}].")
        w("")
    curve = pub.get("learning_curve") or []
    if curve:
        w("Learning curve of the rotation-averaged overlap model, three random draws of the "
          "training structures at each size. Each draw has a label spread of its own, so the "
          "error in eV is not comparable from row to row and q2 is the column to read:")
        w("")
        w("| training structures | q2 | error (eV) |")
        w("| ---: | ---: | ---: |")
        for c in curve:
            w(f"| {c['structures']} | {c['q2']:+.3f} | {c['rmse']:.3f} |")
        w("")
    w("## What the published models reach on the whole collection")
    w("")
    w(f"Table 2 of the paper, on its test part ({PUBLISHED_TEST[0]} hops of "
      f"{PUBLISHED_TEST[1]} structures, none of them in the training part), fitted on "
      f"{PUBLISHED_TRAIN[0]} hops of {PUBLISHED_TRAIN[1]} structures:")
    w("")
    w("| model | mean absolute error (eV) | error (eV) | R2 |")
    w("| --- | ---: | ---: | ---: |")
    for name, mae, rmse_, r2 in PUBLISHED:
        w(f"| {name} | {mae:.2f} | {rmse_:.2f} | {r2:.2f} |")
    w("")
    w("Those models are fitted on about forty times as many hops as the sample above, which is "
      "the one difference the protocol does not remove, so the learning curve is what says how "
      "much of the distance the size of the training set explains.")
    w("")
    proj = projection(curve)
    if proj:
        per, steps, arrives = proj
        a, b = curve[-2], curve[-1]
        w(f"Between {a['structures']} and {b['structures']} training structures the overlap model "
          f"gains {b['q2'] - a['q2']:+.3f} in q2, which is {per:+.3f} for every doubling of the "
          f"training set. The training part of the collection holds {PUBLISHED_TRAIN[1]} "
          f"structures, {steps:.1f} doublings beyond this sample.")
        if arrives >= 0.99:
            w(f"That rate would carry the model past q2 1.00 long before it got there, so the "
              f"rate cannot hold over that distance and this curve says nothing about where the "
              f"model would arrive. It is too short to read; the stored run draws it over "
              f"{CURVE[-1]} structures.")
        else:
            w(f"That rate carries the model to about q2 {arrives:+.2f}, short of the "
              f"{PUBLISHED[0][3]:+.2f} the graph network reaches. A learning curve is not a law "
              f"and may bend either way, but at the rate measured here the size of the training "
              f"set does not account for the distance.")
        w("")
    forests = pub.get("forests") or []
    if forests:
        w("## The same descriptions, fitted the way the published models are fitted")
        w("")
        w("The two published models that come closest to the graph network are a random forest "
          "and a gradient-boosted forest on hand-made features of the hop, not kernel models. "
          "The table at the top therefore mixes two questions: whether this program's "
          "**description** of a hop is behind, or whether fitting a description with a **kernel** "
          "is. The rows below fit a forest on this program's own descriptions, over the same "
          "hops and the same folds, which separates the two. The random forest is the same kind "
          "of model the paper reports; the boosted forest is the histogram boosting of "
          "scikit-learn, standing in for the paper's XGBoost.")
        w("")
        w("The comparison favours the kernel. Every kernel row above is given the ridge that "
          "maximises the same out-of-fold q2 the row reports, so its figure is optimistic by "
          "whatever that search is worth; the forests are fitted at fixed settings and search "
          "for nothing.")
        w("")
        w("Both kernels here divide every feature row by its own length, so a kernel entry is "
          "the cosine of the angle between two rows and the length of a row is divided out. A "
          "forest handed the raw rows would be handed more than the kernel ever had, and the "
          "comparison would no longer be about the fitting method. The **rows** column says "
          "which was given: *as the kernel* is the arm that compares the two fitting methods on "
          "one description, and *raw* asks separately what the row length is worth. The geometry "
          "kernel divides out no row length, so it has only one form.")
        w("")
        w("| description | rows | columns | fitted with | q2 | error (eV) "
          "| mean absolute error (eV) | against the same description fitted with a kernel |")
        w("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |")
        for m in forests:
            p = pairs.get((m["key"], m["on"]))
            cell = ("--" if p is None else
                    f"{p['diff']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]")
            name = m["name"].split(",")[0]
            w(f"| {name} | {m.get('rows', '--')} | {m.get('columns', 0)} | {m['method']} | "
              f"{m['q2']:+.3f} | {m['rmse']:.3f} | {m['mae']:.3f} | {cell} |")
        w("")
        p = pairs.get((pub.get("best_forest"), f"overlap {pub['best_rho']}"))
        if p:
            w(f"The best of these forests against the best model of item BT7, the overlap in the "
              f"hop frame: {p['diff']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}].")
            w("")
        # what the rows say, from the numbers themselves: the local description carries the
        # finding, because it is the one the kernels of this program are built on
        key = f"invariant {pub['best_rho']}"
        kernel = next((m for m in pub["models"] if m.get("key") == key), None)
        arms = [f for f in forests if f["on"] == key]
        same = [f for f in arms if f.get("rows") == "as the kernel"]
        raw = [f for f in arms if f.get("rows") == "raw"]
        if kernel and same and raw:
            a = max(same, key=lambda f: f["q2"])
            b = max(raw, key=lambda f: f["q2"])
            pa = pairs.get((a["key"], key))
            w("**What these rows say.** The local description is the one the field kernels of "
              "this program are built on, and it carries two findings, which the rows column "
              "keeps apart.")
            w("")
            w(f"*The fitting method is worth a large part of the distance.* On exactly the rows "
              f"the kernel is built from -- the same description, nothing added -- the "
              f"{a['method']} reaches q2 {a['q2']:+.3f} where the kernel reaches "
              f"{kernel['q2']:+.3f}, and the mean absolute error falls from {kernel['mae']:.3f} "
              f"to {a['mae']:.3f} eV"
              + ("" if pa is None else
                 f" ({pa['diff']:+.3f} [{pa['low']:+.3f}, {pa['high']:+.3f}])")
              + ". So this program's description of a hop was not what was behind the published "
                "models on this set; fitting it with a kernel was.")
            w("")
            w(f"*The cosine normalisation throws away something that carries the barrier.* "
              f"Handing the forest the raw rows as well, so that it can see how large an "
              f"environment's features are and not only their direction, adds a further "
              f"{b['q2'] - a['q2']:+.3f} in q2 and takes the mean absolute error down to "
              f"{b['mae']:.3f} eV. The kernel cannot use that length by construction: it divides "
              f"every row by it. This is a property of the kernel of this program, not of the "
              f"hop models, so it is worth testing wherever that kernel is used.")
            w("")
            w(f"Set beside the published models, this program's description of a hop fitted with "
              f"a forest reaches a mean absolute error of {b['mae']:.3f} eV on "
              f"{pub['n_samples']} hops, against {PUBLISHED[0][1]:.2f} eV for the graph network "
              f"and {PUBLISHED[1][1]:.2f} eV for the published forest, both fitted on "
              f"{PUBLISHED_TRAIN[0]} hops.")
            w("")
    nested = pub.get("nested") or []
    if nested:
        w("### Can a kernel hold what the forests found?")
        w("")
        w("The kernels of this program are linear. "
          "`cmfqsar.invariant.autocorrelation_kernel` rescales the channels, divides every row "
          "by its own length and takes the inner product, so ridge regression on it is a linear "
          "model in the feature space, while a forest is not. The first guess is therefore that "
          "the barrier is not a linear function of these features. A Gaussian kernel on the same "
          "rows tests that guess and stays a kernel method: the rows are unit vectors, so the "
          "squared distance between two of them is 2 - 2K and the whole family is a function of "
          "the matrix already computed.")
        w("")
        w("These rows are fitted more honestly than the rows above. The width and the ridge are "
          "chosen inside each training part, on an inner split that holds out whole structures, "
          "and the test fold is touched once with the choice already made. The linear kernel is "
          "given the same treatment, so the pair can be compared without the ridge search of the "
          "first table favouring either. That is why a nested linear row can score below the "
          "same kernel in the first table: the first table chose its ridge on the folds it "
          "reports.")
        w("")
        w("| description | kernel | q2 | error (eV) | mean absolute error (eV) | widths chosen "
          "| against the same kernel in the first table |")
        w("| --- | --- | ---: | ---: | ---: | --- | ---: |")
        for m in nested:
            p = pairs.get((m["key"], m["on"]))
            cell = ("--" if p is None else
                    f"{p['diff']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]")
            picked = ", ".join(sorted({c["setting"] for c in m.get("chosen", [])}))
            w(f"| {m['name'].split(',')[0]} | {m['kernel']} | {m['q2']:+.3f} | {m['rmse']:.3f} "
              f"| {m['mae']:.3f} | {picked or '--'} | {cell} |")
        w("")
        key = f"invariant {pub['best_rho']}"
        gl = pairs.get((f"nested Gaussian {key}", f"nested linear {key}"))
        gf = pairs.get((f"nested Gaussian {key}", f"rf {key} as the kernel"))
        cl = pairs.get(("nested Gaussian counts", "nested linear counts"))
        lin = next((m for m in nested if m["key"] == f"nested linear {key}"), None)
        forest = next((f for f in forests if f["key"] == f"rf {key} as the kernel"), None)
        if gl and gf and lin and forest:
            gap = forest["q2"] - lin["q2"]
            share = gl["diff"] / gap if gap > 0 else float("nan")
            w(f"**The guess is part of the answer, and the smaller part.** On the local "
              f"description the width is worth {gl['diff']:+.3f} "
              f"[{gl['low']:+.3f}, {gl['high']:+.3f}] over the linear kernel given the same "
              f"treatment, which is clear of zero but small: the whole distance from that "
              f"kernel to the random forest on the very same rows is {gap:+.3f} in q2, so a "
              f"curved response to these features accounts for about {share:.0%} of it. The "
              f"Gaussian kernel is still {gf['diff']:+.3f} "
              f"[{gf['low']:+.3f}, {gf['high']:+.3f}] behind that forest.")
            w("")
            if cl:
                w(f"On the counts the width is worth nothing at all "
                  f"({cl['diff']:+.3f} [{cl['low']:+.3f}, {cl['high']:+.3f}]), so this is not a "
                  f"general gain from curving a kernel; it is specific to the description that "
                  f"has many columns.")
                w("")
            w("**What the rest is more likely to be: the metric.** A Gaussian kernel measures "
              "one distance over all the columns at once, so the few columns that carry the "
              "barrier are diluted by the many that do not, and no single width repairs that. A "
              "tree picks columns one at a time and ignores the rest. On this reading the "
              "question is which columns count and how they are weighted, which is a different "
              "repair from putting a width on the kernel, and it is a reading and not yet a "
              "measurement. Item M15 of the plan is where it is recorded.")
            w("")
        row = next((m for m in nested if m["key"] == f"nested linear {key}"), None)
        p = pairs.get((f"nested linear {key}", key))
        if row and p:
            w(f"One number in passing. The nested linear kernel scores "
              f"{p['diff']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}] against the same kernel in "
              f"the first table, which is what choosing the ridge on the folds it reports is "
              f"worth. Every kernel row of the first table is overstated by about that much and "
              f"the forest rows by nothing, so the forest's advantage there is understated.")
            w("")
    tri = pub.get("triangles") or []
    if tri:
        key = f"invariant {pub['best_rho']}"
        w("### The window the ion passes through")
        w("")
        w("A barrier is set by the narrowest place along the path: a window of three or four "
          "framework atoms that the ion has to pass. The two-point description of an environment "
          "carries the distance between two neighbours but not the two sides that meet at the "
          "centre, so it does not resolve the angle there, and two different windows can leave "
          "the same distance histogram. `cmfqsar.triangles` (plan item C5) resolves it for an "
          "environment whose first atom is the centre, and the first atom of a hop environment "
          "is a marker at the midpoint, so the triangles it builds are exactly the midpoint with "
          "two framework atoms. This stays a linear kernel on the fields, so the co-field "
          "survives, which a forest and a Gaussian width do not offer.")
        w("")
        w("The reach, the grid spacing and the exponent are chosen inside each training part, as "
          "in the section above. The mixture row searches a convex sum of the two-point and "
          "three-distance kernels whose grid includes both ends, so it can fall back on either.")
        w("")
        w("| description | q2 | error (eV) | mean absolute error (eV) | chosen inside the folds |")
        w("| --- | ---: | ---: | ---: | --- |")
        for m in tri:
            picked = ", ".join(sorted({c["setting"] for c in m.get("chosen", [])}))
            w(f"| {m['kernel']} | {m['q2']:+.3f} | {m['rmse']:.3f} | {m['mae']:.3f} "
              f"| {picked or '--'} |")
        w("")
        rows = (("the two-point description, nested, on the same environments",
                 f"nested linear {key}"),
                ("the random forest on the two-point rows", f"rf {key} as the kernel"),
                ("the boosted forest on the two-point rows", f"gb {key} as the kernel"),
                ("the random forest on the **raw** two-point rows, which has the row length "
                 "every normalised kernel here divides out", f"rf {key} raw"),
                ("the overlap kernel in the hop frame, the best model of item BT7",
                 f"overlap {pub['best_rho']}"))
        lines_any = [(what, pairs.get(("nested three-distance", b))) for what, b in rows]
        if any(p for _, p in lines_any):
            w("The three-distance kernel against:")
            w("")
            for what, p in lines_any:
                if p:
                    w(f"* {what}: **{p['diff']:+.3f} "
                      f"[{p['low']:+.3f}, {p['high']:+.3f}]**")
            w("")
        p = pairs.get(("nested two-point and three-distance", "nested three-distance"))
        if p:
            w(f"Mixing the two-point kernel back in is worth {p['diff']:+.3f} "
              f"[{p['low']:+.3f}, {p['high']:+.3f}] over the three-distance kernel alone.")
            w("")
        gl = pairs.get(("nested three-distance", f"nested linear {key}"))
        gw = pairs.get((f"nested Gaussian {key}", f"nested linear {key}"))
        if gl and gw:
            which = ("more" if gl["diff"] > gw["diff"] else "less")
            w(f"**What this says.** Against the same two-point description fitted the same way, "
              f"resolving the angle at the midpoint is worth {gl['diff']:+.3f} and putting a "
              f"width on the kernel is worth {gw['diff']:+.3f}, so the angle is worth {which}. "
              f"It is also the repair that keeps the model a linear kernel on fields, which a "
              f"forest and a Gaussian width do not.")
            w("")
        settings = pub.get("triangle_settings") or []
        if len(settings) > 1:
            qs = [s["q2"] for s in settings]
            spread = max(qs) - min(qs)
            w(f"**Does the gain depend on the grid?** Each setting on its own, with the ridge "
              f"chosen on the folds it reports, so these are the optimistic kind and are here "
              f"only to show how flat the choice is:")
            w("")
            w("| setting | channels | q2 | mean absolute error (eV) |")
            w("| --- | ---: | ---: | ---: |")
            for s in settings:
                w(f"| {s['setting']} | {s['channels']} | {s['q2']:+.3f} | {s['mae']:.3f} |")
            w("")
            w(f"The whole span of those settings is {spread:.3f} in q2, so the gain does not "
              f"rest on the grid being tuned.")
            w("")
        kept = [s for s in settings if "row length kept" in s["setting"]]
        plain = [s for s in settings if "row length kept" not in s["setting"]]
        if kept and plain:
            k, b = kept[0], max(plain, key=lambda s: s["q2"])
            chose = any("row length kept" in c["setting"]
                        for m in tri for c in m.get("chosen", []))
            helps = k["q2"] > b["q2"]
            reading = ("so the row length carries something here as well, and the two findings "
                       "of this section and the last add together rather than replacing each "
                       "other"
                       if helps else
                       "so on this description the row length adds nothing, and whatever the "
                       "forest on the raw two-point rows is using, a three-distance kernel does "
                       "not reach it by keeping the length")
            w(f"**And the row length, which item M15 asks about.** The same description with the "
              f"row length left in rather than divided out scores q2 {k['q2']:+.3f} against "
              f"{b['q2']:+.3f} for the best normalised setting, and the nested arm "
              + ("chose it in at least one fold" if chose else "never chose it") + " -- "
              + reading + ".")
            w("")
        w("**One caution about this section.** The range of the grid was placed after a first "
          "look at this same sample. Nothing was chosen on a test fold, but the grid itself is "
          "not independent of the data, so a fresh set would settle it; how flat the settings "
          "are is what makes this a small worry rather than a large one.")
        w("")
    fv = pub.get("free_volume") or []
    if fv:
        key = f"invariant {pub['best_rho']}"
        w("### The free volume at the three sites, as a cell and as a field")
        w("")
        w("The strongest single feature of the published models on this benchmark is the "
          "**smallest volume of the Voronoi polyhedra** around the three sites of the hop -- its "
          "start, its midpoint and its end -- with the largest of the three next in their "
          "ranking. That quantity is a minimum over atoms, which looks unlike a field, but it is "
          "the narrow-width limit of a ratio of fields. Writing d(i) for the squared distance "
          "from a point to atom i less that atom's squared radius, the radical Voronoi cell of "
          "the centre is where d(centre) is smallest, and")
        w("")
        w("        w(r) = exp(-beta d(centre)) / sum over i of exp(-beta d(i))")
        w("")
        w("tends to the indicator function of that cell as beta grows. The denominator is a "
          "steric field of this program with unit weights and a Gaussian of width "
          "1/sqrt(2 beta), so the volume is the integral of a ratio of fields and the cell is "
          "its sharp limit. Both are measured below: the cell is the published feature and a "
          "width is this program's version of it.")
        w("")
        fs_rows = pub.get("free_settings") or []
        if fs_rows:
            w("Each setting on its own, on the four columns the three sites give (their "
              "smallest, largest and mean volume, and the midpoint alone). The ridge is chosen "
              "on the folds these rows report, so they are the optimistic kind:")
            w("")
            w("| setting | q2 | mean absolute error (eV) | mean volume at the midpoint (A^3) "
              "|")
            w("| --- | ---: | ---: | ---: |")
            for s in fs_rows:
                w(f"| {s['setting']} | {s['q2']:+.3f} | {s['mae']:.3f} "
                  f"| {s['midpoint_volume']:.2f} |")
            w("")
            cell = next((s for s in fs_rows if s["setting"] == "Voronoi cell"), None)
            soft = [s for s in fs_rows if s["setting"] != "Voronoi cell"]
            if cell and soft:
                best = max(soft, key=lambda s: s["q2"])
                gap = cell["q2"] - best["q2"]
                answer = ("the field version loses nothing worth having"
                          if abs(gap) < 0.02 else
                          ("the cell is ahead of every field version of it, so the sharp "
                           "minimum carries something the smooth ratio does not"
                           if gap > 0 else
                           "the field version is ahead of the cell itself"))
                w(f"**Can a field carry it?** The cell scores {cell['q2']:+.3f} and the best "
                  f"field version {best['q2']:+.3f} ({best['setting']}), a difference of "
                  f"{gap:+.3f}: {answer}.")
                w("")
        w("With the width and the ridge chosen inside each training part:")
        w("")
        w("| description | q2 | error (eV) | mean absolute error (eV) | chosen inside the folds "
          "|")
        w("| --- | ---: | ---: | ---: | --- |")
        for m in fv:
            picked = ", ".join(sorted({c["setting"] for c in m.get("chosen", [])}))
            w(f"| {m['kernel']} | {m['q2']:+.3f} | {m['rmse']:.3f} | {m['mae']:.3f} "
              f"| {picked or '--'} |")
        w("")
        for what, a_key, b_key in (
                ("Four numbers of free volume against the counts reference of rule 1",
                 "nested free volume", "counts"),
                ("against the hop length with the bottleneck",
                 "nested free volume", "geometry"),
                ("against the whole two-point description fitted the same way",
                 "nested free volume", f"nested linear {key}"),
                ("Adding the cell to the three-distance kernel",
                 "nested three-distance and free volume", "nested three-distance"),
                ("That mixture against the random forest on the raw two-point rows",
                 "nested three-distance and free volume", f"rf {key} raw")):
            p = pairs.get((a_key, b_key))
            if p:
                w(f"* {what}: **{p['diff']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]**")
        w("")
        add = pairs.get(("nested three-distance and free volume", "nested three-distance"))
        ref = pairs.get(("nested free volume", "counts"))
        if add and ref:
            covered = add["low"] <= 0.0 <= add["high"]
            w(f"**What this says.** Four numbers of free volume are level with the counts "
              f"reference of the first table, which has hundreds of channels "
              f"({ref['diff']:+.3f} [{ref['low']:+.3f}, {ref['high']:+.3f}]), so the published "
              f"feature is a strong descriptor in few terms. But adding it to the "
              f"three-distance kernel is worth {add['diff']:+.3f} "
              f"[{add['low']:+.3f}, {add['high']:+.3f}]"
              + (", which does not clear zero: the three-distance block already carries what "
                 "the free volume says, which is what a description of the triangles around the "
                 "midpoint should carry. So the published feature is not what this program was "
                 "missing."
                 if covered else
                 ", which is clear of zero: the free volume carries something the "
                 "three-distance block does not."))
            w("")
        w("**One caution.** The volumes are integrated on a cubic grid of "
          f"{FREE_STEP:g} angstrom inside a ball of {FREE_RADIUS:g} angstrom, which reproduces "
          "an exactly known cell to under one per cent but leaves the sharp cell with a little "
          "grid noise that the smooth versions mostly avoid. That favours the field versions "
          "slightly, so a cell that wins anyway has won against the arrangement.")
        w("")
    tf = pub.get("triangle_forests") or []
    if tf:
        key = f"invariant {pub['best_rho']}"
        w("### A forest on the window, which closes the description side")
        w("")
        w("Two things closed most of the distance to the published models in the sections "
          "above. One is the **description**: the angle at the midpoint, and the length of a "
          "row. The other is the **fitting method**: a forest instead of a linear kernel on the "
          "same rows. The best model so far mixes the two, being a forest given the raw "
          "two-point rows, so it is not clear which of them it owes its lead to. The rows below "
          "separate them by fitting a forest on the three-distance rows, over the same hops and "
          "the same folds. If the better description is worth to a forest what it was worth to "
          "the kernel, the description side is still open; if it is worth nothing, what is left "
          "is the fitting method alone.")
        w("")
        w(f"The rows are the ones the winning kernel of the last section is built from: the "
          f"channels scaled as that kernel scales them, with the row length left in, which is "
          f"the setting the nested arm chose in every fold. The normalised form is not measured "
          f"here -- it would pair with a setting that was never chosen, and this feature matrix "
          f"is {tf[0]['columns']} columns wide, which costs hours per arm.")
        w("")
        w("| rows | fitting method | columns | q2 | error (eV) | mean absolute error (eV) |")
        w("| --- | --- | ---: | ---: | ---: | ---: |")
        for m in tf:
            w(f"| {m['rows']} | {m['method']} | {m['columns']} | {m['q2']:+.3f} "
              f"| {m['rmse']:.3f} | {m['mae']:.3f} |")
        w("")
        for m in tf:
            p = pairs.get((m["key"], "nested three-distance"))
            if p:
                w(f"* {m['method'].capitalize()} on the rows {m['rows']}, against the kernel on "
                  f"the same description: **{p['diff']:+.3f} [{p['low']:+.3f}, "
                  f"{p['high']:+.3f}]**")
        best = pub.get("best_triangle_forest")
        for what, b_key in (("against the random forest on the raw two-point rows",
                             f"rf {key} raw"),
                            ("against the best kernel model of this item",
                             "nested three-distance and free volume")):
            p = pairs.get((best, b_key))
            if p:
                w(f"* The best of them {what}: **{p['diff']:+.3f} [{p['low']:+.3f}, "
                  f"{p['high']:+.3f}]**")
        w("")
        gains = [pairs.get((m["key"], "nested three-distance")) for m in tf]
        gains = [g for g in gains if g]
        if gains:
            top = max(gains, key=lambda g: g["diff"])
            clear = top["low"] > 0.0
            w("**What this says.** The best forest on the three-distance rows is "
              f"{top['diff']:+.3f} [{top['low']:+.3f}, {top['high']:+.3f}] against the kernel "
              "on the same rows"
              + (", which is clear of zero: a forest still finds something in this description "
                 "that a linear kernel on it does not, so the fitting method is part of what "
                 "remains and resolving the angle at the midpoint has not closed it."
                 if clear else
                 ", which does not clear zero. On the two-point description the same comparison "
                 "was a clear gain for the forest, so resolving the angle at the midpoint has "
                 "closed the distance between the two fitting methods: there is nothing left "
                 "here for a forest to find that the kernel on the same rows cannot reach."))
            w("")
        catch = pairs.get((pub.get("best_triangle_forest"), f"rf {key} raw"))
        if catch:
            behind = catch["high"] < 0.0
            w("**And the forest that was unaccounted for.** The best forest on these rows is "
              f"{catch['diff']:+.3f} [{catch['low']:+.3f}, {catch['high']:+.3f}] against the "
              "random forest on the raw two-point rows"
              + (", so that forest is still ahead of every model here and what it is using is "
                 "still not accounted for."
                 if behind else
                 ", so the forest that was ahead of every kernel in the sections above is no "
                 "longer ahead of a forest given this description."))
            w("")
    fcurve = pub.get("forest_curve") or []
    if fcurve and curve:
        shown = next((f for f in forests if f["key"] == pub.get("best_forest")), None)
        which = "the best forest arm"
        if shown:
            which = (f"{shown['name'].split(',')[0]}, rows {shown.get('rows', '--')}, "
                     f"{shown['method']}")
        w("Both model classes learn at a rate of their own, and the rate is what the distance to "
          "the published training set has to be crossed at. The two curves below are measured on "
          "the same drawn subsets of structures:")
        w("")
        w(f"| training structures | q2, overlap in the hop frame | q2, {which} |")
        w("| ---: | ---: | ---: |")
        by_n = {c["structures"]: c for c in fcurve}
        for c in curve:
            f = by_n.get(c["structures"])
            w(f"| {c['structures']} | {c['q2']:+.3f} | "
              + ("--" if f is None else f"{f['q2']:+.3f}") + " |")
        w("")
        fproj = projection(fcurve)
        if fproj:
            per, steps, arrives = fproj
            sentence = (f"The forest gains {per:+.3f} in q2 for every doubling of the training "
                        f"set over the last step of its curve")
            if proj:
                sentence += f", against {proj[0]:+.3f} for the overlap model"
            if arrives < 0.99:
                sentence += (f". At that rate it reaches about q2 {arrives:+.2f} at the "
                             f"{PUBLISHED_TRAIN[1]} structures the published models are fitted "
                             f"on")
            w(sentence + ".")
            w("")
    text = "\n".join(lines)
    with open(os.path.join(results_dir, f"table{tag}.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(text)
    return text


# ----------------------------------------------------------------------------- the command


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prepared", default=PREPARED, help="the prepared sample to read")
    ap.add_argument("--max-structures", type=int, default=0,
                    help="read only the first N structures of the sample (a quick look)")
    ap.add_argument("--rho", nargs="*", type=float, default=list(RHOS),
                    help="envelope widths to measure (default "
                         + ", ".join(f"{r:g}" for r in RHOS) + ")")
    ap.add_argument("--r-max", type=float, default=R_MAX,
                    help=f"radius of the environment about the midpoint, A (default {R_MAX:g})")
    ap.add_argument("--site-r", type=float, default=SITE_R,
                    help=f"cutoff of the site profile of the site-resolved reference, A "
                         f"(default {SITE_R:g})")
    ap.add_argument("--folds", type=int, default=FOLDS,
                    help=f"folds holding out whole structures (default {FOLDS})")
    ap.add_argument("--seed", type=int, default=0, help="the fold split and the draws (default 0)")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP,
                    help=f"paired bootstrap draws (default {BOOTSTRAP})")
    ap.add_argument("--jobs", type=int, default=8, help="processes for the overlap kernel")
    ap.add_argument("--report", action="store_true",
                    help="rebuild the table from the stored result and stop")
    return ap


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    tag = run_tag(args)

    if args.report:
        print(write_table(RESULTS, tag))
        return

    if not os.path.exists(args.prepared):
        raise SystemExit(f"{args.prepared} is missing: run "
                         f"benchmarks/prepare_litraj.py --fetch first")
    log_path = os.path.join(RESULTS, f"log{tag}.txt")
    log_file = open(log_path, "w", encoding="utf-8", newline="\n")

    def log(text: str):
        print(text, flush=True)
        log_file.write(text + "\n")
        log_file.flush()

    t0 = time.time()
    out = run(args, log)
    with open(os.path.join(RESULTS, f"public{tag}.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    minutes = (time.time() - t0) / 60
    log(f"done in {minutes:.1f} minutes")
    log_file.close()
    write_provenance(args, tag, argv, minutes=round(minutes, 1))
    print(write_table(RESULTS, tag))


if __name__ == "__main__":
    main()
