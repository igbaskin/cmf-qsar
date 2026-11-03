#!/usr/bin/env python
"""
A cation probe measured on the energy a cation gains: plan item EC7.

The probe module (``cmfqsar/probes.py``) turns the question "what would a small group gain by
sitting here?" into a field.  Item EC7 asks for that probe to be an ion -- lithium, sodium,
potassium or magnesium -- so that the field of an electrolyte solvent or an ionophore is the
energy a cation would feel in it.  Three things then have to be measured, and this script
measures them on ``examples/ether/``: 1,050 saturated acyclic ethers with the binding energy of
Li(+), Na(+) and Mg(2+) at wB97X-D3/def2-TZVP, each in the relaxed geometry of its lithium
complex, from the MLElectrolytes distribution of Wang et al., Chem. Commun. 56 (2020) 8920.

What is measured
----------------
**1. Where the probe says the ion sits** (``--stage probe``).  A probe field is a map of the
energy, so its deepest point is a prediction of the ion's position, and the data carry the
position the DFT relaxation gave.  This is the one test of a probe that needs no regression and
no fitting of any kind.  Two references that need no field: the centroid of the molecule's
oxygens, and the centroid of its heavy atoms.

**2. The binding energy** (``--stage bind``).  Nested cross-validation, five outer folds, the
ridge and the field width chosen inside each fold, under two fold designs: molecules at random,
and one oxygen count held out at a time, which asks for a chain length the model has not seen.
Five models on the same folds and the same samples:

    counts          the oxygen count, the carbon-to-oxygen ratio and the number of heavy atoms
    counts+homo     the same with the molecule's HOMO energy, which is a quantum quantity
    fields          the five ordinary fields, through the alignment-free kernel
    probe           the matching ion probe field alone
    fields+probe    the sum of the two kernels

The first two are the baselines item BT3 taught this program to measure against: a quantity that
is read off the formula, not computed from the structure.  The oxygen count alone has a
correlation of -0.62 with the lithium binding energy, so it is a baseline with real strength.

**3. The selectivity between two cations** (``--stage select``).  The difference of the lithium
and sodium energies is what an ionophore is designed for, and the oxygen count says nothing
about it (correlation -0.05).  The same five models are measured on that difference, with a
sixth: both ion probe fields together, in which the difference of the two fields is one of the
functions the ridge can form.  The stage closes with the multi-task part of item E1: the three
ions as three endpoints of one kernel, their covariance taken from descriptors of the ion itself
(radius, charge, polarisability), and each ion in turn removed **entirely** and predicted from
the other two.

What the frame is, and what it costs
------------------------------------
There is none.  A glyme has no scaffold to superimpose, so every kernel here is the
alignment-free field autocorrelation of ``cmfqsar/invariant.py``.  The coordinates are the
relaxed complex with the ion removed, so the conformation is the bound one for lithium and a
lithium-shaped one for sodium and magnesium; ``examples/ether/README.md`` states that limit.

Usage
-----
    python benchmarks/run_ether.py                      # every stage, then the table
    python benchmarks/run_ether.py --stage probe
    python benchmarks/run_ether.py --stage bind --limit 200
    python benchmarks/run_ether.py --report
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

from rdkit import Chem, RDLogger                                            # noqa: E402

from cmfqsar.cofields import field_volume, make_grid                        # noqa: E402
from cmfqsar.fields import FieldSet, build_fields                           # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               block_sizes, channel_scale, expanded_field_count, feature_matrix)
from cmfqsar.model import _Learner, cv_predict, r2_score, rmse              # noqa: E402
from cmfqsar.multitask import (Observations, cv_predict as mt_cv_predict,   # noqa: E402
                               descriptor_covariance, fit as mt_fit, log_evidence,
                               score_by_task, standardise as standardise_rows, task_covariance,
                               task_kernel)

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "ether")
DATA = os.path.join(ROOT, "examples", "ether")
SDF = os.path.join(DATA, "geometries.sdf")

#: the protocol constants of the other benchmarks, so that the numbers are comparable
SEED = 0
OUTER, INNER = 5, 10
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
#: widths of the ordinary fields, chosen inside each fold.  A probe field has no width of its
#: own: its Gaussians come from the expansion in cmfqsar/probes.py
ALPHAS = (0.3, 0.5, 1.0, 2.0)
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")

#: ion -> (probe name, the property holding its binding energy, printed label)
IONS = {
    "li": ("lithium", "li_ebind", "Li+"),
    "na": ("sodium", "na_ebind", "Na+"),
    "mg": ("magnesium", "mg_ebind", "Mg2+"),
}
ION_ORDER = ("li", "na", "mg")
#: descriptors of the endpoint itself: Shannon's six-coordinate ionic radius in angstrom, the
#: charge in electrons, and the free-ion polarisability in cubic angstrom (Mahan 1980).  They are
#: what lets an ion that was never measured still have a mean and a covariance row.
ION_DESCRIPTORS = ("ionic radius", "charge", "polarisability")
ION_FEATURES = {
    "li": (0.76, 1.0, 0.0280),
    "na": (1.02, 1.0, 0.1480),
    "mg": (0.72, 2.0, 0.0710),
}
#: the one descriptor the mean of an endpoint is a linear function of: the charge, which is
#: column 1 of ION_FEATURES. A divalent ion binds about three times as strongly as a monovalent
#: one, and with three endpoints a design matrix may hold an intercept and one column, no more.
MEAN_COLUMN = 1
#: the grid of the task covariance. Three endpoints do not support a finer one, and every
#: evaluation of the evidence is a solve over all the measurements at once
TASK_LENGTH_SCALES = (0.5, 1.0, 2.0)
TASK_NUGGETS = (0.1, 0.3)
#: the grid the probe field's deepest point is searched on, angstrom
GRID_MARGIN, GRID_STEP = 3.0, 0.4
#: how close the deepest point has to be to the relaxed ion to count as found
HIT = (1.0, 1.5)


# --------------------------------------------------------------------------------- the data


def load(limit=0):
    """The molecules, their names, the ion position and everything the table carries."""
    supplier = Chem.SDMolSupplier(SDF, removeHs=False, sanitize=True)
    mols, names, rows = [], [], []
    for m in supplier:
        if m is None:
            continue
        props = m.GetPropsAsDict()
        rows.append(props)
        names.append(m.GetProp("_Name"))
        mols.append(m)
    if not mols:
        raise SystemExit(f"{SDF} holds no molecule; run "
                         f"examples/make_ether_examples.py --fetch first")
    if limit and len(mols) > limit:
        keep = np.linspace(0, len(mols) - 1, limit).astype(int)
        mols = [mols[i] for i in keep]
        names = [names[i] for i in keep]
        rows = [rows[i] for i in keep]
    return mols, names, rows


def target(rows, key):
    """One endpoint as an array, with NaN where the value is missing upstream."""
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is None or v == "":
            continue
        out[i] = float(v)
    return out


def ion_position(row):
    return np.array([float(v) for v in str(row["ion_xyz"]).split()], float)


def count_features(mols, rows, with_homo=False):
    """The baseline read off the formula: what any model here has to beat.

    Two molecules of the distribution carry no HOMO energy. Dropping them would give the model
    that uses it a different sample from the others, and then no comparison would be paired, so
    the missing value is replaced by the median of the set and the count is reported.
    """
    X = []
    for m, r in zip(mols, rows):
        X.append([float(r["o_count"]), float(r["c_to_o"]), float(m.GetNumHeavyAtoms())])
    X = np.asarray(X, float)
    if not with_homo:
        return X, 0
    homo = np.array([float(r["homo"]) if r.get("homo") not in (None, "") else np.nan
                     for r in rows])
    missing = int(np.sum(~np.isfinite(homo)))
    if missing:
        homo = np.where(np.isfinite(homo), homo, float(np.nanmedian(homo)))
    return np.column_stack([X, homo]), missing


def standardise(train, other=None):
    """`other` (or `train` itself) centred and scaled by the training columns."""
    mean, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    rows = train if other is None else other
    return (rows - mean) / sd


def linear_kernel(A, B=None):
    B = A if B is None else B
    return np.asarray(B, float) @ np.asarray(A, float).T


# ------------------------------------------------------------------------------ the features


def field_features(mols, names, fields=FIELDS, probes=(), alpha=0.5):
    """The autocorrelation features of one field set, with its channel blocks."""
    fs = FieldSet.create(physchem=list(fields), probes=list(probes), alpha=alpha,
                         charge_model="gasteiger", param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    spec = AutocorrelationSpec()
    # a probe field's weights sit on its Gaussian terms, so the field set has to be passed: the
    # alignment-free kernel expands each term into a channel of its own
    F = feature_matrix(mf, spec, fieldset=fs)
    return F, spec, block_sizes(spec, expanded_field_count(fs))


def build_all_features(mols, names, probe_names, verbose=True):
    """Every feature matrix the stages need: the ordinary fields per width, each probe once."""
    out = {}
    t0 = time.time()
    for alpha in ALPHAS:
        out[("fields", alpha)] = field_features(mols, names, FIELDS, (), alpha)
        # the electrostatic field alone is the comparison item EC7 names for an ion probe
        out[("electrostatic", alpha)] = field_features(mols, names, ("electrostatic",), (), alpha)
        if verbose:
            print(f"  fields at alpha {alpha:g}: {out[('fields', alpha)][0].shape}, "
                  f"electrostatic {out[('electrostatic', alpha)][0].shape} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    for pr in probe_names:
        # a probe field's Gaussians come from its own expansion, so alpha does not enter
        out[("probe", pr)] = field_features(mols, names, (), (pr,), 1.0)
        if verbose:
            print(f"  probe {pr}: {out[('probe', pr)][0].shape} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    return out


def kernel_of(entry, train_rows, rows=None, scale=None):
    """The cosine-normalised autocorrelation kernel of one feature matrix."""
    F, spec, blocks = entry
    A = F[train_rows]
    sc = channel_scale(A, spec.n_bins, blocks=blocks) if scale is None else scale
    B = None if rows is None else F[rows]
    return autocorrelation_kernel(A, B, scale=sc, n_bins=spec.n_bins, blocks=blocks), sc


# ------------------------------------------------------------------- stage 1: where the ion is


def deepest_point(mol, name, probe_name):
    """The point where the probe field is deepest, and the field's value there."""
    fs = FieldSet.create(physchem=[], probes=[probe_name], alpha=1.0,
                         charge_model="gasteiger", param_scheme="rdkit")
    mf = build_fields(mol, fs, name)
    grid = make_grid(mf.coords, margin=GRID_MARGIN, spacing=GRID_STEP)
    volume = field_volume(mf, fs.specs[0], 0, grid).ravel()
    k = int(np.argmin(volume))
    return grid.points()[k], float(volume[k])


def probe_stage(mols, names, rows, probe_name="lithium"):
    """Does the probe field put the ion where the relaxation put it?"""
    t0 = time.time()
    d_probe, d_oxygen, d_heavy, depths = [], [], [], []
    for i, (m, n, r) in enumerate(zip(mols, names, rows)):
        ion = ion_position(r)
        point, depth = deepest_point(m, n, probe_name)
        pos = m.GetConformer().GetPositions()
        oxygen = np.array([k for k, a in enumerate(m.GetAtoms()) if a.GetSymbol() == "O"])
        heavy = np.array([k for k, a in enumerate(m.GetAtoms()) if a.GetSymbol() != "H"])
        d_probe.append(float(np.linalg.norm(point - ion)))
        d_oxygen.append(float(np.linalg.norm(pos[oxygen].mean(axis=0) - ion)))
        d_heavy.append(float(np.linalg.norm(pos[heavy].mean(axis=0) - ion)))
        depths.append(depth)
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(mols)} ({time.time() - t0:.0f} s)", flush=True)

    def describe(d):
        d = np.asarray(d, float)
        rec = {"mean": float(d.mean()), "sd": float(d.std(ddof=0)),
               "median": float(np.median(d))}
        for h in HIT:
            rec[f"within_{h:g}A"] = float(np.mean(d <= h))
        return rec

    out = {"probe": probe_name, "n": len(mols), "grid_step": GRID_STEP,
           "grid_margin": GRID_MARGIN,
           "probe_minimum": describe(d_probe),
           "oxygen_centroid": describe(d_oxygen),
           "heavy_atom_centroid": describe(d_heavy),
           "depth_mean": float(np.mean(depths)), "depth_sd": float(np.std(depths)),
           "elapsed_s": time.time() - t0}
    for key in ("probe_minimum", "oxygen_centroid", "heavy_atom_centroid"):
        rec = out[key]
        print(f"  {key:20s} {rec['mean']:.2f} +/- {rec['sd']:.2f} A from the relaxed ion, "
              f"{rec['within_1A']:.0%} within 1.0 A", flush=True)
    return out


# ----------------------------------------------------------------- the folds and the regression


def random_folds(n, k=OUTER, seed=SEED):
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    return [(np.setdiff1d(np.arange(n), g), g) for g in np.array_split(order, k)]


def grouped_folds(groups):
    """One fold per distinct group: here, one oxygen count held out at a time."""
    groups = np.asarray(groups)
    out = []
    for g in sorted(set(groups.tolist())):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        if len(te) and len(tr):
            out.append((tr, te))
    return out


def model_kernel(model, feats, ion_probes, alpha, rows_tr, rows_te=None, X=None, scales=None):
    """The kernel of one named model, on the training rows and optionally the test rows."""
    scales = {} if scales is None else scales
    parts_tr, parts_te = [], []
    if model in ("counts", "counts+homo"):
        Xs_tr = standardise(X[rows_tr])
        K_tr = linear_kernel(Xs_tr)
        if rows_te is None:
            return K_tr, None, scales
        return K_tr, linear_kernel(Xs_tr, standardise(X[rows_tr], X[rows_te])), scales
    source = ("electrostatic" if model == "electrostatic" else "fields")
    if model in ("fields", "fields+probe", "electrostatic"):
        key = (source, alpha)
        K, sc = kernel_of(feats[key], rows_tr, None, scales.get(key))
        scales[key] = sc
        parts_tr.append(K)
        if rows_te is not None:
            parts_te.append(kernel_of(feats[key], rows_tr, rows_te, sc)[0])
    if model in ("probe", "fields+probe", "two probes"):
        for pr in ion_probes:
            K, sc = kernel_of(feats[("probe", pr)], rows_tr, None, scales.get(("probe", pr)))
            scales[("probe", pr)] = sc
            parts_tr.append(K)
            if rows_te is not None:
                parts_te.append(kernel_of(feats[("probe", pr)], rows_tr, rows_te, sc)[0])
    K_tr = parts_tr[0] if len(parts_tr) == 1 else np.sum(parts_tr, axis=0)
    K_te = None
    if rows_te is not None:
        K_te = parts_te[0] if len(parts_te) == 1 else np.sum(parts_te, axis=0)
    return K_tr, K_te, scales


def nested(model, feats, ion_probes, y, folds, X=None, label=""):
    """Out-of-fold predictions, with the ridge and the field width chosen inside each fold."""
    pred = np.full(len(y), np.nan)
    per_fold = []
    alphas = ALPHAS if model in ("fields", "fields+probe", "electrostatic") else (ALPHAS[0],)
    t0 = time.time()
    scales = {}
    for k, (tr, te) in enumerate(folds):
        best = None
        for alpha in alphas:
            K_tr, _unused, scales = model_kernel(model, feats, ion_probes, alpha, tr,
                                                 None, X, scales)
            for lam in LAMS:
                p = cv_predict(K_tr, y[tr], "krr", {"lam": lam}, folds=INNER, seed=SEED)
                err = rmse(y[tr], p)
                if best is None or err < best[0]:
                    best = (err, alpha, lam)
        _err, alpha, lam = best
        K_tr, K_te, scales = model_kernel(model, feats, ion_probes, alpha, tr, te, X, scales)
        learner = _Learner("krr", {"lam": lam}, True).fit(K_tr, y[tr])
        pred[te] = learner.predict(K_te)
        per_fold.append({"fold": k, "alpha": alpha, "lam": lam, "n_test": int(len(te)),
                         "mae": float(np.mean(np.abs(pred[te] - y[te]))),
                         "rmse": float(rmse(y[te], pred[te]))})
    maes = np.array([f["mae"] for f in per_fold])
    rec = {"model": model, "n": int(len(y)), "mae_mean": float(maes.mean()),
           "mae_sd": float(maes.std(ddof=0)),
           "mae_pooled": float(np.mean(np.abs(pred - y))),
           "rmse_pooled": float(rmse(y, pred)), "r2_pooled": float(r2_score(y, pred)),
           "per_fold": per_fold, "elapsed_s": time.time() - t0,
           "abs_errors": [float(v) for v in np.abs(pred - y)]}
    print(f"    {label}{model:<14} MAE {rec['mae_mean']:.3f} +/- {rec['mae_sd']:.3f} eV, "
          f"R2 {rec['r2_pooled']:+.3f}  ({rec['elapsed_s']:.0f} s)", flush=True)
    return rec


def ion_task_covariance(K, obs, Zs, rows):
    """The task kernel over a subset of the ions, chosen by the evidence.

    The ions' descriptors are standardised over **all three** ions rather than over the subset:
    the spread of the ion properties is a property of the set of ions, and standardising over
    two of them puts those two at a fixed distance from each other whatever they are, which
    would make every pair of endpoints equally related.

    Returns None when no setting gives a finite evidence. That happens when the design matrix of
    the mean is singular, which it is as soon as the charge does not vary among the endpoints
    that remain: the mean of the removed ion is then not a quantity the data can reach.
    """
    best = None
    for ls in TASK_LENGTH_SCALES:
        for eps in TASK_NUGGETS:
            T = task_kernel(Zs[rows], float(ls), float(eps), "rbf")
            for lam in LAMS[1:4]:
                v = log_evidence(K, obs, T, float(lam), fit_bias=True)
                if np.isfinite(v) and (best is None or v > best[0]):
                    best = (float(v), T, float(lam), float(ls), float(eps))
    return best


def paired_interval(a, b, seed=SEED, draws=2000):
    """The paired difference of two models' absolute errors, with a bootstrap interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(draws)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(np.mean(d)), "low": float(lo), "high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


MODELS = ("counts", "counts+homo", "electrostatic", "fields", "probe", "fields+probe")


# ------------------------------------------------------------------- stage 2: the binding energy


def bind_stage(mols, names, rows, feats):
    """Every model on every ion, under both fold designs."""
    out = {"ions": {}, "models": list(MODELS), "lams": list(LAMS), "alphas": list(ALPHAS),
           "fields": list(FIELDS), "outer": OUTER, "inner": INNER}
    counts, _none = count_features(mols, rows, with_homo=False)
    counts_homo, missing_homo = count_features(mols, rows, with_homo=True)
    out["homo_filled_with_the_median"] = missing_homo
    o_count = np.array([float(r["o_count"]) for r in rows])
    for ion in ION_ORDER:
        probe_name, prop, label = IONS[ion]
        y_all = target(rows, prop)
        ok = np.where(np.isfinite(y_all))[0]
        y = y_all[ok]
        print(f"  {label}: {len(ok)} molecules, {y.min():.2f} to {y.max():.2f} eV, "
              f"mean {y.mean():.2f}, sd {y.std(ddof=0):.2f}", flush=True)
        sub = {key: (entry[0][ok], entry[1], entry[2]) for key, entry in feats.items()}
        designs = {"random": random_folds(len(ok)),
                   "one oxygen count held out": grouped_folds(o_count[ok])}
        rec = {"n": int(len(ok)), "mean": float(y.mean()), "sd": float(y.std(ddof=0)),
               "mae_of_the_mean": float(np.mean(np.abs(y - y.mean()))), "designs": {}}
        for design, folds in designs.items():
            print(f"   {design}: {len(folds)} folds", flush=True)
            got = {}
            for model in MODELS:
                X = counts_homo if model == "counts+homo" else counts
                got[model] = nested(model, sub, (probe_name,), y, folds, X[ok],
                                    label=f"{design[:6]:<7}")
            rec["designs"][design] = {
                "models": got,
                "probe_vs_counts": paired_interval(got["probe"]["abs_errors"],
                                                   got["counts"]["abs_errors"]),
                "fields_vs_counts": paired_interval(got["fields"]["abs_errors"],
                                                    got["counts"]["abs_errors"]),
                "probe_vs_fields": paired_interval(got["probe"]["abs_errors"],
                                                   got["fields"]["abs_errors"]),
                "probe_vs_electrostatic": paired_interval(
                    got["probe"]["abs_errors"], got["electrostatic"]["abs_errors"]),
                "both_vs_counts": paired_interval(got["fields+probe"]["abs_errors"],
                                                  got["counts"]["abs_errors"]),
            }
        out["ions"][ion] = rec
    return out


# -------------------------------------------------------------- stage 3: selectivity and sharing


def select_stage(mols, names, rows, feats):
    """The lithium-minus-sodium difference, and the three ions as three endpoints."""
    out = {}
    li, na = target(rows, "li_ebind"), target(rows, "na_ebind")
    ok = np.where(np.isfinite(li) & np.isfinite(na))[0]
    y = li[ok] - na[ok]
    o_count = np.array([float(r["o_count"]) for r in rows])
    counts, _none = count_features(mols, rows, with_homo=False)
    counts_homo, missing_homo = count_features(mols, rows, with_homo=True)
    sub = {key: (entry[0][ok], entry[1], entry[2]) for key, entry in feats.items()}
    print(f"  selectivity on {len(ok)} molecules: {y.min():+.2f} to {y.max():+.2f} eV, "
          f"mean {y.mean():+.2f}, sd {y.std(ddof=0):.2f}", flush=True)
    folds = random_folds(len(ok))
    got = {}
    for model in MODELS + ("two probes",):
        X = counts_homo if model == "counts+homo" else counts
        probes = ("lithium", "sodium") if model == "two probes" else ("lithium",)
        got[model] = nested(model, sub, probes, y, folds, X[ok])
    out["selectivity"] = {
        "n": int(len(ok)), "mean": float(y.mean()), "sd": float(y.std(ddof=0)),
        "mae_of_the_mean": float(np.mean(np.abs(y - y.mean()))),
        "r_with_oxygen_count": float(np.corrcoef(o_count[ok], y)[0, 1]),
        "models": got,
        "two_probes_vs_counts": paired_interval(got["two probes"]["abs_errors"],
                                                got["counts"]["abs_errors"]),
        "two_probes_vs_one": paired_interval(got["two probes"]["abs_errors"],
                                             got["probe"]["abs_errors"]),
    }

    # ---- the three ions as three endpoints of one kernel
    Y_all = np.column_stack([target(rows, IONS[i][1]) for i in ION_ORDER])
    keep = np.where(np.any(np.isfinite(Y_all), axis=1))[0]
    Y = Y_all[keep]
    Z = np.array([ION_FEATURES[i] for i in ION_ORDER], float)
    # the mean is a linear function of the charge alone. With three endpoints a design of more
    # columns than that fits each endpoint's mean exactly, which is the free bias it replaces.
    Zi = np.column_stack([np.ones(len(ION_ORDER)), Z[:, MEAN_COLUMN]])
    Zs = standardise_rows(Z)                     # standardised over the whole set of ions
    obs = Observations.from_columns(Y, names=[IONS[i][2] for i in ION_ORDER],
                                    task_features=Zi)
    F, spec, blocks = feats[("fields", 0.5)]
    A = F[keep]
    K = autocorrelation_kernel(A, scale=channel_scale(A, spec.n_bins, blocks=blocks),
                               n_bins=spec.n_bins, blocks=blocks)
    print(f"  three endpoints: {len(obs)} measurements over {len(keep)} molecules "
          f"{obs.counts()}", flush=True)
    shared = {}
    for label, T in (("independent", np.eye(len(ION_ORDER))),
                     ("one correlation 0.5", task_covariance(len(ION_ORDER), 0.5)),
                     ("one correlation 0.9", task_covariance(len(ION_ORDER), 0.9))):
        pred = mt_cv_predict(K, obs, lam=0.1, folds=OUTER, seed=SEED, T=T)
        shared[label] = {"by_task": score_by_task(obs, pred),
                         "mae": float(np.nanmean(np.abs(pred - obs.y)))}
        print(f"    {label:<22} MAE {shared[label]['mae']:.3f} eV", flush=True)
    dc = descriptor_covariance(K, obs, Z, length_scales=TASK_LENGTH_SCALES,
                               nuggets=TASK_NUGGETS, lams=LAMS[1:4])
    pred = mt_cv_predict(K, obs, lam=dc.lam, folds=OUTER, seed=SEED, T=dc.T)
    shared["descriptors"] = {"by_task": score_by_task(obs, pred),
                             "mae": float(np.nanmean(np.abs(pred - obs.y))),
                             "length_scale": dc.length_scale, "nugget": dc.nugget,
                             "lam": dc.lam, "evidence": dc.evidence,
                             "T": [[float(v) for v in row] for row in dc.T]}
    print(f"    {'descriptors':<22} MAE {shared['descriptors']['mae']:.3f} eV "
          f"(length scale {dc.length_scale:g}, nugget {dc.nugget:g})", flush=True)

    held = {}
    for t, ion in enumerate(ION_ORDER):
        others = [s for s in range(len(ION_ORDER)) if s != t]
        truth_all = Y[:, t]
        has = np.where(np.isfinite(truth_all))[0]
        truth = truth_all[has]
        Zk = np.column_stack([np.ones(len(others)), Z[others][:, MEAN_COLUMN]])
        z_new = np.array([1.0, Z[t, MEAN_COLUMN]], float)
        obs_k = Observations.from_columns(Y[:, others],
                                         names=[IONS[ION_ORDER[s]][2] for s in others],
                                         task_features=Zk)
        # the reference that needs no model: the mean of the ions that were measured
        block = Y[np.ix_(has, others)]
        have = np.any(np.isfinite(block), axis=1)
        mean_of_others = np.full(len(has), np.nan)
        mean_of_others[have] = np.nanmean(block[have], axis=1)
        varies = bool(np.ptp(Z[others][:, MEAN_COLUMN]) > 1e-9)
        rec = {"n": int(len(has)), "charge_varies_in_the_rest": varies,
               "mae_of_the_other_ions": (float(np.mean(np.abs(
                   mean_of_others[have] - truth[have]))) if have.any() else float("nan")),
               "mae_of_the_training_mean": float(np.mean(np.abs(truth - np.mean(obs_k.y)))),
               "true_mean": float(truth.mean())}
        best = ion_task_covariance(K, obs_k, Zs, others) if varies else None
        if best is None:
            rec["refused"] = (
                "the charge is the same in both ions that remain, so the design matrix of the "
                "mean has two identical rows and the mean of the removed ion is not a quantity "
                "the data determine. A divalent ion cannot be predicted from two monovalent "
                "ones." if not varies else
                "no setting of the task covariance has a finite evidence")
            print(f"    {IONS[ion][2]} removed entirely: refused, "
                  f"{'the charge does not vary in the two ions that remain' if not varies else 'the evidence is not finite anywhere'}; "
                  f"the other ions' mean is {rec['mae_of_the_other_ions']:.3f} eV away",
                  flush=True)
            held[ion] = rec
            continue
        _v, T, lam, ls, eps = best
        f = mt_fit(K, obs_k, lam=lam, T=T)
        row = task_kernel(Zs[others], ls, eps, "rbf", Z_new=Zs[[t]])[0]
        pred = f.predict_new_task(K, row, features=z_new)[has]
        rec.update({"mae": float(np.mean(np.abs(pred - truth))),
                    "r2": float(r2_score(truth, pred)),
                    "length_scale": ls, "nugget": eps, "lam": lam,
                    "covariance_row": [float(v) for v in row],
                    "predicted_mean": float(pred.mean())})
        held[ion] = rec
        print(f"    {IONS[ion][2]} removed entirely: MAE {rec['mae']:.3f} eV, "
              f"R2 {rec['r2']:+.3f}; the other ions' mean gives "
              f"{rec['mae_of_the_other_ions']:.3f}", flush=True)
    out["sharing"] = {"covariances": shared, "held_out_ion": held,
                      "descriptors": list(ION_DESCRIPTORS),
                      "features": {i: list(ION_FEATURES[i]) for i in ION_ORDER}}
    return out


# ----------------------------------------------------------------------------------- the report


def _row(name, rec):
    return (f"| {name} | {rec['mae_mean']:.3f} +/- {rec['mae_sd']:.3f} | "
            f"{rec['r2_pooled']:+.3f} |")


def write_table(path=None):
    """The table of the three stages, from the JSON each one wrote."""
    path = path or RESULTS
    out = ["# A cation probe on ether binding energies (plan item EC7)", ""]
    probe_json = os.path.join(path, "probe.json")
    if os.path.exists(probe_json):
        p = json.load(open(probe_json, encoding="utf-8"))
        out += [f"## Where the probe says the ion sits", "",
                f"The deepest point of the {p['probe']} probe field against the position the "
                f"relaxation gave, on {p['n']} complexes, searched on a grid of "
                f"{p['grid_step']} A.", "",
                "| Prediction of the ion position | Distance from the relaxed ion (A) | "
                "Within 1.0 A | Within 1.5 A |", "|---|---|---|---|"]
        for key, label in (("probe_minimum", "deepest point of the probe field"),
                           ("oxygen_centroid", "centroid of the oxygens"),
                           ("heavy_atom_centroid", "centroid of the heavy atoms")):
            r = p[key]
            out.append(f"| {label} | {r['mean']:.2f} +/- {r['sd']:.2f} | "
                       f"{r['within_1A']:.0%} | {r['within_1.5A']:.0%} |")
        out.append("")
    bind_json = os.path.join(path, "bind.json")
    if os.path.exists(bind_json):
        b = json.load(open(bind_json, encoding="utf-8"))
        out += ["## The binding energy", "",
                f"Nested cross-validation, {b['outer']} outer folds, the ridge and the field "
                f"width chosen on {b['inner']} inner folds. Mean absolute error over the folds, "
                f"in eV.", ""]
        for ion in ION_ORDER:
            if ion not in b["ions"]:
                continue
            rec = b["ions"][ion]
            out += [f"### {IONS[ion][2]}: {rec['n']} molecules, "
                    f"mean {rec['mean']:.2f} eV, standard deviation {rec['sd']:.2f}", ""]
            for design, d in rec["designs"].items():
                out += [f"**Folds: {design}.** Predicting the mean of the set costs "
                        f"{rec['mae_of_the_mean']:.3f} eV.", "",
                        "| Model | MAE (eV) | R2 |", "|---|---|---|"]
                for model in b["models"]:
                    if model in d["models"]:
                        out.append(_row(model, d["models"][model]))
                out.append("")
                for key, label in (("fields_vs_counts", "fields minus counts"),
                                   ("probe_vs_counts", "probe minus counts"),
                                   ("probe_vs_electrostatic",
                                    "probe minus the electrostatic field alone"),
                                   ("probe_vs_fields", "probe minus fields"),
                                   ("both_vs_counts", "fields+probe minus counts")):
                    v = d.get(key)
                    if v is None:
                        continue
                    out.append(f"* Paired over the same molecules, {label}: "
                               f"{v['mean']:+.3f} eV [{v['low']:+.3f}, {v['high']:+.3f}], "
                               f"{'excludes' if v['excludes_zero'] else 'includes'} zero.")
                out.append("")
    sel_json = os.path.join(path, "select.json")
    if os.path.exists(sel_json):
        s = json.load(open(sel_json, encoding="utf-8"))
        sel = s["selectivity"]
        out += ["## The selectivity between lithium and sodium", "",
                f"The difference of the two binding energies on the {sel['n']} molecules that "
                f"carry both: mean {sel['mean']:+.3f} eV, standard deviation {sel['sd']:.3f}, "
                f"correlation with the oxygen count {sel['r_with_oxygen_count']:+.3f}. "
                f"Predicting the mean costs {sel['mae_of_the_mean']:.3f} eV.", "",
                "| Model | MAE (eV) | R2 |", "|---|---|---|"]
        for model, rec in sel["models"].items():
            out.append(_row(model, rec))
        out.append("")
        for key, label in (("two_probes_vs_counts", "two probes minus counts"),
                           ("two_probes_vs_one", "two probes minus the lithium probe alone")):
            v = sel[key]
            out.append(f"* Paired, {label}: {v['mean']:+.3f} eV "
                       f"[{v['low']:+.3f}, {v['high']:+.3f}], "
                       f"{'excludes' if v['excludes_zero'] else 'includes'} zero.")
        share = s["sharing"]
        out += ["", "## The three ions as three endpoints", "",
                "One kernel, the covariance between the endpoints as named. The descriptor "
                "covariance takes it from the ion's "
                + ", ".join(share["descriptors"]) + ".", "",
                "| Covariance | MAE (eV) | " + " | ".join(f"q2 {IONS[i][2]}"
                                                           for i in ION_ORDER) + " |",
                "|---|---|" + "---|" * len(ION_ORDER)]
        for label, rec in share["covariances"].items():
            q2 = " | ".join(f"{t['q2']:+.3f}" for t in rec["by_task"])
            out.append(f"| {label} | {rec['mae']:.3f} | {q2} |")
        out += ["", "**An ion removed entirely** and predicted from the other two through the "
                "descriptor covariance, beside the reference that needs no model: the mean of "
                "the ions that were measured.", "",
                "| Ion removed | n | MAE (eV) | R2 | MAE of the other ions' mean (eV) |",
                "|---|---|---|---|---|"]
        for ion in ION_ORDER:
            if ion not in share["held_out_ion"]:
                continue
            h = share["held_out_ion"][ion]
            if "refused" in h:
                out.append(f"| {IONS[ion][2]} | {h['n']} | refused | refused | "
                           f"{h['mae_of_the_other_ions']:.3f} |")
                continue
            out.append(f"| {IONS[ion][2]} | {h['n']} | {h['mae']:.3f} | {h['r2']:+.3f} | "
                       f"{h['mae_of_the_other_ions']:.3f} |")
        for ion in ION_ORDER:
            h = share["held_out_ion"].get(ion, {})
            if "refused" in h:
                why = str(h["refused"]).rstrip(".")
                out += ["", f"{IONS[ion][2]} is refused: {why}."]
        out.append("")
    text = "\n".join(out) + "\n"
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="A cation probe on ether binding energies (EC7).")
    ap.add_argument("--stage", choices=("probe", "bind", "select", "all"), default="all")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only this many molecules, for a quick check")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    if args.report:
        print(write_table())
        return 0

    mols, names, rows = load(args.limit)
    print(f"{len(mols)} molecules from {os.path.relpath(SDF, ROOT)}", flush=True)
    stages = ("probe", "bind", "select") if args.stage == "all" else (args.stage,)

    feats = None
    if "bind" in stages or "select" in stages:
        feats = build_all_features(mols, names, [IONS[i][0] for i in ION_ORDER])

    if "probe" in stages:
        print("stage probe: where the probe field says the ion sits", flush=True)
        out = probe_stage(mols, names, rows)
        json.dump(out, open(os.path.join(RESULTS, "probe.json"), "w", encoding="utf-8"),
                  indent=1)
    if "bind" in stages:
        print("stage bind: the binding energy of each ion", flush=True)
        out = bind_stage(mols, names, rows, feats)
        json.dump(out, open(os.path.join(RESULTS, "bind.json"), "w", encoding="utf-8"), indent=1)
    if "select" in stages:
        print("stage select: the selectivity, and the three ions as three endpoints", flush=True)
        out = select_stage(mols, names, rows, feats)
        json.dump(out, open(os.path.join(RESULTS, "select.json"), "w", encoding="utf-8"),
                  indent=1)
    write_table()
    print(f"wrote {os.path.join(RESULTS, 'table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
