#!/usr/bin/env python
"""A kinetic selectivity as the activity: chiral phosphoric acids on 1,075 N,S-acetal formations.

The series (``examples/make_denmark_examples.py``) is the screen of Zahrt et al., Science 363
(2019) eaau5631: 43 catalysts, each on the 25 combinations of 5 imines and 5 thiols, with the
enantiomeric excess of every reaction turned into ``ddG = R T ln((1 + ee) / (1 - ee))`` at
298.15 K, the free-energy difference between the two diastereomeric transition states.  It is a
difference of two rate constants, so the activity is kinetic, and it is the first catalytic
series in the repository that is machine-readable.

The model.  The catalyst is the molecule and the frame is the phosphoric acid (phosphorus at
the origin, acid oxygens on +z, the two C2 images as the conformers of an ensemble).  The 25
substrate combinations are 25 endpoints of one multi-task model (``cmfqsar.multitask``): one
kernel over the catalysts from the five physico-chemical fields, one covariance between the
endpoints, and the pair kernel ``K(catalyst, catalyst') T(endpoint, endpoint')``.  Two
covariances are measured: one shared correlation (every pair of substrate combinations alike),
and a covariance read off the substrates themselves (which imine, which thiol), which is the
form that can give a never-measured substrate combination a row.  Alpha, lambda and the
covariance are chosen by the marginal likelihood on the whole table, as the two-endpoint
benchmark does.

What is measured, each with the baseline that predicts the same thing without a field:

  catalysts     leave-one-catalyst-out: every reaction of one catalyst is predicted from the
                42 others.  Baseline: the mean of the endpoint over the training catalysts.
                This is the published paper's "external catalysts" case (MAD 0.211 kcal/mol on
                19 held-out catalysts trained on 24).
  reactions     10-fold over the reactions: a reaction is predicted from the others, among them
                the same catalyst on other substrates.  Baseline: the additive model
                ``catalyst effect + substrate effect``, fitted by least squares.
  substrates    leave-one-substrate-combination-out: an endpoint no catalyst was measured on,
                predicted from its imine and its thiol through the descriptor covariance.
                Baseline: the catalyst's mean over the other 24 endpoints.
  imines        leave-one-imine-out (5 endpoints at a time), and thiols likewise.

Every number comes with the published comparison where there is one, a bootstrap interval over
the catalysts for the difference from the baseline, and the field-by-field reading of the
catalyst holdout (which field carries the selectivity), as the Hammett benchmark does.  The
same five fields are also compared without a frame, through the rotation-invariant kernel, for
six widths of that kernel, each with the interval of its difference from the framed model.  The
co-field of the model, averaged over the endpoints, is drawn in the acid's frame.

Usage
-----
    python benchmarks/run_denmark.py                 # everything, results/denmark/
    python benchmarks/run_denmark.py --single        # one C2 image instead of the pair
    python benchmarks/run_denmark.py --quick         # a coarser grid, for a smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from cmfqsar import multitask as MT  # noqa: E402
from cmfqsar.cofields import evaluate_gaussians  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, mae, r2_score, rmse  # noqa: E402

DATA = os.path.join(ROOT, "examples", "denmark")
OUT_DIR = os.path.join(HERE, "results", "denmark")

PHYSCHEM = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
LAMS = [1e-3, 1e-2, 0.1, 1.0, 10.0]
RHOS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
#: the kernel without a frame (the autocorrelation of the fields): the widths tried along its
#: distance axis, in A, and the longest distance it resolves.  Until 14 November 2026 only the
#: default width of 0.75 A was run, which is far from the best one on this series.
AC_WIDTHS = [0.15, 0.25, 0.35, 0.5, 0.75, 1.5]
AC_D_MAX = 16.0
IMINES = ["1", "2", "3", "4", "5"]
THIOLS = ["A", "B", "C", "D", "E"]
PUBLISHED = {"external catalysts": 0.211, "new substrate combinations": 0.161}
RUNS = 1000
SEED = 0


# --------------------------------------------------------------------------------- the table


def read_reactions(path: str) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["ddG_kcal"] = float(r["ddG_kcal"])
        r["ee_percent"] = float(r["ee_percent"])
    return rows


def task_name(imine: str, thiol: str) -> str:
    return f"{imine}_{thiol}"


def observations(rows: Sequence[dict], names: Sequence[str]) -> MT.Observations:
    """(43 catalysts x 25 endpoints), in the order of the catalyst file and of the endpoints.

    The endpoints run imines outer and thiols inner, so endpoint ``t`` is imine ``t // 5`` and
    thiol ``t % 5``.
    """
    tasks = [task_name(i, t) for i in IMINES for t in THIOLS]
    Y = np.full((len(names), len(tasks)), np.nan)
    index = {n: k for k, n in enumerate(names)}
    for r in rows:
        Y[index[r["catalyst"]], tasks.index(task_name(r["imine"], r["thiol"]))] = r["ddG_kcal"]
    return MT.Observations.from_columns(Y, names=tasks)


def imine_of(task: np.ndarray) -> np.ndarray:
    return np.asarray(task, int) // len(THIOLS)


def thiol_of(task: np.ndarray) -> np.ndarray:
    return np.asarray(task, int) % len(THIOLS)


def task_descriptors() -> np.ndarray:
    """One-hot imine and one-hot thiol: endpoints sharing a substrate are close."""
    rows = []
    for i in IMINES:
        for t in THIOLS:
            rows.append([1.0 if i == j else 0.0 for j in IMINES]
                        + [1.0 if t == u else 0.0 for u in THIOLS])
    return np.array(rows, float)


# -------------------------------------------------------------------------------- the kernels


def load_catalysts(path: str, ensemble: bool):
    ds = load_dataset(path, ensemble_by="name" if ensemble else None)
    return ds.mols, ds.names


def catalyst_model(mols, names, fields: Sequence[str], alpha: float, ensemble: bool,
                   source: str = "fields",
                   ac_sigma: Optional[float] = None) -> Tuple[np.ndarray, CMFModel, AtomTable]:
    fs = FieldSet.create(physchem=list(fields), alpha=alpha,
                         conformers="ensemble" if ensemble else "single")
    mf = [build_fields(m, fs, name=n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    extra = {} if ac_sigma is None else {"ac_sigma": float(ac_sigma), "ac_d_max": AC_D_MAX}
    model = CMFModel(fieldset=fs, method="krr", fit_bias=True, kernel_source=source,
                     normalize=(source != "fields"), **extra)
    K = model.training_kernel(table)
    return K, model, table


# ---------------------------------------------------------------------------- the selections


def choose_rho(kernels: Dict[float, np.ndarray], obs: MT.Observations,
               lams: Sequence[float]) -> dict:
    """(alpha, lambda, rho) of the one-correlation covariance, by the evidence.

    The correlation grid reaches 0.99: the 25 endpoints share the catalyst's selectivity to a
    large extent, and the default grid of :func:`cmfqsar.multitask.evidence_rho` stops at 0.95.
    """
    best = None
    for alpha, K in kernels.items():
        for lam in lams:
            f = MT.evidence_rho(K, obs, rhos=RHOS, lams=[lam])
            if best is None or f.evidence > best["evidence"]:
                best = {"alpha": float(alpha), "lam": float(lam), "evidence": float(f.evidence),
                        "rho": float(f.correlations[0, 1]), "T": f.T}
    return best


def choose_descriptor(kernels: Dict[float, np.ndarray], obs: MT.Observations, Z: np.ndarray,
                      lams: Sequence[float]) -> dict:
    """(alpha, lambda, length scale, nugget) of the substrate covariance, by the evidence."""
    best = None
    for alpha, K in kernels.items():
        d = MT.descriptor_covariance(K, obs, Z, lams=lams)
        if best is None or d.evidence > best["evidence"]:
            best = {"alpha": float(alpha), "lam": float(d.lam), "evidence": float(d.evidence),
                    "length_scale": float(d.length_scale), "nugget": float(d.nugget), "T": d.T}
    return best


# ------------------------------------------------------------------------------- the splits


def predict_pairs(K: np.ndarray, f: MT.MultiTaskFit, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    return np.array([float(K[m] @ f.coef[:, t] + f.bias[t]) for m, t in pairs])


def cv_catalysts(K: np.ndarray, obs: MT.Observations, T: np.ndarray, lam: float) -> np.ndarray:
    return MT.cv_predict(K, obs, lam=lam, T=T, folds=obs.n_mol, seed=None)


def cv_reactions(K: np.ndarray, obs: MT.Observations, T: np.ndarray, lam: float,
                 folds: int = 10, seed: int = SEED) -> np.ndarray:
    out = np.full(len(obs), np.nan)
    order = np.random.RandomState(seed).permutation(len(obs))
    for g in np.array_split(order, folds):
        tr = np.setdiff1d(np.arange(len(obs)), g)
        f = MT.fit(K, obs.subset(tr), lam=lam, T=T)
        out[g] = predict_pairs(K, f, [(int(obs.molecule[p]), int(obs.task[p])) for p in g])
    return out


def held_out_bias(bias: np.ndarray, trained: Sequence[int], task: int) -> float:
    """The mean of an endpoint no catalyst was measured on, from the fitted biases.

    The bias of a trained endpoint is its mean over the catalysts (the ``sum a = 0`` constraint
    of the fit).  A new endpoint gets the additive estimate: the mean over the trained endpoints,
    plus the deviation of those sharing its imine, plus the deviation of those sharing its
    thiol; a factor with no trained endpoint contributes nothing.
    """
    trained = np.asarray(trained, int)
    b = np.asarray(bias, float)[trained]
    mean = float(b.mean())
    same_imine = imine_of(trained) == imine_of(np.array([task]))[0]
    same_thiol = thiol_of(trained) == thiol_of(np.array([task]))[0]
    dev_i = float(b[same_imine].mean() - mean) if same_imine.any() else 0.0
    dev_t = float(b[same_thiol].mean() - mean) if same_thiol.any() else 0.0
    return mean + dev_i + dev_t


def cv_tasks(K: np.ndarray, obs: MT.Observations, T: np.ndarray, lam: float,
             groups: Sequence[Sequence[int]]) -> np.ndarray:
    """Hold out whole endpoints.

    The held-out endpoint's row of the covariance ties its coefficients to the trained
    endpoints (that is what the substrate covariance is for); its mean is the additive estimate
    of :func:`held_out_bias`.
    """
    out = np.full(len(obs), np.nan)
    for g in groups:
        held = set(int(t) for t in g)
        te = np.array([p for p in range(len(obs)) if int(obs.task[p]) in held], int)
        tr = np.array([p for p in range(len(obs)) if int(obs.task[p]) not in held], int)
        f = MT.fit(K, obs.subset(tr), lam=lam, T=T)
        trained = sorted(set(int(t) for t in obs.task[tr]))
        for p in te:
            m, t = int(obs.molecule[p]), int(obs.task[p])
            out[p] = float(K[m] @ f.coef[:, t]) + held_out_bias(f.bias, trained, t)
    return out


# ----------------------------------------------------------------------------- the baselines


def additive_fit(obs: MT.Observations, train: np.ndarray,
                 factors: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], float]:
    """``y = mu + sum of one effect per factor`` by least squares, effects centred over the
    levels seen in training, so that an unseen level (effect zero) is predicted at the mean."""
    cols = [int(f.max()) + 1 for f in factors]
    X = np.zeros((len(train), 1 + sum(cols)))
    X[:, 0] = 1.0
    start = 1
    for f, n in zip(factors, cols):
        X[np.arange(len(train)), start + f[train]] = 1.0
        start += n
    w = np.linalg.lstsq(X, obs.y[train], rcond=None)[0]
    mu = float(w[0])
    effects = []
    start = 1
    for f, n in zip(factors, cols):
        e = w[start:start + n].copy()
        seen = np.unique(f[train])
        mu += float(e[seen].mean())
        e -= e[seen].mean()
        e[np.setdiff1d(np.arange(n), seen)] = 0.0
        effects.append(e)
        start += n
    return effects, mu


def additive_cv(obs: MT.Observations, folds: Sequence[np.ndarray],
                three_factor: bool = False) -> np.ndarray:
    """Out-of-fold predictions of ``catalyst + endpoint`` or of ``catalyst + imine + thiol``."""
    if three_factor:
        factors = [obs.molecule, imine_of(obs.task), thiol_of(obs.task)]
    else:
        factors = [obs.molecule, obs.task]
    out = np.full(len(obs), np.nan)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(obs)), te)
        effects, mu = additive_fit(obs, tr, factors)
        out[te] = mu + sum(e[f[te]] for e, f in zip(effects, factors))
    return out


def folds_by_catalyst(obs: MT.Observations) -> List[np.ndarray]:
    return [np.where(obs.molecule == m)[0] for m in range(obs.n_mol)]


def folds_by_tasks(obs: MT.Observations, groups: Sequence[Sequence[int]]) -> List[np.ndarray]:
    return [np.where(np.isin(obs.task, list(g)))[0] for g in groups]


def folds_random(obs: MT.Observations, folds: int = 10, seed: int = SEED) -> List[np.ndarray]:
    order = np.random.RandomState(seed).permutation(len(obs))
    return [np.sort(g) for g in np.array_split(order, folds)]


# -------------------------------------------------------------------------------- the scores


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    ok = np.isfinite(p)
    return {"n": int(ok.sum()), "mad": float(mae(y[ok], p[ok])), "rmse": float(rmse(y[ok], p[ok])),
            "q2": float(r2_score(y[ok], p[ok]))}


def bootstrap_catalysts(obs: MT.Observations, a: np.ndarray, b: np.ndarray,
                        runs: int = RUNS, seed: int = SEED) -> dict:
    """The difference in MAD and in q2 (`a` minus `b`), resampling catalysts with replacement."""
    rng = np.random.RandomState(seed)
    by_mol = [np.where(obs.molecule == m)[0] for m in range(obs.n_mol)]
    d_mad, d_q2 = [], []
    for _ in range(int(runs)):
        pick = rng.randint(0, obs.n_mol, obs.n_mol)
        idx = np.concatenate([by_mol[m] for m in pick])
        y, pa, pb = obs.y[idx], a[idx], b[idx]
        ok = np.isfinite(pa) & np.isfinite(pb)
        if ok.sum() < 4 or np.std(y[ok]) <= 0:
            continue
        d_mad.append(mae(y[ok], pa[ok]) - mae(y[ok], pb[ok]))
        d_q2.append(r2_score(y[ok], pa[ok]) - r2_score(y[ok], pb[ok]))
    return {"mad_difference": float(np.mean(d_mad)), "mad_low": float(np.percentile(d_mad, 2.5)),
            "mad_high": float(np.percentile(d_mad, 97.5)),
            "q2_difference": float(np.mean(d_q2)), "q2_low": float(np.percentile(d_q2, 2.5)),
            "q2_high": float(np.percentile(d_q2, 97.5))}


def report_line(label: str, s: dict, base: Optional[dict] = None, boot: Optional[dict] = None) -> str:
    txt = f"    {label:34s} MAD {s['mad']:.3f}  RMSE {s['rmse']:.3f}  q2 {s['q2']:+.3f}"
    if base is not None:
        txt += f"   (baseline MAD {base['mad']:.3f}, q2 {base['q2']:+.3f})"
    if boot is not None:
        txt += (f"   MAD difference {boot['mad_difference']:+.3f} "
                f"[{boot['mad_low']:+.3f}, {boot['mad_high']:+.3f}]")
    return txt


# ------------------------------------------------------------------------------ the co-field


def cofield_slice(model: CMFModel, coef: np.ndarray, f: int, pts: np.ndarray) -> np.ndarray:
    """``h_f c_f C_f`` at the points, for the dual coefficients `coef` over the training set."""
    model.coef = np.asarray(coef, float)
    centres, coeffs, widths = model.cofield_coefficients(f)
    v = evaluate_gaussians(pts, centres, coeffs, widths)
    return v * model.fieldset.specs[f].h * model.scale[f]


def plot_cofields(model: CMFModel, coef: np.ndarray, skeleton: np.ndarray, fields: Sequence[str],
                  path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    planes = [("xy", 2.5), ("xy", 4.5), ("xz", 0.0)]
    xs = np.arange(-9.0, 9.0, 0.25)
    ys = np.arange(-9.0, 9.0, 0.25)
    zs = np.arange(-4.0, 10.0, 0.25)
    fig, axes = plt.subplots(len(fields), len(planes), figsize=(3.9 * len(planes), 3.4 * len(fields)),
                             squeeze=False)
    for i, fname in enumerate(fields):
        f = model.fieldset.names.index(fname)
        slices = []
        for kind, level in planes:
            if kind == "xy":
                X, Y = np.meshgrid(xs, ys, indexing="xy")
                pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, level)])
                shape, extent = X.shape, (xs[0], xs[-1], ys[0], ys[-1])
            else:
                X, Zc = np.meshgrid(xs, zs, indexing="xy")
                pts = np.column_stack([X.ravel(), np.full(X.size, level), Zc.ravel()])
                shape, extent = X.shape, (xs[0], xs[-1], zs[0], zs[-1])
            slices.append((cofield_slice(model, coef, f, pts).reshape(shape), extent, kind, level))
        vmax = max(np.abs(s[0]).max() for s in slices) or 1.0
        for j, (v, extent, kind, level) in enumerate(slices):
            ax = axes[i, j]
            im = ax.imshow(v, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="equal")
            if kind == "xy":
                ax.scatter(skeleton[:, 0], skeleton[:, 1], s=6, color="black", alpha=0.6)
                ax.set_xlabel("x / A", fontsize=8)
                ax.set_ylabel("y / A", fontsize=8)
                ax.set_title(f"{model.fieldset.specs[f].label}, z = {level:g} A", fontsize=9)
            else:
                ax.scatter(skeleton[:, 0], skeleton[:, 2], s=6, color="black", alpha=0.6)
                ax.set_xlabel("x / A", fontsize=8)
                ax.set_ylabel("z / A", fontsize=8)
                ax.set_title(f"{model.fieldset.specs[f].label}, y = {level:g} A", fontsize=9)
            ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(panels: Sequence[Tuple[str, np.ndarray, np.ndarray]], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.8), squeeze=False)
    for ax, (label, y, p) in zip(axes[0], panels):
        ok = np.isfinite(p)
        ax.scatter(y[ok], p[ok], s=8, alpha=0.6, color="tab:blue")
        lo, hi = min(y.min(), p[ok].min()), max(y.max(), p[ok].max())
        ax.plot([lo, hi], [lo, hi], color="gray", lw=0.8)
        s = scores(y, p)
        ax.set_title(f"{label}: MAD {s['mad']:.3f}, q2 {s['q2']:+.2f}", fontsize=9)
        ax.set_xlabel("measured ddG / kcal/mol", fontsize=8)
        ax.set_ylabel("cross-validated prediction", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------- the run


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--single", action="store_true", help="one C2 image per catalyst, not both")
    ap.add_argument("--quick", action="store_true", help="a coarser grid and fewer bootstrap runs")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    alphas = [0.5, 1.2] if a.quick else ALPHAS
    lams = [1e-2, 1.0] if a.quick else LAMS
    runs = 100 if a.quick else RUNS
    t0 = time.time()

    sdf = os.path.join(DATA, "catalysts.sdf" if a.single else "catalysts_sym.sdf")
    mols, names = load_catalysts(sdf, ensemble=not a.single)
    rows = read_reactions(os.path.join(DATA, "reactions.csv"))
    known = set(names)
    missing = sorted({r["catalyst"] for r in rows} - known)
    if missing:
        print(f"WARNING: {len(missing)} catalysts of the table are not in {sdf}; their reactions "
              f"are left out: {' '.join(missing)}")
        rows = [r for r in rows if r["catalyst"] in known]
    obs = observations(rows, names)
    y = obs.y
    print(f"{len(names)} catalysts, {obs.n_tasks} endpoints, {len(obs)} reactions; "
          f"ddG mean {y.mean():.3f}, sd {y.std():.3f} kcal/mol", flush=True)
    results = {"n_catalysts": len(names), "n_tasks": obs.n_tasks, "n": len(obs),
               "sd": float(y.std()), "ensemble": not a.single, "alphas": alphas, "lams": lams,
               "published": PUBLISHED, "splits": {}, "fields": {}}

    # ---- the kernels for every alpha, five fields together
    kernels, models, tables = {}, {}, {}
    for alpha in alphas:
        K, model, table = catalyst_model(mols, names, PHYSCHEM, alpha, ensemble=not a.single)
        kernels[alpha], models[alpha], tables[alpha] = K, model, table
    print(f"kernels for {len(alphas)} alphas ({time.time() - t0:.0f} s)", flush=True)

    # ---- the two covariances, by the evidence
    one = choose_rho(kernels, obs, lams)
    print(f"one correlation: alpha {one['alpha']:g}, lambda {one['lam']:g}, rho {one['rho']:+.3f} "
          f"(evidence {one['evidence']:.1f})", flush=True)
    desc = choose_descriptor(kernels, obs, task_descriptors(), lams)
    print(f"substrate covariance: alpha {desc['alpha']:g}, lambda {desc['lam']:g}, length scale "
          f"{desc['length_scale']:g}, nugget {desc['nugget']:g} (evidence {desc['evidence']:.1f})",
          flush=True)
    results["one_rho"] = {k: v for k, v in one.items() if k != "T"}
    results["substrate_covariance"] = {k: v for k, v in desc.items() if k != "T"}
    K1, Kd = kernels[one["alpha"]], kernels[desc["alpha"]]

    # ---- the splits
    splits = {}
    groups_tasks = [[t] for t in range(obs.n_tasks)]
    groups_imines = [[t for t in range(obs.n_tasks) if obs.task_names[t].startswith(i + "_")]
                     for i in IMINES]
    groups_thiols = [[t for t in range(obs.n_tasks) if obs.task_names[t].endswith("_" + u)]
                     for u in THIOLS]

    print("leave-one-catalyst-out", flush=True)
    p_one = cv_catalysts(K1, obs, one["T"], one["lam"])
    p_desc = cv_catalysts(Kd, obs, desc["T"], desc["lam"])
    base = additive_cv(obs, folds_by_catalyst(obs))
    splits["catalysts"] = {"baseline": "endpoint mean over the training catalysts",
                           "published": "external catalysts", "models": {}}
    for label, p in (("one correlation", p_one), ("substrate covariance", p_desc)):
        s = scores(y, p)
        b = bootstrap_catalysts(obs, p, base, runs=runs)
        splits["catalysts"]["models"][label] = {**s, "bootstrap": b, "predicted": p.tolist()}
        print(report_line(label, s, scores(y, base), b), flush=True)
    splits["catalysts"]["baseline_scores"] = scores(y, base)

    print("10-fold over the reactions", flush=True)
    folds = folds_random(obs)
    base = additive_cv(obs, folds)
    splits["reactions"] = {"baseline": "catalyst effect + substrate effect, least squares",
                           "published": "new substrate combinations", "models": {}}
    for label, K, o, T, lam in (("one correlation", K1, obs, one["T"], one["lam"]),
                                ("substrate covariance", Kd, obs, desc["T"], desc["lam"])):
        p = cv_reactions(K, o, T, lam)
        s = scores(y, p)
        b = bootstrap_catalysts(obs, p, base, runs=runs)
        splits["reactions"]["models"][label] = {**s, "bootstrap": b, "predicted": p.tolist()}
        print(report_line(label, s, scores(y, base), b), flush=True)
    splits["reactions"]["baseline_scores"] = scores(y, base)

    for name, groups in (("substrates", groups_tasks), ("imines", groups_imines),
                         ("thiols", groups_thiols)):
        print(f"leave-one-{name[:-1]}-out", flush=True)
        base = additive_cv(obs, folds_by_tasks(obs, groups), three_factor=True)
        p = cv_tasks(Kd, obs, desc["T"], desc["lam"], groups)
        s = scores(y, p)
        b = bootstrap_catalysts(obs, p, base, runs=runs)
        splits[name] = {"baseline": "catalyst effect + imine effect + thiol effect, least squares",
                        "published": "new substrate combinations" if name == "substrates" else "",
                        "models": {"substrate covariance": {**s, "bootstrap": b,
                                                            "predicted": p.tolist()}},
                        "baseline_scores": scores(y, base)}
        print(report_line("substrate covariance", s, scores(y, base), b), flush=True)
    results["splits"] = splits

    # ---- which field carries it: the catalyst holdout, one field at a time, and no frame
    print("field by field (leave-one-catalyst-out, one correlation)", flush=True)
    for fname in PHYSCHEM:
        ks = {}
        for alpha in alphas:
            K, _, _ = catalyst_model(mols, names, [fname], alpha, ensemble=not a.single)
            ks[alpha] = K
        sel = choose_rho(ks, obs, lams)
        p = cv_catalysts(ks[sel["alpha"]], obs, sel["T"], sel["lam"])
        s = scores(y, p)
        results["fields"][fname] = {**s, "alpha": sel["alpha"], "lam": sel["lam"], "rho": sel["rho"]}
        print(report_line(fname, s), flush=True)

    # ---- the same five fields without a frame.  The autocorrelation kernel does not depend on
    # the attenuation; its own setting is the width along the distance axis, and every width is
    # reported, so that the comparison with the frame does not rest on one choice of it
    ks, per_width, held_out = {}, {}, {}
    for width in AC_WIDTHS:
        ks[width], _, _ = catalyst_model(mols, names, PHYSCHEM, alphas[0], ensemble=not a.single,
                                         source="autocorrelation", ac_sigma=width)
        sel = choose_rho({width: ks[width]}, obs, lams)
        held_out[width] = cv_catalysts(ks[width], obs, sel["T"], sel["lam"])
        per_width[width] = {**scores(y, held_out[width]), "lam": sel["lam"], "rho": sel["rho"],
                            "evidence": sel["evidence"]}
        print(report_line(f"invariant, width {width:g} A", per_width[width]), flush=True)
    chosen = max(per_width, key=lambda w: per_width[w]["evidence"])     # by the evidence
    results["fields"]["invariant"] = {**{k: v for k, v in per_width[chosen].items() if k != "evidence"},
                                      "alpha": None, "width": float(chosen)}
    framed = np.array(splits["catalysts"]["models"]["one correlation"]["predicted"])
    results["invariant"] = {
        "d_max": AC_D_MAX, "chosen_width": float(chosen),
        "widths": {f"{w:g}": {**per_width[w],
                              "frame_minus_this": bootstrap_catalysts(obs, framed, held_out[w],
                                                                      runs=runs)}
                   for w in AC_WIDTHS}}
    b = results["invariant"]["widths"][f"{chosen:g}"]["frame_minus_this"]
    print(f"    the width with the highest evidence is {chosen:g} A; five fields in the frame minus "
          f"it: q2 {b['q2_difference']:+.3f} [{b['q2_low']:+.3f}, {b['q2_high']:+.3f}]", flush=True)

    # ---- the co-field, averaged over the endpoints, in the acid's frame
    model, table = models[one["alpha"]], tables[one["alpha"]]
    y_mean = np.array([y[obs.molecule == m].mean() for m in range(obs.n_mol)])
    model.params["lam"] = one["lam"]
    model.fit(table, y_mean, cv_folds=0)
    full = MT.fit(K1, obs, lam=one["lam"], T=one["T"])
    coef_mean = full.coef.mean(axis=1)
    skeleton = np.array([mols[0].GetConformer().GetAtomPosition(i)
                         for i in range(mols[0].GetNumAtoms())
                         if mols[0].GetAtomWithIdx(i).GetAtomicNum() > 1], float)
    plot_cofields(model, coef_mean, skeleton, ["steric", "electrostatic"],
                  os.path.join(a.out_dir, "cofields.png"),
                  "Co-field of the selectivity, mean over the 25 endpoints "
                  "(red: raises ddG per unit of field; dots: catalyst 1_i)")
    panels = [("catalysts held out", y, np.array(splits["catalysts"]["models"]["one correlation"]["predicted"])),
              ("reactions held out", y, np.array(splits["reactions"]["models"]["one correlation"]["predicted"])),
              ("substrates held out", y, np.array(splits["substrates"]["models"]["substrate covariance"]["predicted"]))]
    plot_scatter(panels, os.path.join(a.out_dir, "cv_scatter.png"))

    results["seconds"] = time.time() - t0
    with open(os.path.join(a.out_dir, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    write_table(results, os.path.join(a.out_dir, "table.md"))
    print(f"done in {results['seconds']:.0f} s; results in {a.out_dir}")
    return 0


def write_table(res: dict, path: str) -> None:
    lines = ["# A kinetic selectivity as the activity: the Denmark catalyst series", "",
             f"{res['n_catalysts']} catalysts x {res['n_tasks']} substrate combinations = "
             f"{res['n']} reactions; ddG sd {res['sd']:.3f} kcal/mol. "
             f"{'Both C2 images' if res['ensemble'] else 'One C2 image'} per catalyst. "
             f"Published: MAD {res['published']['external catalysts']:.3f} kcal/mol on external "
             f"catalysts, {res['published']['new substrate combinations']:.3f} on new substrate "
             "combinations (Zahrt et al. 2019, a 600/475 design).", "",
             "| Held out | Model | MAD | RMSE | q2 | Baseline MAD | Baseline q2 | MAD difference [95 %] |",
             "|---|---|---|---|---|---|---|---|"]
    for split, d in res["splits"].items():
        for label, s in d["models"].items():
            b = s["bootstrap"]
            lines.append(f"| {split} | {label} | {s['mad']:.3f} | {s['rmse']:.3f} | {s['q2']:+.3f} | "
                         f"{d['baseline_scores']['mad']:.3f} | {d['baseline_scores']['q2']:+.3f} | "
                         f"{b['mad_difference']:+.3f} [{b['mad_low']:+.3f}, {b['mad_high']:+.3f}] |")
    lines += ["", "Baselines: " + "; ".join(f"{k}: {v['baseline']}" for k, v in res["splits"].items()), "",
              "## Which field carries the selectivity (catalysts held out, one correlation)", "",
              "| Kernel | MAD | RMSE | q2 | alpha | lambda | rho |", "|---|---|---|---|---|---|---|"]
    for f, s in res["fields"].items():
        alpha = "-" if s["alpha"] is None else f"{s['alpha']:g}"
        label = f if "width" not in s else f"{f} (no frame; width {s['width']:g} A, see below)"
        lines.append(f"| {label} | {s['mad']:.3f} | {s['rmse']:.3f} | {s['q2']:+.3f} | {alpha} | "
                     f"{s['lam']:g} | {s['rho']:+.3f} |")
    inv = res.get("invariant")
    if inv:
        lines += ["", "## The same five fields without a frame, width by width", "",
                  "The rotation-invariant kernel (the autocorrelation of the fields) compares two "
                  "catalysts by the distances between their atoms. It does not depend on the "
                  "attenuation of the fields. Its own setting is the width along the distance "
                  f"axis; distances are resolved up to {inv['d_max']:g} A. The ridge and the "
                  "correlation of the endpoints are chosen by the evidence for every width, and "
                  "the row of the table above is the width with the highest evidence "
                  f"({inv['chosen_width']:g} A). The last column is the q2 of the five fields in "
                  "the frame minus the q2 of this row, with a bootstrap interval over the "
                  "catalysts.", "",
                  "| Width, A | MAD | RMSE | q2 | lambda | rho | evidence | frame minus this, q2 [95 %] |",
                  "|---|---|---|---|---|---|---|---|"]
        for w, s in inv["widths"].items():
            b = s["frame_minus_this"]
            lines.append(f"| {w} | {s['mad']:.3f} | {s['rmse']:.3f} | {s['q2']:+.3f} | {s['lam']:g} | "
                         f"{s['rho']:+.3f} | {s['evidence']:.1f} | {b['q2_difference']:+.3f} "
                         f"[{b['q2_low']:+.3f}, {b['q2_high']:+.3f}] |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
