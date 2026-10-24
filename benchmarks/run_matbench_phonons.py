#!/usr/bin/env python
"""
Continuous molecular fields of crystals on a Matbench task: ``matbench_phonons``.

1,265 inorganic crystals from the Materials Project with the frequency of the highest peak of
the phonon density of states (cm^-1) computed by DFPT (Petretto et al. 2018), one of the
thirteen tasks of Matbench v0.1 (Dunn et al., npj Comput. Mater. 6 (2020) 138).  A property set
by the stiffest bond in the cell, which is what a typed pair-distance kernel should see.

The model is the periodic field autocorrelation of ``cmfqsar/periodic.py``: the distance
distributions of the field weights over the neighbour list of the cell, per atom, with the
uniform background removed and a tapered cutoff; kernel ridge regression on the cosine-normalised
linear kernel of those channels.  The fields are the electrostatic (oxidation states), steric
(atomic volume) and electronegativity fields and one continuous indicator field per element that
occurs in at least ``--min-structures`` structures, plus one for the rest.

Protocol: the Matbench split, five folds of ``KFold(shuffle=True, random_state=18012019)`` over
the structures in file order; inside every training fold the width/spacing of the channels and
the regularisation are chosen by 10-fold cross-validation; the metric is the mean absolute error
on the held-out folds, mean and standard deviation over the five, as the Matbench leaderboard
reports it (https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_phonons/).

Variants
--------
    elements   the indicator fields only: typed partial pair distributions, the baseline of
               Schuett et al. (2014) and the Valle-Oganov fingerprint in kernel form
    physchem   the three physico-chemical fields only
    physchem+mass  the three plus the inverse square root of the atomic mass, the field of the
               light atoms, since the top of a phonon spectrum goes as 1/sqrt(m)
    all        the indicator and the physico-chemical fields (the model)
    all+mass   the same with the mass field
    all-nobg   as "all" without the background subtraction (control)
    all-hard   as "all" with a hard cutoff (control)
    all-log    as "all" on the logarithm of the target, predictions back-transformed
    all-gauss  as "all" with a Gaussian kernel layer on top (nonlinear headroom; no co-pattern)
    physchem+mass-gauss, all+mass-gauss  the same layer on the sets with the mass field

The data file is downloaded once to ``benchmarks/cache/matbench`` (not part of the repository);
results go to ``benchmarks/results/matbench_phonons``.

Usage
-----
    python benchmarks/run_matbench_phonons.py [--variants all,elements] [--force]
    python benchmarks/run_matbench_phonons.py --report
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar.invariant import autocorrelation_kernel, channel_scale, field_pairs  # noqa: E402
from cmfqsar.model import _Learner, cv_predict, r2_score, rmse  # noqa: E402
from cmfqsar.periodic import (INORGANIC_FIELDS, PeriodicSpec, crystal_fields,  # noqa: E402
                              elements_in, inorganic_fieldset, periodic_block_sizes,
                              periodic_channel_labels, periodic_feature_matrix,
                              structures_from_matbench)

URL = "https://ml.materialsproject.org/projects/matbench_phonons.json.gz"
CACHE = os.path.join(HERE, "cache", "matbench")
RESULTS = os.path.join(HERE, "results", "matbench_phonons")
SPLIT_SEED = 18012019                   # the seed of the Matbench v0.1 KFold
OUTER, INNER = 5, 10
LAMS = (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
SIGMAS = (0.35, 0.5, 0.75)
GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0)     # Gaussian layer, relative to the median squared distance
R_MAX, TAPER = 10.0, 2.0

MASS = INORGANIC_FIELDS + ("prop:inverse_sqrt_mass",)
VARIANTS = {
    "elements": dict(physchem=(), elements=True),
    "physchem": dict(physchem=INORGANIC_FIELDS, elements=False),
    "physchem+mass": dict(physchem=MASS, elements=False),
    "all": dict(physchem=INORGANIC_FIELDS, elements=True),
    "all+mass": dict(physchem=MASS, elements=True),
    "all-nobg": dict(physchem=INORGANIC_FIELDS, elements=True, background=False),
    "all-hard": dict(physchem=INORGANIC_FIELDS, elements=True, taper=0.0),
    "all-log": dict(physchem=INORGANIC_FIELDS, elements=True, log=True),
    "all-gauss": dict(physchem=INORGANIC_FIELDS, elements=True, gauss=True),
    "physchem+mass-gauss": dict(physchem=MASS, elements=False, gauss=True),
    "all+mass-gauss": dict(physchem=MASS, elements=True, gauss=True),
}


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, os.path.basename(URL))
    if not os.path.exists(path):
        print(f"Downloading {URL} ...")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (cmfqsar benchmark)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as fh:
            fh.write(r.read())
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def matbench_folds(n: int):
    """The five Matbench folds: sklearn's shuffled KFold with the Matbench seed."""
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=OUTER, shuffle=True, random_state=SPLIT_SEED).split(np.arange(n)))


def gauss_layer(K: np.ndarray, K_tr: np.ndarray, gamma: float, scale: float) -> np.ndarray:
    """exp(-gamma D^2 / scale) on cosine-normalised kernels, where D^2 = 2 - 2K."""
    return np.exp(-gamma * (2.0 - 2.0 * K) / scale)


def median_d2(K_tr: np.ndarray) -> float:
    d2 = 2.0 - 2.0 * K_tr[np.triu_indices_from(K_tr, k=1)]
    m = float(np.median(d2))
    return m if m > 1e-12 else 1.0


def features_for(atoms, names, fs, spec, progress=True):
    crystals = [crystal_fields(a, fs, nm) for a, nm in zip(atoms, names)]
    t0 = time.time()
    F = periodic_feature_matrix(crystals, spec, n_fields=fs.n_fields,
                                progress=(lambda m, f: print(f"\r  {m:<50}", end="", flush=True)) if progress else None)
    if progress:
        print(f"\r  features {F.shape}, {time.time() - t0:.0f} s" + " " * 20)
    return F, crystals


def columns_for(fs_full, fs_sub, n_bins):
    """The columns of the full feature matrix that make up the feature matrix of a subset of its
    fields: the pair blocks of the subset, in the subset's own pair order."""
    names = [sp.name for sp in fs_full.specs]
    idx = [names.index(sp.name) for sp in fs_sub.specs]
    pos = {fg: p for p, fg in enumerate(field_pairs(len(names)))}
    cols = []
    for a, b in field_pairs(len(idx)):
        f, g = idx[a], idx[b]
        p = pos[(f, g)] if (f, g) in pos else pos[(g, f)]
        cols.extend(range(p * n_bins, (p + 1) * n_bins))
    return np.array(cols, int)


def run_variant(tag, y, names, elements, args, fs_full, feats_full, specs):
    cfg = VARIANTS[tag]
    fs = inorganic_fieldset(elements if cfg["elements"] else [], physchem=cfg["physchem"],
                            charge_model=args.charges)
    log, gauss = cfg.get("log", False), cfg.get("gauss", False)
    print(f"\n=== {tag}: {fs.n_fields} fields" + (", log target" if log else "")
          + (", Gaussian layer" if gauss else ""))
    feats = {s: feats_full[s][:, columns_for(fs_full, fs, specs[s].n_bins)] for s in specs}
    t = np.log(y) if log else y.copy()
    folds = matbench_folds(len(y))
    pred = np.full(len(y), np.nan)
    chosen, per_fold = [], []
    t0 = time.time()
    for k, (tr, te) in enumerate(folds):
        best = None
        for s, F in feats.items():
            blocks = periodic_block_sizes(specs[s], fs.n_fields)
            sc = channel_scale(F[tr], specs[s].n_bins, blocks=blocks)
            K_tr = autocorrelation_kernel(F[tr], scale=sc, n_bins=specs[s].n_bins, blocks=blocks)
            gammas = GAMMAS if gauss else (None,)
            med = median_d2(K_tr) if gauss else None
            for g in gammas:
                Kg = gauss_layer(K_tr, K_tr, g, med) if gauss else K_tr
                for lam in LAMS:
                    p = cv_predict(Kg, t[tr], "krr", {"lam": lam}, folds=INNER, seed=0)
                    err = rmse(t[tr], p)
                    if best is None or err < best[0]:
                        best = (err, s, g, lam, sc, blocks, med)
        err, s, g, lam, sc, blocks, med = best
        F = feats[s]
        K_tr = autocorrelation_kernel(F[tr], scale=sc, n_bins=specs[s].n_bins, blocks=blocks)
        K_te = autocorrelation_kernel(F[tr], F[te], scale=sc, n_bins=specs[s].n_bins, blocks=blocks)
        if gauss:
            K_tr, K_te = gauss_layer(K_tr, K_tr, g, med), gauss_layer(K_te, K_tr, g, med)
        learner = _Learner("krr", {"lam": lam}, True).fit(K_tr, t[tr])
        p = learner.predict(K_te)
        pred[te] = np.exp(p) if log else p
        mae = float(np.mean(np.abs(pred[te] - y[te])))
        per_fold.append({"fold": k, "sigma": s, "gamma": g, "lam": lam, "inner_rmse": float(err),
                         "mae": mae, "rmse": float(rmse(y[te], pred[te])), "n_test": int(len(te))})
        chosen.append((s, g, lam))
        print(f"  fold {k}: sigma {s:g}, lam {lam:g}" + (f", gamma {g:g}" if gauss else "")
              + f"  ->  MAE {mae:.1f} cm-1  ({time.time() - t0:.0f} s)")
    maes = np.array([f["mae"] for f in per_fold])
    out = {
        "variant": tag, "n": int(len(y)), "fields": [sp.label for sp in fs.specs],
        "n_fields": fs.n_fields, "log_target": log, "gauss_layer": gauss,
        "r_max": R_MAX, "taper": cfg.get("taper", TAPER), "background": cfg.get("background", True),
        "sigmas": list(SIGMAS), "lams": list(LAMS), "gammas": list(GAMMAS) if gauss else [],
        "mae_mean": float(maes.mean()), "mae_sd": float(maes.std(ddof=0)),
        "rmse_pooled": float(rmse(y, pred)), "r2_pooled": float(r2_score(y, pred)),
        "mae_pooled": float(np.mean(np.abs(pred - y))),
        "per_fold": per_fold, "elapsed_s": time.time() - t0,
        "names": list(names), "y": [float(v) for v in y], "y_pred": [float(v) for v in pred],
    }
    print(f"  MAE {maes.mean():.1f} +/- {maes.std():.1f} cm-1 over the five folds; "
          f"pooled RMSE {out['rmse_pooled']:.1f}, R2 {out['r2_pooled']:.3f}")
    # the co-pattern: the model on every structure with the settings the folds chose most often
    if not gauss:
        s, _, lam = max(set(chosen), key=chosen.count)
        F, spec = feats[s], specs[s]
        blocks = periodic_block_sizes(spec, fs.n_fields)
        sc = channel_scale(F, spec.n_bins, blocks=blocks)
        K = autocorrelation_kernel(F, scale=sc, n_bins=spec.n_bins, blocks=blocks)
        learner = _Learner("krr", {"lam": lam}, True).fit(K, t)
        Z = F * sc
        Z = Z / np.sqrt(np.maximum(np.sum(Z * Z, axis=1, keepdims=True), 1e-300))
        w = Z.T @ learner.alpha_
        sd = Z.std(axis=0)
        labels = periodic_channel_labels(fs, spec)
        order = np.argsort(-np.abs(w) * sd)
        top = [{"channel": labels[c], "weight": float(w[c]), "spread": float(sd[c]),
                "effect": float(w[c] * sd[c])} for c in order[:25]]
        out["co_pattern"] = {"sigma": s, "lam": lam, "top": top}
        print("  co-pattern (weight x spread, the channels that move the prediction most):")
        for row in top[:12]:
            print(f"    {row['effect']:+9.2f}  {row['channel']}")
    return out


def report(path):
    rows = []
    for tag in VARIANTS:
        f = os.path.join(RESULTS, f"{tag}.json")
        if os.path.exists(f):
            rows.append(json.load(open(f)))
    lines = ["# Continuous molecular fields of crystals: Matbench phonons", "",
             "Last peak of the phonon density of states (cm^-1), 1,265 structures, the five Matbench "
             "folds; MAE mean +/- sd over the folds, RMSE and R2 on the pooled held-out predictions. "
             "Hyper-parameters (sigma, lambda, gamma) chosen by 10-fold CV inside every training fold.",
             "", "| variant | fields | MAE (cm^-1) | RMSE | R2 | sigma chosen | time (s) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        sig = ", ".join(f"{f['sigma']:g}" for f in r["per_fold"])
        lines.append(f"| {r['variant']} | {r['n_fields']} | {r['mae_mean']:.1f} +/- {r['mae_sd']:.1f} | "
                     f"{r['rmse_pooled']:.1f} | {r['r2_pooled']:.3f} | {sig} | {r['elapsed_s']:.0f} |")
    lines += ["", "Leaderboard reference (matbench v0.1, MAE in cm^-1, from the Matbench site at the "
              "time of writing, October 2026): dummy 324.0; RF-SCM/Magpie 67.6; CGCNN 57.8; CrabNet 55.1; "
              "SchNet 39.0; MODNet 34.3; coGN 29.7; ALIGNN 29.5; coNGN 28.9; MegNet 28.8.", ""]
    for r in rows:
        if "co_pattern" in r:
            lines += [f"## Co-pattern of `{r['variant']}` (sigma {r['co_pattern']['sigma']:g} A, "
                      f"lambda {r['co_pattern']['lam']:g})", "",
                      "Weight of each channel in the linear model times its spread over the set: the "
                      "channels that move the prediction most, sign included.", "",
                      "| effect | channel |", "|---|---|"]
            lines += [f"| {t['effect']:+.2f} | {t['channel']} |" for t in r["co_pattern"]["top"][:15]]
            lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Report written to {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated subset of "
                    + ", ".join(VARIANTS))
    ap.add_argument("--min-structures", type=int, default=10,
                    help="an element gets an indicator field when it occurs in this many structures")
    ap.add_argument("--charges", default="oxidation", choices=["oxidation", "electronegativity"])
    ap.add_argument("--force", action="store_true", help="recompute variants that have results")
    ap.add_argument("--report", action="store_true", help="only write the Markdown table")
    args = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    if args.report:
        report(os.path.join(RESULTS, "table.md"))
        return 0
    table = fetch()
    atoms, y, names = structures_from_matbench(table)
    elements = elements_in(atoms, min_structures=args.min_structures)
    print(f"{len(atoms)} structures, {sum(len(a) for a in atoms)} atoms, target {y.min():.0f}-"
          f"{y.max():.0f} cm-1 (median {np.median(y):.0f}); {len(elements)} elements in at least "
          f"{args.min_structures} structures: {' '.join(elements)}")
    todo = [v for v in args.variants.split(",") if v]
    groups = {}
    for tag in todo:
        out_path = os.path.join(RESULTS, f"{tag}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"\n=== {tag}: results exist, skipped (--force to redo)")
            continue
        cfg = VARIANTS[tag]
        groups.setdefault((cfg.get("background", True), cfg.get("taper", TAPER)), []).append(tag)
    fs_full = inorganic_fieldset(elements, physchem=MASS, charge_model=args.charges)
    for (background, taper), tags in groups.items():
        # the features of the full field set once per setting; every variant is a column subset
        specs = {s: PeriodicSpec(r_max=R_MAX, step=s, sigma=s, taper=taper, background=background)
                 for s in SIGMAS}
        print(f"\n--- features of all {fs_full.n_fields} fields, background "
              f"{'removed' if background else 'kept'}, taper {taper:g} A, for {', '.join(tags)}")
        feats_full = {}
        for s, spec in specs.items():
            print(f"  sigma = {s:g} A:", end="")
            feats_full[s] = features_for(atoms, names, fs_full, spec)[0]
        for tag in tags:
            out = run_variant(tag, y, names, elements, args, fs_full, feats_full, specs)
            with open(os.path.join(RESULTS, f"{tag}.json"), "w") as fh:
                json.dump(out, fh, indent=1)
        del feats_full
    report(os.path.join(RESULTS, "table.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
